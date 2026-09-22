# Bug candidate G: with dynamic shapes, an output that is a same-shape reshape/view alias of a graph input
# comes back with the wrong element order when the input was mutated through a transposed view.
import torch
print("torch", torch.__version__)

def g_reshape(t0):
    v2 = t0.transpose(1, 2)
    v3 = t0.reshape(t0.shape)
    v2.masked_fill_(v2 > 2, 5)
    return v3
def g_view(t0):
    v2 = t0.transpose(1, 2); v3 = t0.view(t0.shape); v2.masked_fill_(v2 > 2, 5); return v3
def g_alias_before_nomut(t0):
    v2 = t0.transpose(1, 2); v3 = t0.reshape(t0.shape); return v3                 # control: no mutation
def g_mut_direct(t0):
    v3 = t0.reshape(t0.shape); t0.masked_fill_(t0 > 2, 5); return v3               # control: mutate base directly
def g_after(t0):
    v2 = t0.transpose(1, 2); v2.masked_fill_(v2 > 2, 5); v3 = t0.reshape(t0.shape); return v3   # alias created after mutation
def g_2d(t0):
    v2 = t0.t(); v3 = t0.reshape(t0.shape); v2.masked_fill_(v2 > 2, 5); return v3  # 2-D version
def g_ret_input(t0):
    v2 = t0.transpose(1, 2); v2.masked_fill_(v2 > 2, 5); return t0                 # control: return input itself

def check(name, f, dynamic, shape=(2, 3, 4)):
    base = torch.arange(-12, 12, dtype=torch.int64).reshape(shape)
    e0 = base.clone(); re = f(e0)
    torch._dynamo.reset()
    c0 = base.clone(); rc = torch.compile(f, backend="aot_eager", dynamic=dynamic)(c0)
    ok_out, ok_in = torch.equal(re, rc), torch.equal(e0, c0)
    print(f"  {name:22s} dynamic={str(dynamic):5s} out {'OK' if ok_out else 'MISMATCH'} input {'OK' if ok_in else 'MISMATCH'}"
          f"  strides eager={re.stride()} compiled={rc.stride()}  aliases input: eager={re.untyped_storage().data_ptr()==e0.untyped_storage().data_ptr()} compiled={rc.untyped_storage().data_ptr()==c0.untyped_storage().data_ptr()}")
    if not ok_out:
        print("     eager   :", re.flatten().tolist()[:12]); print("     compiled:", rc.flatten().tolist()[:12])
for dyn in (None, True):
    for name, f in [("reshape(t0.shape)", g_reshape), ("view(t0.shape)", g_view), ("no mutation", g_alias_before_nomut), ("mutate base directly", g_mut_direct), ("alias after mutation", g_after), ("return input", g_ret_input)]:
        check(name, f, dyn)
    check("2-D t()+reshape", g_2d, dyn, shape=(4, 6))
