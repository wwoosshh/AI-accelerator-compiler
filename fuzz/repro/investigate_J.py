import torch
torch._logging.set_logs(recompiles=True)
def fn(t0, t1, t2, ta, k, lst, dct):
    v1 = t2.reshape(t2.shape)
    v2 = v1.reshape((24,))
    t2[:1, :8] = 5
    idx4 = torch.tensor([5], device=v2.device)
    v2.index_add_(0, idx4, ta.narrow(0, 0, 1))
    return (t2, lst, dct)
def mk(seed):
    g = torch.Generator().manual_seed(seed)
    t0 = torch.randint(-8, 9, (2, 12), generator=g).cuda(); t1 = torch.randint(-8, 9, (24,), generator=g).cuda(); t2 = torch.randint(-8, 9, (3, 8), generator=g).cuda()
    return t0, t1, t2
for variant in ["with_ta_alias", "ta_independent", "no_k_change"]:
    torch._dynamo.reset()
    cf = torch.compile(fn, backend="inductor", dynamic=True)
    print(f"--- {variant}")
    for call in range(3):
        t0, t1, t2 = mk(call)
        ta = t0.view(-1) if variant == "with_ta_alias" else torch.randint(-8, 9, (24,), generator=torch.Generator().manual_seed(100 + call)).cuda()
        k = (call % 5) - 2 if variant != "no_k_change" else 0
        e = fn(t0.clone(), t1.clone(), t2.clone(), (ta.clone() if variant != "with_ta_alias" else None), k, [], {}) if variant != "with_ta_alias" else None
        # eager reference (must rebuild alias for eager)
        et0 = t0.clone(); eta = et0.view(-1) if variant == "with_ta_alias" else ta.clone(); et2 = t2.clone()
        er = fn(et0, t1.clone(), et2, eta, k, [], {})
        ct0 = t0.clone(); cta = ct0.view(-1) if variant == "with_ta_alias" else ta.clone(); ct2 = t2.clone()
        cr = cf(ct0, t1.clone(), ct2, cta, k, [], {})
        print(f"  call{call}: t2 {'OK' if torch.equal(er[0], cr[0]) else 'MISMATCH'} diff_idx={(er[0].flatten()!=cr[0].flatten()).nonzero().flatten().tolist()}")
