# run with: TORCH_LOGS=output_code python repro/dump_code_L.py > repro/inductor_code_L.txt 2>&1
import torch
def fn(dst, src):
    dst[0:, :] = src[0:, :]
    v = src.reshape(-1)
    v.index_fill_(0, torch.tensor([1, 11, 23], device=src.device), -2)
    return src
dst = torch.zeros(2, 12, dtype=torch.int64, device="cuda")
src = torch.arange(24, dtype=torch.int64, device="cuda").reshape(2, 12)
torch.compile(fn, backend="inductor")(dst, src)
print("dst after compiled call:", dst.flatten().tolist())
