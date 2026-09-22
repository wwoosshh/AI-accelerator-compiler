# -*- coding: utf-8 -*-
"""Family L (found by aliasfuzz E2, 2026-09-22): an input-to-input copy observes a LATER in-place mutation of
its source when the source is mutated through a view.

    dst[:] = src            # copy src -> dst (both are graph inputs)
    v = src.reshape(-1)
    v.index_fill_(0, idx, -2)   # mutate src through the view, after the copy

Eager: dst holds src's values from before the fill. torch.compile(inductor): dst holds -2 at the filled positions,
i.e. the copy was performed after the mutation. aot_eager is correct -> Inductor. Static and dynamic shapes.
torch 2.14.0+cu130 and nightly 2.15.0.dev20260921 (with our patches 0001/0002/0004/0005 applied) both reproduce.
"""
import torch

IDX = [1, 11, 23]


def make(variant):
    def fn_reshape(dst, src):
        dst[0:, :] = src[0:, :]
        v = src.reshape(-1)
        v.index_fill_(0, torch.tensor(IDX, device=src.device), -2)
        return src

    def fn_view(dst, src):
        dst[0:, :] = src[0:, :]
        v = src.view(-1)
        v.index_fill_(0, torch.tensor(IDX, device=src.device), -2)
        return src

    def fn_direct(dst, src):  # no view: mutate src itself
        dst[0:, :] = src[0:, :]
        src.index_fill_(1, torch.tensor([1, 11], device=src.device), -2)
        return src

    def fn_copy_(dst, src):  # copy_ instead of setitem
        dst.copy_(src)
        v = src.reshape(-1)
        v.index_fill_(0, torch.tensor(IDX, device=src.device), -2)
        return src

    def fn_view_add_(dst, src):  # a different in-place op through the view
        dst[0:, :] = src[0:, :]
        v = src.reshape(-1)
        v.add_(100)
        return src
    return locals()[variant]


def run(fn, backend, dynamic, dtype):
    torch.manual_seed(0)
    dst = torch.zeros(2, 12, dtype=dtype, device="cuda")
    src = torch.randint(-8, 9, (2, 12), device="cuda").to(dtype)
    f = fn if backend is None else torch.compile(fn, backend=backend, dynamic=dynamic)
    f(dst, src)
    return dst, src


if __name__ == "__main__":
    print("torch", torch.__version__)
    for variant in ("fn_reshape", "fn_view", "fn_direct", "fn_copy_", "fn_view_add_"):
        for backend in ("aot_eager", "inductor"):
            for dynamic in (None, True):
                for dtype in (torch.int64, torch.float32):
                    torch._dynamo.reset()
                    fn = make(variant)
                    (d_e, s_e), (d_c, s_c) = run(fn, None, None, dtype), run(fn, backend, dynamic, dtype)
                    ok_dst, ok_src = torch.equal(d_e, d_c), torch.equal(s_e, s_c)
                    tag = "OK " if (ok_dst and ok_src) else "BUG"
                    print(f"{tag} {variant:13s} {backend:9s} dynamic={str(dynamic):5s} {str(dtype):14s} dst equal={ok_dst} src equal={ok_src}"
                          + ("" if ok_dst else f" | dst at {IDX}: eager {d_e.flatten()[IDX].tolist()} compiled {d_c.flatten()[IDX].tolist()}"))
