# -*- coding: utf-8 -*-
"""Apply fix 0006 (remove_noop_ops must not replace a copy/clone by a source that aliases an input
mutated later in the graph) to a given torch/_inductor/fx_passes/post_grad.py. Idempotent."""
import sys
p = sys.argv[1]
s = open(p, encoding='utf-8', newline='').read()
if 'first_mutation_loc' in s:
    print('already applied:', p); sys.exit(0)
def rep(old, new):
    global s
    assert s.count(old) == 1, (old[:50], s.count(old))
    s = s.replace(old, new)
rep("""    for node in graph.nodes:
        if node.target in noop_registry:
""", """    # Graph inputs that are mutated inside this graph (the functionalization epilogue writes
    # them back with copy_). A non-view noop (aten.copy / aten.clone) whose source aliases such
    # an input must keep its copy if any of its users runs after the mutation, otherwise that
    # user observes the mutated input instead of the snapshot the copy represented, e.g.
    #   dst[0:, :] = src[0:, :]; src.add_(1)
    # lowered to `copy_(src, add); copy_(dst, src)` and copied the *updated* src into dst.
    node_order = {n: i for i, n in enumerate(graph.nodes)}
    first_mutation_loc: dict[int | None, int] = {}
    for n in graph.nodes:
        if n.target is torch.ops.aten.copy_.default and isinstance(n.args[0], torch.fx.Node):
            st = get_node_storage(n.args[0])
            if st is not None and st in input_storages:
                first_mutation_loc[st] = min(first_mutation_loc.get(st, node_order[n]), node_order[n])

    for node in graph.nodes:
        if node.target in noop_registry:
""")
rep("""            is_valid, args, kwargs = get_fake_args_kwargs(node)
            if not is_valid:
                continue
            if same_meta(node, src) and cond(*args, **kwargs):
""", """            # Keep a real copy of an input that is mutated before one of this node's users runs.
            if not node_is_view and src_storage in first_mutation_loc:
                mutation_loc = first_mutation_loc[src_storage]
                if any(node_order.get(u, mutation_loc + 1) > mutation_loc for u in node.users):
                    continue

            is_valid, args, kwargs = get_fake_args_kwargs(node)
            if not is_valid:
                continue
            if same_meta(node, src) and cond(*args, **kwargs):
""")
open(p, 'w', encoding='utf-8', newline='').write(s)
print('applied:', p)
