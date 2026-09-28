"""Why the notebooks train RLT through a logits-only adapter under DataParallel.

torch DataParallel gathers replica outputs by recursing into them and dies
on non-tensor leaves ('int' object is not iterable). RLT's returned memories
hold a Python int step counter (TransformerMemories.step), so a bare wrapped
RLT crashes DP gather on multi-GPU runs while the plain Transformer (pure
tensor output) is fine. This test pins that premise against the real model
on CPU: if RLT ever stops returning int leaves, the adapter can go away.
"""
import torch

from RLT import RLT


def _leaves(obj):
    if isinstance(obj, (tuple, list)):
        for x in obj:
            yield from _leaves(x)
    elif isinstance(obj, dict):
        for x in obj.values():
            yield from _leaves(x)
    elif hasattr(obj, "_fields"):  # namedtuple
        yield from _leaves(tuple(obj))
    else:
        yield obj


def test_rlt_memories_contain_int_step_counter():
    torch.manual_seed(0)
    m = RLT(num_tokens=2, dim=32, enc_depth=1, dec_depth=1)
    m.eval()
    with torch.no_grad():
        out, mem = m(torch.randint(0, 2, (2, 12)))
    assert isinstance(out, torch.Tensor)
    int_leaves = [x for x in _leaves(mem) if isinstance(x, int)]
    assert int_leaves, "expected an int step counter in RLT memories"
