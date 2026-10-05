"""Activation-checkpointing policy contracts.

All four policies must agree on values and gradients; only *where* activations
live may differ. ``selective_op`` is checked behaviourally, by counting the
aten ops that really execute across forward + backward: a saved op runs once,
a recomputed one runs twice.
"""

from collections import Counter

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.checkpoint import noop_context_fn

from iris3b.models import IrisDiT, ac
from tiny_config import tiny_model_config


class OpCounter(TorchDispatchMode):
    """Counts the aten ops that actually run; a saved op is never re-executed."""

    def __init__(self):
        self.counts: Counter[str] = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.counts[str(func)] += 1
        return func(*args, **(kwargs or {}))


def tiny():
    return tiny_model_config()


def tiny_args():
    return torch.randn(1, 3, 32, 32), torch.tensor([5.0]), torch.randn(1, 16, 32)


def sdpa_key(counts: Counter[str]) -> str:
    """The forward SDPA op this build dispatched to (CPU and CUDA differ)."""
    keys = [k for k in counts if "_scaled_dot_product" in k and not k.endswith("_backward.default")]
    assert len(keys) == 1, keys
    return keys[0]


def op_counts(model: IrisDiT, args, policy: str, every: int = 2) -> Counter[str]:
    model.activation_checkpointing = policy
    model.ac_selective_every = every
    model.zero_grad(set_to_none=True)
    counter = OpCounter()
    with counter:
        model(*args).x.square().mean().backward()
    return counter.counts


# -- stride arithmetic ------------------------------------------------------


def test_selective_layer_checkpoints_every_third_block():
    got = [ac.should_checkpoint("selective_layer", i, 3) for i in range(7)]
    assert got == [True, False, False, True, False, False, True]


def test_blanket_policies_ignore_the_stride():
    assert not any(ac.should_checkpoint("none", i, 2) for i in range(4))
    assert all(ac.should_checkpoint("full", i, 2) for i in range(4))
    assert all(ac.should_checkpoint("selective_op", i, 2) for i in range(4))


# -- save list --------------------------------------------------------------


def test_save_list_covers_attention_matmul_and_convolution():
    ops = ac.resolve_save_ops()
    for op in (
        torch.ops.aten.mm.default,
        torch.ops.aten.bmm.default,
        torch.ops.aten.addmm.default,
        torch.ops.aten.convolution.default,
        torch.ops.aten._scaled_dot_product_flash_attention.default,
        torch.ops.aten._scaled_dot_product_efficient_attention.default,
    ):
        assert op in ops
    assert all(isinstance(op, torch._ops.OpOverload) for op in ops)
    assert 0 < len(ops) <= len(ac.SAVE_OP_NAMES)


def test_save_list_skips_names_this_torch_does_not_have():
    ops = ac.resolve_save_ops(("mm", "definitely_not_an_aten_op", "convolution"))
    assert ops == (torch.ops.aten.mm.default, torch.ops.aten.convolution.default)


def test_save_list_skips_packets_without_a_default_overload():
    # aten.softmax exists but is reachable only through typed overloads
    assert getattr(torch.ops.aten.softmax, "default", None) is None
    assert ac.resolve_save_ops(("softmax", "mm")) == (torch.ops.aten.mm.default,)


def test_context_fn_is_torch_default_unless_selective_op():
    for policy in ("none", "full", "selective_layer"):
        assert ac.checkpoint_context(policy) is noop_context_fn
    context_fn = ac.checkpoint_context("selective_op")
    assert context_fn is not noop_context_fn
    first, second = context_fn(), context_fn()
    assert len(first) == len(second) == 2
    # a region must never share its save cache with another region
    assert first[0] is not second[0]


# -- config surface ---------------------------------------------------------


def test_validate_policy_rejects_an_unknown_policy():
    with pytest.raises(ValueError, match="activation_checkpointing"):
        ac.validate_policy("selective", 2)


def test_validate_policy_rejects_a_non_positive_stride():
    with pytest.raises(ValueError, match="ac_selective_every"):
        ac.validate_policy("selective_layer", 0)
    ac.validate_policy("selective_op", 0)  # the stride is irrelevant here


# -- equivalence ------------------------------------------------------------


@pytest.mark.parametrize("policy", ("full", "selective_op", "selective_layer"))
def test_forward_matches_the_uncheckpointed_model(policy):
    torch.manual_seed(0)
    model = IrisDiT(tiny()).train()
    args = tiny_args()
    expected = model(*args).x
    model.activation_checkpointing = policy
    torch.testing.assert_close(model(*args).x, expected)


@pytest.mark.parametrize("policy", ("full", "selective_op", "selective_layer"))
def test_gradients_match_the_uncheckpointed_model(policy):
    torch.manual_seed(0)
    model = IrisDiT(tiny()).train()
    args = tiny_args()

    model(*args).x.square().mean().backward()
    expected = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    assert expected

    model.zero_grad(set_to_none=True)
    model.activation_checkpointing = policy
    model(*args).x.square().mean().backward()
    for name, grad in expected.items():
        torch.testing.assert_close(model.get_parameter(name).grad, grad, msg=name)


# -- what each policy actually recomputes -----------------------------------


def test_selective_op_saves_attention_and_replays_pointwise():
    torch.manual_seed(0)
    model = IrisDiT(tiny()).train()
    args = tiny_args()
    none = op_counts(model, args, "none")
    full = op_counts(model, args, "full")
    selective = op_counts(model, args, "selective_op")
    attn = sdpa_key(none)
    pointwise = "aten.silu.default"

    # the linear text adapter has no attention, so every SDPA call in this
    # model sits inside a checkpointed region
    assert none[attn] > 0
    assert full[attn] == 2 * none[attn]
    assert selective[attn] == none[attn]

    assert none[pointwise] < full[pointwise]
    assert selective[pointwise] == full[pointwise]


def test_selective_layer_recomputes_only_the_strided_blocks():
    torch.manual_seed(0)
    cfg = tiny_model_config(depth=4)
    model = IrisDiT(cfg).train()
    args = tiny_args()
    none = op_counts(model, args, "none")
    strided = op_counts(model, args, "selective_layer", every=2)
    attn = sdpa_key(none)

    # one attention per block; the trunk and the pixel stack stride separately
    recomputed = sum(ac.should_checkpoint("selective_layer", i, 2) for i in range(cfg.depth))
    recomputed += sum(ac.should_checkpoint("selective_layer", i, 2) for i in range(cfg.pixel.depth))
    assert none[attn] == cfg.depth + cfg.pixel.depth
    assert strided[attn] == none[attn] + recomputed


def test_eval_mode_enters_no_region():
    torch.manual_seed(0)
    model = IrisDiT(tiny()).eval()
    args = tiny_args()
    counts = op_counts(model, args, "full")
    attn = sdpa_key(counts)
    assert counts[attn] == model.cfg.depth + model.cfg.pixel.depth
