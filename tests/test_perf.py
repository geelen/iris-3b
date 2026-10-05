import pytest
import torch
from torch import nn

from iris3b.config import OptimizerConfig
from iris3b.train.ema import EMA
from iris3b.train.optim import build_optimizer


def _tiny_model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(8, 16), nn.SiLU(), nn.Linear(16, 4))


def test_foreach_ema_matches_loop_bitwise():
    """The multi-tensor EMA path must produce bit-identical shadows."""
    model_a, model_b = _tiny_model(), _tiny_model()
    ema_loop = EMA(model_a, decay=0.999, foreach=False)
    ema_fast = EMA(model_b, decay=0.999, foreach=True)
    for _ in range(5):
        with torch.no_grad():
            for pa, pb in zip(model_a.parameters(), model_b.parameters(), strict=True):
                delta = torch.randn_like(pa) * 0.01
                pa.add_(delta)
                pb.add_(delta)
        ema_loop.update(model_a)
        ema_fast.update(model_b)
    for (na, a), (nb, b) in zip(
        ema_loop.module.state_dict().items(), ema_fast.module.state_dict().items(), strict=True
    ):
        assert na == nb
        assert torch.equal(a, b), f"EMA mismatch in {na}"


def test_foreach_ema_dtype_mismatch_falls_back():
    model = _tiny_model()
    ema = EMA(model, decay=0.999, foreach=True)
    ema.module.half()  # force a shadow/source dtype mismatch
    ema.update(model)  # must not raise; falls back to the casting loop
    assert ema.foreach is False


def test_fused_flag_rejected_for_non_adamw():
    params = list(_tiny_model().parameters())
    with pytest.raises(ValueError, match="fused_adamw"):
        build_optimizer(OptimizerConfig(name="muon"), params, 256, fused=True)
