from dataclasses import replace

import torch

from iris3b.models import IrisDiT, get_preset
from tiny_config import tiny_model_config


def tiny():
    return tiny_model_config()


def test_iris_3b_parameter_count_and_topology():
    from iris3b.models.blocks.mmdit import MMDiTBlock
    from iris3b.models.blocks.single_stream import SingleStreamBlock

    with torch.device("meta"):
        model = IrisDiT(get_preset("iris-3b"))
    assert model.num_parameters == 2_987_511_888
    # one third dual-stream, at the bottom, then the single-stream tail
    kinds = [type(block) for block in model.blocks]
    assert kinds == [MMDiTBlock] * 8 + [SingleStreamBlock] * 16
    # shared_bias under a hybrid must alias the tail's "shared" core onto the
    # image core, otherwise it silently builds a third adaLN core
    assert sorted(model.modulation_cores) == ["adaln_img", "adaln_txt"]


def test_feature_capture():
    cfg = tiny()
    model = IrisDiT(cfg).eval()
    args = (torch.randn(1, 3, 32, 32), torch.tensor([1.0]), torch.randn(1, 16, 32))
    out = model(*args, capture=(1, cfg.depth))
    assert sorted(out.features) == [1, cfg.depth]
    assert out.features[1].shape == (1, 64, cfg.hidden_size)
    assert not torch.equal(out.features[1], out.features[cfg.depth])
    assert model(*args).features == {}


def test_zero_init_head_means_zero_output_only_with_adaln_zero():
    # head is zero-init, but adaLN gates are NOT zero by default: with the
    # default init only the final projection forces exact zeros
    model = IrisDiT(tiny()).eval()
    out = model(torch.randn(1, 3, 32, 32), torch.tensor([0.0]), torch.randn(1, 16, 32))
    assert torch.all(out.x == 0)


def test_adaln_zero_init_flag():
    cfg = replace(tiny(), adaln_zero_init=True)
    model = IrisDiT(cfg)
    for name, module in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if leaf.startswith("adaln"):
            assert torch.all(module.weight == 0), name
            assert torch.all(module.bias == 0), name
