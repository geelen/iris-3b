from pathlib import Path

from iris3b.config import Config, load_config


def test_yaml_and_dotlist_composition():
    cfg = load_config(
        "configs/iris3b/stage1_256.yaml",
        overrides=["model.depth=6", "flow.shift=3.0", "train.batch_size=2"],
    )
    assert cfg.model.depth == 6  # dotlist wins over defaults
    assert cfg.flow.shift == 3.0  # dotlist wins over yaml
    assert cfg.train.batch_size == 2
    assert cfg.model.rope_aspect == "square"  # yaml wins over defaults
    assert cfg.repa.weight == 0.5
    assert cfg.model.hidden_size == 2560  # untouched default


def test_stage_configs_parse():
    root = Path(__file__).parent.parent / "configs"
    paths = sorted(root.rglob("*.yaml"))
    assert paths
    for path in paths:
        cfg = load_config(path)
        assert isinstance(cfg, Config), path
