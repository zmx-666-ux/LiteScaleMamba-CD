from pathlib import Path

import yaml


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "hybrid.yaml"


def load_model_kwargs(config_path=DEFAULT_CONFIG):
    data = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("model"), dict):
        raise ValueError("The config must contain a model mapping")
    return data["model"]


def build_model(config_path=DEFAULT_CONFIG, mobilenet_pretrained=None, vssm_pretrained=None):
    from models.ChangeHybridBCD import ChangeHybridBCD

    kwargs = load_model_kwargs(config_path)
    return ChangeHybridBCD(
        mobilenet_pretrained=mobilenet_pretrained,
        vssm_pretrained=vssm_pretrained,
        **kwargs,
    )
