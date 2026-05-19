import os
from functools import lru_cache
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"


@lru_cache(maxsize=1)
def get_config(path: str | None = None) -> dict:
    cfg_path = Path(path or os.environ.get("BETTERDAY_CONFIG", DEFAULT_CONFIG_PATH))
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Resolve storage paths relative to project root if not absolute
    for key in ("data_dir", "index_path", "db_path", "thumbnail_dir"):
        p = Path(cfg["storage"][key])
        if not p.is_absolute():
            cfg["storage"][key] = str((PROJECT_ROOT / p).resolve())
    return cfg


def ensure_storage_dirs(cfg: dict) -> None:
    Path(cfg["storage"]["data_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["storage"]["thumbnail_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["storage"]["index_path"]).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg["storage"]["db_path"]).parent.mkdir(parents=True, exist_ok=True)
