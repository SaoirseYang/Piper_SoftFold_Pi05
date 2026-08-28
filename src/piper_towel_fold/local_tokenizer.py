"""Map Hub tokenizer ids to a local offline copy.

GPU machines (allinai2 / A600) cannot reach Hugging Face. Paligemma tokenizer
files live under ``<repo>/third_party/google/paligemma-3b-pt-224``. Call
``install()`` before training or loading a pi05 policy so:

* ``AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")`` reads local files
* saved ``policy_preprocessor.json`` can keep working without hand-editing
  ``google`` → an absolute path
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

HUB_PALIGEMMA = "google/paligemma-3b-pt-224"
_INSTALLED = False


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_tokenizer_dir() -> Path:
    return repo_root() / "third_party" / "google" / "paligemma-3b-pt-224"


def resolve_paligemma_tokenizer() -> Path | None:
    """Return the local paligemma tokenizer directory, or None if missing."""
    candidates: list[Path] = []
    env = os.environ.get("PALIGEMMA_TOKENIZER_PATH", "").strip()
    if env:
        candidates.append(Path(env).expanduser())
    candidates.extend(
        [
            default_tokenizer_dir(),
            Path("/data/yangjingwen/code/SoftFold/third_party/google/paligemma-3b-pt-224"),
            Path("/mnt/disk/fyx/piper/third_party/google/paligemma-3b-pt-224"),
            Path("/mnt/disk/fyx/piper/google/paligemma-3b-pt-224"),
        ]
    )
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve() if path.exists() else path
        if resolved in seen:
            continue
        seen.add(resolved)
        if _looks_like_tokenizer_dir(path):
            return path
    return None


def _looks_like_tokenizer_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    return (path / "tokenizer.json").is_file() or (path / "tokenizer.model").is_file()


def rewrite_tokenizer_name(name: str | None) -> str | None:
    """Rewrite Hub ids / stale absolute paths onto this machine's local copy."""
    if not name:
        return name
    text = str(name)
    if "paligemma-3b-pt-224" not in text and text != HUB_PALIGEMMA:
        return text
    local = resolve_paligemma_tokenizer()
    if local is None:
        return text
    local_s = str(local)
    if text in {HUB_PALIGEMMA, local_s}:
        return local_s
    if text.endswith("google/paligemma-3b-pt-224") or text.endswith("paligemma-3b-pt-224"):
        if not Path(text).exists():
            return local_s
    return text


def rewrite_checkpoint_tokenizers(pretrained_model_dir: str | Path) -> list[Path]:
    """Rewrite tokenizer_name fields in checkpoint JSON so they point locally."""
    root = Path(pretrained_model_dir)
    if not root.is_dir():
        return []
    changed: list[Path] = []
    for path in root.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if _rewrite_obj(data):
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            changed.append(path)
    return changed


def rewrite_output_dir_tokenizers(output_dir: str | Path) -> list[Path]:
    """Rewrite every checkpoint under a training output_dir."""
    root = Path(output_dir)
    changed: list[Path] = []
    if not root.exists():
        return changed
    seen: set[Path] = set()
    for pretrained in root.glob("checkpoints/*/pretrained_model"):
        resolved = pretrained.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        changed.extend(rewrite_checkpoint_tokenizers(pretrained))
    return changed


def _rewrite_obj(obj: Any) -> bool:
    changed = False
    if isinstance(obj, dict):
        for key, value in list(obj.items()):
            if key in {"tokenizer_name", "paligemma_tokenizer_name", "text_tokenizer_name"} and isinstance(
                value, str
            ):
                rewritten = rewrite_tokenizer_name(value)
                if rewritten != value:
                    obj[key] = rewritten
                    changed = True
            elif _rewrite_obj(value):
                changed = True
    elif isinstance(obj, list):
        for item in obj:
            if _rewrite_obj(item):
                changed = True
    return changed


def install() -> Path | None:
    """Patch tokenizer loading for the current process. Idempotent."""
    global _INSTALLED
    local = resolve_paligemma_tokenizer()
    if _INSTALLED:
        return local
    _patch_tokenizer_processor()
    _patch_auto_tokenizer()
    _INSTALLED = True
    if local is not None:
        print(f"Offline paligemma tokenizer: {local}")
    else:
        print(
            "warning: local paligemma tokenizer not found; "
            f"expected {default_tokenizer_dir()} or PALIGEMMA_TOKENIZER_PATH"
        )
    return local


def _patch_tokenizer_processor() -> None:
    try:
        from lerobot.processor.tokenizer_processor import TokenizerProcessorStep
    except ImportError:
        return

    original = TokenizerProcessorStep.__post_init__

    def patched(self, *args, **kwargs):  # noqa: ANN001
        if getattr(self, "tokenizer_name", None):
            self.tokenizer_name = rewrite_tokenizer_name(self.tokenizer_name)
        return original(self, *args, **kwargs)

    TokenizerProcessorStep.__post_init__ = patched  # type: ignore[method-assign]


def _patch_auto_tokenizer() -> None:
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return

    original = AutoTokenizer.from_pretrained

    def patched(pretrained_model_name_or_path, *args, **kwargs):  # noqa: ANN001
        path = rewrite_tokenizer_name(str(pretrained_model_name_or_path))
        if path and Path(path).exists():
            kwargs.setdefault("local_files_only", True)
        return original(path, *args, **kwargs)

    AutoTokenizer.from_pretrained = patched  # type: ignore[method-assign]
