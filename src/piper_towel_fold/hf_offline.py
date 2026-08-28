"""Run lerobot-train with offline paligemma tokenizer remapping.

Usage (same args as lerobot-train):

    python -m piper_towel_fold.hf_offline --dataset.repo_id=... ...
"""

from __future__ import annotations

import sys

from .local_tokenizer import install, rewrite_output_dir_tokenizers


def main() -> None:
    install()
    from lerobot.scripts.lerobot_train import main as train_main

    sys.argv[0] = "lerobot-train"
    try:
        train_main()
    finally:
        output_dir = _output_dir_from_argv(sys.argv)
        if output_dir:
            changed = rewrite_output_dir_tokenizers(output_dir)
            if changed:
                print("Rewrote tokenizer_name in:")
                for path in changed:
                    print(f"  {path}")


def _output_dir_from_argv(argv: list[str]) -> str | None:
    for arg in argv:
        if arg.startswith("--output_dir="):
            return arg.split("=", 1)[1]
    for i, arg in enumerate(argv[:-1]):
        if arg == "--output_dir":
            return argv[i + 1]
    return None


if __name__ == "__main__":
    main()
