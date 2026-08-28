from pathlib import Path

from piper_towel_fold.local_tokenizer import (
    HUB_PALIGEMMA,
    rewrite_checkpoint_tokenizers,
    rewrite_tokenizer_name,
)


def test_rewrite_missing_local_keeps_hub_id(tmp_path, monkeypatch):
    monkeypatch.delenv("PALIGEMMA_TOKENIZER_PATH", raising=False)
    monkeypatch.setattr(
        "piper_towel_fold.local_tokenizer.resolve_paligemma_tokenizer",
        lambda: None,
    )
    assert rewrite_tokenizer_name(HUB_PALIGEMMA) == HUB_PALIGEMMA


def test_rewrite_hub_id_to_local(tmp_path, monkeypatch):
    tok = tmp_path / "google" / "paligemma-3b-pt-224"
    tok.mkdir(parents=True)
    (tok / "tokenizer.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "piper_towel_fold.local_tokenizer.resolve_paligemma_tokenizer",
        lambda: tok,
    )
    assert rewrite_tokenizer_name(HUB_PALIGEMMA) == str(tok)
    assert rewrite_tokenizer_name("/mnt/disk/fyx/piper/google/paligemma-3b-pt-224") == str(tok)


def test_rewrite_checkpoint_json(tmp_path, monkeypatch):
    tok = tmp_path / "paligemma-3b-pt-224"
    tok.mkdir()
    (tok / "tokenizer.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "piper_towel_fold.local_tokenizer.resolve_paligemma_tokenizer",
        lambda: tok,
    )
    ckpt = tmp_path / "pretrained_model"
    ckpt.mkdir()
    path = ckpt / "policy_preprocessor.json"
    path.write_text(
        '{"steps":[{"registry_name":"tokenizer_processor","config":{"tokenizer_name":"google/paligemma-3b-pt-224"}}]}',
        encoding="utf-8",
    )
    changed = rewrite_checkpoint_tokenizers(ckpt)
    assert path in changed
    data = path.read_text(encoding="utf-8")
    assert str(tok) in data
    assert HUB_PALIGEMMA not in data
