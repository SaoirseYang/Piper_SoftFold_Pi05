#!/usr/bin/env python3
"""Probe SoftFold async deploy instance health (ckpt / preload / tmux / tunnel).

READY = remote Policy Server logged model preload complete
        ("All keys loaded successfully!" from lerobot load; deploy probes this line).
DEPLOYING = server tmux up but that line not seen yet.

Usage:
  python3 scripts/lib/deploy_status.py --run-dir runs/async_deploy/<id> \\
    --ssh-host A600 --repo-root /data/.../SoftFold [--json] [--check-id N] [--preflight]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

# Matched in remote server log or tmux pane (model weight load during preload_at_startup).
PRELOAD_READY_MARKERS = (
    "All keys loaded successfully!",
    "Startup preload finished",
)

PRELOAD_ERROR_MARKERS = (
    "Traceback (most recent call last)",
    "CUDA out of memory",
    "RuntimeError",
    "FileNotFoundError",
    "OSError:",
)


def _ssh(host: str, remote_cmd: str, timeout: int = 45) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host, remote_cmd],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        return proc.returncode, out.strip()
    except (subprocess.TimeoutExpired, OSError) as exc:
        return 1, str(exc)


def _tunnel_alive(run_dir: Path, inst: dict[str, Any]) -> bool:
    pid_file = run_dir / inst["tunnel_pid_file"]
    if not pid_file.is_file():
        return False
    try:
        pid = int(pid_file.read_text().strip())
    except ValueError:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _local_port_open(port: int) -> bool:
    try:
        proc = subprocess.run(
            ["bash", "-lc", f"exec 3<>/dev/tcp/127.0.0.1/{port}"],
            check=False,
            capture_output=True,
            timeout=2,
        )
        return proc.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def _text_preload_ready(text: str) -> bool:
    return any(m in text for m in PRELOAD_READY_MARKERS)


def _text_preload_failed(text: str) -> bool:
    return any(m in text for m in PRELOAD_ERROR_MARKERS)


def _remote_server_log(repo_root: str, run_id: str, inst_id: int) -> str:
    return f"{repo_root.rstrip('/')}/runs/async_deploy/{run_id}/server_{inst_id}.log"


def _shell_quote(path: str) -> str:
    import shlex

    return shlex.quote(path)


def _probe_preload_remote(
    ssh_host: str,
    *,
    repo_root: str,
    run_id: str,
    instances: list[dict[str, Any]],
) -> dict[int, str]:
    """Return per-instance preload state: done | loading | failed | no_server | unknown."""
    parts = ["set +e", "echo '__BEGIN__'"]
    marker_re = "|".join(re.escape(m) for m in PRELOAD_READY_MARKERS)
    err_re = "|".join(re.escape(m) for m in PRELOAD_ERROR_MARKERS)

    for inst in instances:
        idx = int(inst["id"])
        sess = inst["tmux_session"]
        log = _remote_server_log(repo_root, run_id, idx)
        q_log = _shell_quote(log)
        q_sess = _shell_quote(sess)
        parts.append(
            f"if tmux has-session -t {q_sess} 2>/dev/null; then "
            f"  if [ -f {q_log} ] && grep -qE '{marker_re}' {q_log}; then "
            f"    echo PRELOAD_{idx}=done; "
            f"  elif [ -f {q_log} ] && grep -qE '{err_re}' {q_log}; then "
            f"    echo PRELOAD_{idx}=failed; "
            f"  elif tmux capture-pane -pt {q_sess} -S -8000 2>/dev/null | grep -qE '{marker_re}'; then "
            f"    echo PRELOAD_{idx}=done; "
            f"  elif tmux capture-pane -pt {q_sess} -S -8000 2>/dev/null | grep -qE '{err_re}'; then "
            f"    echo PRELOAD_{idx}=failed; "
            f"  else echo PRELOAD_{idx}=loading; fi; "
            f"else "
            f"  if [ -f {q_log} ] && grep -qE '{marker_re}' {q_log}; then "
            f"    echo PRELOAD_{idx}=done; "
            f"  elif [ -f {q_log} ] && grep -qE '{err_re}' {q_log}; then "
            f"    echo PRELOAD_{idx}=failed; "
            f"  else echo PRELOAD_{idx}=no_server; fi; "
            f"fi"
        )
    parts.append("echo '__END__'")
    code, out = _ssh(ssh_host, "\n".join(parts))
    result: dict[int, str] = {}
    if code == 0 or "__BEGIN__" in out:
        for line in out.splitlines():
            if line.startswith("PRELOAD_") and "=" in line:
                k, v = line.split("=", 1)
                try:
                    result[int(k.replace("PRELOAD_", ""))] = v.strip()
                except ValueError:
                    continue
    return result


def probe(
    run_dir: Path,
    *,
    ssh_host: str,
    repo_root: str,
) -> list[dict[str, Any]]:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    instances = manifest["instances"]
    run_id = str(manifest.get("run_id") or run_dir.name)

    # One SSH round-trip: ckpt dirs + tmux sessions.
    remote_script_parts = [
        "set +e",
        "echo '__BEGIN__'",
    ]
    for inst in instances:
        rel = inst["policy_path"]
        abs_path = rel if rel.startswith("/") else f"{repo_root.rstrip('/')}/{rel}"
        remote_script_parts.append(
            f'p=$(printf %q "{abs_path}"); '
            f'if [ -d "$p" ] && {{ [ -f "$p/config.json" ] || [ -f "$p/model.safetensors" ] || '
            f'[ -f "$p/model.safetensors.index.json" ] || [ -f "$p/pytorch_model.bin" ]; }}; then '
            f'echo CKPT_{inst["id"]}=ok; '
            f'elif [ -d "$p" ]; then echo CKPT_{inst["id"]}=partial; '
            f'else echo CKPT_{inst["id"]}=missing; fi'
        )
        sess = inst["tmux_session"]
        remote_script_parts.append(
            f'if tmux has-session -t $(printf %q "{sess}") 2>/dev/null; then '
            f'echo TMUX_{inst["id"]}=up; else echo TMUX_{inst["id"]}=down; fi'
        )
    remote_script_parts.append("echo '__END__'")
    remote_cmd = "\n".join(remote_script_parts)
    code, out = _ssh(ssh_host, remote_cmd)
    remote_map: dict[str, str] = {}
    if code == 0 or "__BEGIN__" in out:
        for line in out.splitlines():
            if "=" in line and (line.startswith("CKPT_") or line.startswith("TMUX_")):
                k, v = line.split("=", 1)
                remote_map[k.strip()] = v.strip()

    preload_map = _probe_preload_remote(
        ssh_host, repo_root=repo_root, run_id=run_id, instances=instances
    )

    rows: list[dict[str, Any]] = []
    for inst in instances:
        idx = int(inst["id"])
        ckpt = remote_map.get(f"CKPT_{idx}", "unknown")
        tmux = remote_map.get(f"TMUX_{idx}", "unknown")
        preload = preload_map.get(idx, "unknown")
        tunnel = "up" if _tunnel_alive(run_dir, inst) else "down"
        port_ok = _local_port_open(int(inst["port"]))
        server_log = _remote_server_log(repo_root, run_id, idx)

        if ckpt == "missing":
            status = "NO_CKPT"
            ok = False
            detail = "远端没有该 checkpoint 目录（或为空）"
        elif ckpt == "partial":
            status = "CKPT_PARTIAL"
            ok = False
            detail = "远端目录存在但缺少 config.json / 权重文件"
        elif preload == "failed":
            status = "FAILED"
            ok = False
            detail = f"Policy Server 启动/加载失败，见远端日志 {server_log}"
        elif tmux != "up" and preload != "done":
            status = "NO_SERVER"
            ok = False
            detail = "远端 tmux Policy Server 未运行（尚未开始或已退出）"
        elif preload == "loading" or preload == "unknown":
            status = "DEPLOYING"
            ok = False
            detail = "正在部署：等待权重加载完成（日志应出现 All keys loaded successfully!）"
        elif preload == "done":
            if tunnel != "up":
                status = "NO_TUNNEL"
                ok = False
                detail = "模型已加载完成，但本地 SSH tunnel 未建立"
            elif not port_ok:
                status = "DEPLOYING"
                ok = False
                detail = "模型已加载，等待 gRPC 端口就绪"
            else:
                status = "READY"
                ok = True
                detail = "preload 已完成（All keys loaded successfully!），tunnel 与端口正常"
        else:
            status = "UNKNOWN"
            ok = False
            detail = f"未知 preload 状态: {preload}"

        rows.append(
            {
                **inst,
                "ckpt_remote": ckpt,
                "tmux": tmux,
                "preload": preload,
                "tunnel": tunnel,
                "port_open": port_ok,
                "server_log": server_log,
                "status": status,
                "ok": ok,
                "detail": detail,
            }
        )
    return rows


def print_table(rows: list[dict[str, Any]], *, base: str, run_id: str) -> None:
    print(f"run_id={run_id}  base={base}")
    print("READY = 远端日志已输出 All keys loaded successfully!；DEPLOYING = 仍在加载权重")
    hdr = (
        f"{'id':>3}  {'port':>5}  {'steps':>5}  {'chunk':>5}  {'alpha':>5}  "
        f"{'rtc':<4}  {'status':<12}  {'preload':<8}  {'srv':<4}  {'tun':<4}  ckpt_path"
    )
    print(hdr)
    for inst in rows:
        ckpt = inst["policy_path"]
        if "/checkpoints/" in ckpt:
            ckpt = ckpt.split("/checkpoints/")[-1]
        rtc = "on" if inst.get("rtc_enabled") else "off"
        print(
            f"{inst['id']:3d}  {inst['port']:5d}  {inst['num_inference_steps']:5d}  "
            f"{inst['chunk_size_threshold']:5.2f}  {inst['smoothing_alpha']:5.2f}  "
            f"{rtc:<4}  {inst['status']:<12}  {inst['preload']:<8}  {inst['tmux']:<4}  "
            f"{inst['tunnel']:<4}  {ckpt}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--ssh-host", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--check-id", type=int, default=-1)
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Only verify remote checkpoints exist; exit 1 if any missing.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

    if args.preflight:
        import shlex

        missing: list[str] = []
        seen: set[str] = set()
        for inst in manifest["instances"]:
            rel = str(inst["policy_path"])
            if rel in seen:
                continue
            seen.add(rel)
            abs_path = rel if rel.startswith("/") else f"{args.repo_root.rstrip('/')}/{rel}"
            q = shlex.quote(abs_path)
            remote_cmd = (
                f"test -d {q} && "
                f"(test -f {q}/config.json || "
                f"test -f {q}/model.safetensors || "
                f"test -f {q}/model.safetensors.index.json || "
                f"test -f {q}/pytorch_model.bin)"
            )
            code, _ = _ssh(args.ssh_host, remote_cmd)
            mark = "OK" if code == 0 else "MISSING"
            print(f"  [{mark}] {rel}")
            if code != 0:
                missing.append(rel)
        if missing:
            print(
                "ERROR: 以下 checkpoint 在远端不存在或未完成，已中止 deploy（未启动 server/tunnel）:",
                file=__import__("sys").stderr,
            )
            for p in missing:
                print(f"  - {p}", file=__import__("sys").stderr)
            print(
                f"请核对 YAML checkpoints（步数请写引号如 \"050000\"），或: "
                f"ssh {args.ssh_host} 'ls {args.repo_root}/outputs/train/*/checkpoints'",
                file=__import__("sys").stderr,
            )
            raise SystemExit(1)
        print("all checkpoints present on remote.")
        return

    rows = probe(run_dir, ssh_host=args.ssh_host, repo_root=args.repo_root)

    if args.check_id >= 0:
        match = next((r for r in rows if int(r["id"]) == args.check_id), None)
        if match is None:
            raise SystemExit(f"unknown id: {args.check_id}")
        if not match["ok"]:
            raise SystemExit(
                f"instance {args.check_id} not ready: {match['status']} — {match['detail']}\n"
                f"  policy_path: {match['policy_path']}\n"
                f"  remote log: {match.get('server_log', '')}\n"
                f"  tail: ssh {args.ssh_host} 'tail -n 40 {match.get('server_log', '')}'"
            )
        print(json.dumps({"id": args.check_id, "status": match["status"], "ok": True}))
        return

    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        print_table(rows, base=manifest.get("base_config", ""), run_id=manifest.get("run_id", ""))


if __name__ == "__main__":
    main()
