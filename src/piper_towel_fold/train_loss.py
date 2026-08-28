"""Parse LeRobot training logs and judge whether loss converged.

Shared by tools/plot_train_loss.py and start_training.py (post-run summary).
"""

from __future__ import annotations

import csv
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LOSS_RE = re.compile(
    r"(?:\|\s*(?P<tqdm_step>\d+)/(?P<tqdm_total>\d+)\b)?"
    r".*?\bloss:(?P<loss>[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"
    r"(?:\s+grdn:(?P<grdn>[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?))?"
    r"(?:\s+lr:(?P<lr>[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?))?"
    r"(?:.*?epch:(?P<epch>[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?))?",
)

TQDM_ONLY_RE = re.compile(r"\|\s*(?P<tqdm_step>\d+)/(?P<tqdm_total>\d+)\b")

VERDICT_LABELS = {
    "converging_well": "收敛良好",
    "still_decreasing": "仍在下降",
    "plateau": "已进入平台期",
    "diverging_or_unstable": "不稳定/发散",
    "mixed": "趋势不清晰",
}


@dataclass
class MetricPoint:
    step: int
    loss: float
    grdn: float | None = None
    lr: float | None = None
    epch: float | None = None


def parse_log_text(text: str, log_freq: int | None = None) -> list[MetricPoint]:
    """Extract (step, loss) points. Prefer accurate tqdm steps when present."""
    text = text.replace("\r", "\n")
    points: list[MetricPoint] = []
    inferred_steps: list[int | None] = []

    for raw in text.splitlines():
        line = raw.strip()
        if "loss:" not in line:
            continue
        match = LOSS_RE.search(line)
        if not match:
            continue

        loss = float(match.group("loss"))
        grdn = float(match.group("grdn")) if match.group("grdn") else None
        lr = float(match.group("lr")) if match.group("lr") else None
        epch = float(match.group("epch")) if match.group("epch") else None

        step: int | None = None
        if match.group("tqdm_step"):
            step = int(match.group("tqdm_step"))
        else:
            tqdm_match = TQDM_ONLY_RE.search(line)
            if tqdm_match:
                step = int(tqdm_match.group("tqdm_step"))

        inferred_steps.append(step)
        points.append(MetricPoint(step=step or 0, loss=loss, grdn=grdn, lr=lr, epch=epch))

    if not points:
        return []

    known = [s for s in inferred_steps if s is not None]
    if log_freq is None:
        if len(known) >= 2:
            deltas = [b - a for a, b in zip(known, known[1:]) if b > a]
            log_freq = statistics.mode(deltas) if deltas else 100
        else:
            log_freq = 100

    filled: list[MetricPoint] = []
    for idx, point in enumerate(points):
        step = inferred_steps[idx]
        if step is None:
            step = (idx + 1) * log_freq
        if filled and step <= filled[-1].step:
            step = filled[-1].step + log_freq
        filled.append(
            MetricPoint(step=step, loss=point.loss, grdn=point.grdn, lr=point.lr, epch=point.epch)
        )
    return filled


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1 or len(values) < window:
        return list(values)
    out: list[float] = []
    running = 0.0
    for i, value in enumerate(values):
        running += value
        if i >= window:
            running -= values[i - window]
        denom = window if i >= window - 1 else (i + 1)
        out.append(running / denom)
    return out


def linear_slope(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    denom = sum((x - mean_x) ** 2 for x in xs)
    if denom == 0:
        return 0.0
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denom


def analyze(points: list[MetricPoint], tail_frac: float = 0.3) -> dict[str, Any]:
    losses = [p.loss for p in points]
    steps = [p.step for p in points]
    n = len(points)
    tail_n = max(3, int(math.ceil(n * tail_frac)))
    head_n = max(3, min(n // 3, n - tail_n)) if n >= 6 else max(1, n // 2)

    head = losses[:head_n]
    tail = losses[-tail_n:]
    head_mean = statistics.fmean(head)
    tail_mean = statistics.fmean(tail)
    drop_ratio = (head_mean - tail_mean) / abs(head_mean) if head_mean != 0 else 0.0

    tail_points = points[-tail_n:]
    xs = [p.step / 1000.0 for p in tail_points]
    ys = [p.loss for p in tail_points]
    slope_per_1k = linear_slope(xs, ys)

    tail_std = statistics.pstdev(tail) if len(tail) > 1 else 0.0
    fluctuation = (tail_std / abs(tail_mean)) if tail_mean != 0 else 0.0

    min_loss = min(losses)
    min_step = steps[losses.index(min_loss)]

    if drop_ratio >= 0.5 and slope_per_1k <= 0 and fluctuation < 0.25:
        verdict = "converging_well"
        advice = "前段已明显下降，末段斜率≤0 且波动不大，可认为收敛良好，可用 last checkpoint 上机。"
    elif drop_ratio >= 0.3 and slope_per_1k < -0.01:
        verdict = "still_decreasing"
        advice = "仍在明显下降，建议加长 steps 再训一轮，或 resume 继续。"
    elif drop_ratio >= 0.2 and abs(slope_per_1k) < 0.02 and fluctuation < 0.35:
        verdict = "plateau"
        advice = "大致进入平台期，可直接用当前 checkpoint 上机验证。"
    elif slope_per_1k > 0.02 and drop_ratio < 0.2:
        verdict = "diverging_or_unstable"
        advice = "末段 loss 上行/不稳定，检查学习率、数据或 batch。"
    else:
        verdict = "mixed"
        advice = "趋势不够清晰，建议结合曲线图与上机成功率判断。"

    return {
        "n_points": n,
        "first_step": steps[0],
        "last_step": steps[-1],
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "min_loss": min_loss,
        "min_step": min_step,
        "head_mean": head_mean,
        "tail_mean": tail_mean,
        "drop_ratio": drop_ratio,
        "slope_per_1k": slope_per_1k,
        "tail_fluctuation": fluctuation,
        "verdict": verdict,
        "verdict_zh": VERDICT_LABELS.get(verdict, verdict),
        "advice": advice,
    }


def save_csv(points: list[MetricPoint], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "loss", "grdn", "lr", "epch"])
        writer.writeheader()
        for p in points:
            writer.writerow(
                {
                    "step": p.step,
                    "loss": p.loss,
                    "grdn": "" if p.grdn is None else p.grdn,
                    "lr": "" if p.lr is None else p.lr,
                    "epch": "" if p.epch is None else p.epch,
                }
            )


def save_plot(points: list[MetricPoint], path: Path, smooth: int = 5) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    steps = [p.step for p in points]
    losses = [p.loss for p in points]
    smooth_losses = moving_average(losses, smooth)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(steps, losses, color="#9aa0a6", linewidth=1.0, alpha=0.7, label="loss")
    if smooth > 1:
        ax.plot(steps, smooth_losses, color="#1a73e8", linewidth=2.0, label=f"MA({smooth})")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("Training loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def print_report(points: list[MetricPoint], summary: dict[str, Any]) -> None:
    print()
    print("=" * 60)
    print("训练结束 · Loss 收敛判断")
    print("=" * 60)
    print(f"  points:     {summary['n_points']}")
    print(f"  step range: {summary['first_step']} -> {summary['last_step']}")
    print(f"  first loss: {summary['first_loss']:.4f}")
    print(f"  last loss:  {summary['last_loss']:.4f}")
    print(f"  min loss:   {summary['min_loss']:.4f} @ step {summary['min_step']}")
    print(f"  head mean:  {summary['head_mean']:.4f}")
    print(f"  tail mean:  {summary['tail_mean']:.4f}")
    print(f"  drop:       {float(summary['drop_ratio']) * 100:.1f}%")
    print(f"  slope/1k:   {summary['slope_per_1k']:.4f}")
    print(f"  tail flut.: {float(summary['tail_fluctuation']) * 100:.1f}%")
    print(f"  verdict:    {summary['verdict']} ({summary.get('verdict_zh', '')})")
    print(f"  advice:     {summary['advice']}")
    print()
    print("Recent points (last 8):")
    for p in points[-8:]:
        grdn = f" grdn={p.grdn:.3f}" if p.grdn is not None else ""
        epch = f" epch={p.epch:.2f}" if p.epch is not None else ""
        print(f"  step={p.step:6d}  loss={p.loss:.4f}{grdn}{epch}")
    print("=" * 60)


def report_from_log(
    log_path: Path,
    *,
    output_dir: Path | None = None,
    log_freq: int | None = None,
    tail_frac: float = 0.3,
    write_artifacts: bool = True,
    smooth: int = 5,
    min_points: int = 3,
) -> dict[str, Any] | None:
    """Parse a training log, print the convergence verdict, optionally save CSV/PNG."""
    if not log_path.exists():
        print(f"Skip loss report: log not found ({log_path})")
        return None

    text = log_path.read_text(encoding="utf-8", errors="replace")
    points = parse_log_text(text, log_freq=log_freq)
    if len(points) < min_points:
        print(
            f"Skip loss report: only {len(points)} loss point(s) parsed from {log_path}. "
            "Log may be incomplete or format unexpected."
        )
        return None

    summary = analyze(points, tail_frac=tail_frac)
    print_report(points, summary)

    artifact_dir = output_dir or log_path.parent
    if write_artifacts:
        csv_path = artifact_dir / "loss.csv"
        save_csv(points, csv_path)
        print(f"  wrote: {csv_path}")
        plot_path = artifact_dir / "loss.png"
        if save_plot(points, plot_path, smooth=smooth):
            print(f"  wrote: {plot_path}")
        else:
            print("  skip plot: matplotlib not installed")

    return summary
