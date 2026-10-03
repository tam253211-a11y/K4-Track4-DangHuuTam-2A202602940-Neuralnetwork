"""plots.py — ảnh biểu đồ là sản phẩm nộp (xem README mục 6): mỗi thí nghiệm một ảnh figures/<exp_id>.png.

Khi notebook chạy trong code/, lưu vào "../figures/" (ví dụ path = f"../figures/{exp_id}.png").
"""
from __future__ import annotations

import math
import os

import matplotlib.pyplot as plt

MAJORITY_ACC = 0.4876   # mốc "luôn đoán lớp đa số" trên val


def _nan(v):
    return math.nan if v is None else v


def _series(result: dict, key: str):
    return [_nan(v) for v in result["history"].get(key, [])]


def _cfg_text(cfg: dict) -> str:
    parts = [f"{cfg['optimizer']} lr={cfg['lr']:g}", f"loss={cfg['loss']}", f"batch={cfg['batch']}",
             f"hidden={tuple(cfg['hidden'])}", f"init={cfg['init']}", f"dropout={cfg['dropout']}",
             f"clip={cfg['clip_norm']}", cfg["precision"], f"wd={cfg['weight_decay']}", f"seed={cfg['seed']}"]
    if cfg.get("scheduler"):
        parts.append(f"sched={cfg['scheduler']}")
    return ", ".join(parts)


def plot_run(result: dict, path: str) -> None:
    """Vẽ MỘT thí nghiệm thành một ảnh PNG 3 ô:
         (1) train_loss (eval mode, tập con 50k) và val_loss theo epoch
         (2) val_acc và val_macro_f1 theo epoch (kèm mốc đoán đa số)
         (3) grad_norm theo epoch: trung bình và lớn nhất trong epoch, đo TRƯỚC khi clip (kèm ngưỡng clip nếu có)
    Đường đứt dọc = best epoch (val_loss thấp nhất).
    """
    cfg, s = result["cfg"], result["summary"]
    ep = result["history"]["epoch"]
    best = s.get("best_epoch") or 0
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))

    ax = axes[0]
    ax.plot(ep, _series(result, "train_loss"), "o-", ms=3, label="train loss (eval mode, 50k mẫu)")
    ax.plot(ep, _series(result, "val_loss"), "s-", ms=3, label="val loss")
    if s.get("step0_loss") is not None:
        ax.scatter([0], [s["step0_loss"]], c="k", marker="x", zorder=5, label=f"bước 0 = {s['step0_loss']:.3f}")
    ax.set_title("Loss theo epoch"); ax.set_xlabel("epoch"); ax.set_ylabel(f"{cfg['loss']} loss")

    ax = axes[1]
    ax.plot(ep, _series(result, "val_acc"), "o-", ms=3, label="val accuracy")
    ax.plot(ep, _series(result, "val_macro_f1"), "s-", ms=3, label="val macro-F1")
    ax.axhline(MAJORITY_ACC, ls=":", c="gray", label=f"đoán đa số (acc {MAJORITY_ACC})")
    ax.set_title("Val accuracy / macro-F1"); ax.set_xlabel("epoch"); ax.set_ylabel("điểm")

    ax = axes[2]
    ax.plot(ep, _series(result, "grad_norm"), "o-", ms=3, label="grad_norm TB / epoch")
    ax.plot(ep, _series(result, "grad_norm_max"), "^--", ms=3, alpha=0.7, label="grad_norm max / epoch")
    if cfg.get("clip_norm") is not None:
        ax.axhline(cfg["clip_norm"], ls=":", c="red", label=f"ngưỡng clip c = {cfg['clip_norm']}")
    ax.set_yscale("log")
    ax.set_title("‖g‖ toàn cục (trước clip)"); ax.set_xlabel("epoch"); ax.set_ylabel("grad norm (log)")

    for ax in axes:
        if best:
            ax.axvline(best, ls="--", c="green", alpha=0.6, label=f"best epoch = {best}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    status = "  [PHÂN KỲ: " + str(s.get("diverged_at")) + "]" if s.get("diverged") else ""
    fig.suptitle(f"{cfg['exp_id']}  —  {_cfg_text(cfg)}\n"
                 f"best epoch {best}: val_loss {_nan(s.get('best_val_loss')):.4f}, val_acc {_nan(s.get('val_acc')):.4f}, "
                 f"val_macro_f1 {_nan(s.get('val_macro_f1')):.4f}{status}", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_compare(results: list[dict], metric: str, path: str, title: str = "",
                 logy: bool | None = None, show: bool = False) -> None:
    """Vẽ chồng một chỉ số (ví dụ "val_loss", "val_macro_f1", "grad_norm") của nhiều thí nghiệm
    trên cùng một trục, mỗi thí nghiệm một đường, chú thích bằng exp_id. Dùng cho figures/compare_<nhóm>.png.
    Có thể truyền một danh sách metric (vd ["val_loss", "val_macro_f1"]) để vẽ nhiều ô cạnh nhau.
    """
    metrics = [metric] if isinstance(metric, str) else list(metric)
    fig, axes = plt.subplots(1, len(metrics), figsize=(6.5 * len(metrics), 4.4), squeeze=False)
    for ax, m in zip(axes[0], metrics):
        for r in results:
            ax.plot(r["history"]["epoch"], _series(r, m), "o-", ms=2.5, label=r["cfg"]["exp_id"])
        use_log = logy if logy is not None else (m.startswith("grad_norm"))
        if use_log:
            ax.set_yscale("log")
        ax.set_xlabel("epoch"); ax.set_ylabel(m); ax.set_title(m)
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle(title or f"So sánh: {', '.join(metrics)}", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
