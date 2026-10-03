"""optimizer.py — chọn bộ tối ưu và cắt gradient, để `train.py` gọn và mọi thí nghiệm công bằng.

Được dùng torch.optim.* và torch.nn.utils.clip_grad_norm_ (xem README mục 5).

Công thức (slide Chương 4):
    SGD            : w <- w - lr * g
    SGD + momentum : v <- mu * v + g ;  w <- w - lr * v          (dạng PyTorch)
    Adam           : m <- b1 m + (1-b1) g ; v <- b2 v + (1-b2) g^2 ; w <- w - lr * m_hat / (sqrt(v_hat) + eps)
    AdamW          : như Adam nhưng suy giảm trọng số tách riêng: w <- w - lr * wd * w - lr * m_hat / (sqrt(v_hat) + eps)
"""
from __future__ import annotations

import math

import torch

OPTIMIZERS = ("sgd", "sgd_momentum", "adam", "adamw")
SCHEDULERS = ("cosine",)


def build_optimizer(name: str, params, lr: float, weight_decay: float = 0.0,
                    momentum: float = 0.9, betas=(0.9, 0.999), eps: float = 1e-8):
    """Trả về một torch.optim.Optimizer.

    Chú ý: weight_decay của Adam (L2 trộn vào gradient, rồi bị chia cho sqrt(v_hat)) khác weight_decay
    của AdamW (suy giảm tách riêng, không đi qua bước chuẩn hoá của Adam).
    """
    if name not in OPTIMIZERS:
        raise ValueError(f"optimizer phải là một trong {OPTIMIZERS}, nhận {name!r}")
    if lr is None or not lr > 0:
        raise ValueError(f"lr phải là số dương (chọn bằng val), nhận {lr!r}")
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd_momentum":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, betas=tuple(betas), eps=eps, weight_decay=weight_decay)
    return torch.optim.AdamW(params, lr=lr, betas=tuple(betas), eps=eps, weight_decay=weight_decay)


def build_scheduler(optimizer, name: str | None, total_steps: int, **kwargs):
    """(Tuỳ chọn) Bộ lập lịch tốc độ học, gọi scheduler.step() sau MỖI bước cập nhật.

    name:
        None     : không dùng scheduler (baseline)
        "cosine" : CosineAnnealingLR từ lr ban đầu xuống eta_min (mặc định 0) sau total_steps bước
    Nếu dùng scheduler ở một thí nghiệm, ghi vào cột notes của bảng.
    """
    if name is None:
        return None
    if name not in SCHEDULERS:
        raise ValueError(f"scheduler phải là None hoặc một trong {SCHEDULERS}, nhận {name!r}")
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps,
                                                      eta_min=kwargs.get("eta_min", 0.0))


def clip_gradients(params, max_norm: float | None) -> float:
    """Cắt gradient theo chuẩn L2 toàn cục, và TRẢ VỀ chuẩn gradient TRƯỚC KHI cắt.

    max_norm = None: chỉ đo, không cắt (clip_grad_norm_ với max_norm = inf không đổi gradient).
    Giá trị trả về chính là `grad_norm` phải ghi lại ở mỗi bước (để thấy "gai" gradient).
    Khi dùng FP16 + GradScaler: phải scaler.unscale_(optimizer) TRƯỚC khi gọi hàm này.
    """
    params = [p for p in params if p.grad is not None]
    limit = math.inf if max_norm is None else float(max_norm)
    total_norm = torch.nn.utils.clip_grad_norm_(params, limit)
    return float(total_norm)
