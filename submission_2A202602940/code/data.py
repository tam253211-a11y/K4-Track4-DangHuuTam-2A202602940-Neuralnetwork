"""data.py — nạp tập train/eval đã chia sẵn, tách validation từ train, chuẩn hoá, đưa lên thiết bị.

Điều kiện trước: đã chạy `python scripts/split_data.py` (tạo data/processed/train.npz, eval.npz).

Quy ước dữ liệu (xem README mục 2 và 3):
    X : float32, shape (N, 54)   — 10 cột đầu là số liên tục, 44 cột sau là nhị phân (one-hot)
    y : int64,   shape (N,)      — nhãn 0..6
Tập eval CHỈ dùng để chấm điểm cuối. Không dùng nó để chọn cấu hình, chuẩn hoá hay dừng sớm.
"""
from __future__ import annotations

import numpy as np
import torch

N_NUMERIC = 10    # số cột liên tục cần chuẩn hoá (cột 0..9)
N_FEATURES = 54
N_CLASSES = 7
N_TRAIN_FULL, N_EVAL = 464_809, 116_203


def _check_xy(X, y, name):
    assert X.ndim == 2 and X.shape[1] == N_FEATURES, f"{name}: X phải có shape (N, {N_FEATURES}), hiện là {X.shape}"
    assert X.dtype == np.float32, f"{name}: X phải là float32, hiện là {X.dtype}"
    assert y.shape == (len(X),), f"{name}: y phải có shape ({len(X)},), hiện là {y.shape}"
    assert y.dtype == np.int64, f"{name}: y phải là int64, hiện là {y.dtype}"
    assert y.min() >= 0 and y.max() <= N_CLASSES - 1, f"{name}: nhãn phải trong 0..6, hiện là {y.min()}..{y.max()}"


def load_split(processed_dir: str = "data/processed"):
    """Nạp train và eval từ file .npz.

    Trả về: X_train_full, y_train_full, X_eval, y_eval, eval_row_id
    Nhãn đã được split_data.py đổi về 0..6, nên ở đây KHÔNG trừ 1 nữa.
    eval_row_id giữ nguyên thứ tự dòng của eval.npz để ghép với dự đoán ở Part 4.
    """
    with np.load(f"{processed_dir}/train.npz") as tr:
        X_train_full, y_train_full = tr["X"], tr["y"]
    with np.load(f"{processed_dir}/eval.npz") as ev:
        X_eval, y_eval, eval_row_id = ev["X"], ev["y"], ev["row_id"]

    _check_xy(X_train_full, y_train_full, "train")
    _check_xy(X_eval, y_eval, "eval")
    assert len(X_train_full) == N_TRAIN_FULL, f"train phải có {N_TRAIN_FULL} mẫu, hiện có {len(X_train_full)}"
    assert len(X_eval) == N_EVAL, f"eval phải có {N_EVAL} mẫu, hiện có {len(X_eval)}"
    assert eval_row_id.shape == (N_EVAL,) and len(np.unique(eval_row_id)) == N_EVAL, "row_id của eval phải duy nhất"
    return X_train_full, y_train_full, X_eval, y_eval, eval_row_id


def make_val_split(X, y, val_fraction: float = 0.2, seed: int = 42):
    """Tách validation TỪ train (không đụng eval). Phân tầng theo nhãn.

    Trả về: X_tr, y_tr, X_val, y_val
    Seed tách val luôn là 42 cho mọi thí nghiệm; seed huấn luyện (đổi khi đo nhiễu) là chuyện riêng.
    """
    from sklearn.model_selection import train_test_split

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_fraction, stratify=y, random_state=seed, shuffle=True)
    return X_tr, y_tr, X_val, y_val


def fit_standardizer(X_tr):
    """Tính mean và std của N_NUMERIC cột đầu CHỈ trên tập train (sau khi tách val).

    Trả về: mean (shape (10,)), std (shape (10,)), float32.
    Không tính trên val/eval: thống kê của chúng sẽ rò rỉ thông tin về dữ liệu dùng để chọn cấu hình / chấm điểm,
    làm số đo trên val/eval lạc quan hơn thực tế. Mô hình triển khai thật cũng chỉ biết thống kê của dữ liệu huấn luyện.
    """
    num = X_tr[:, :N_NUMERIC].astype(np.float64)      # cộng dồn bằng float64 cho chính xác
    mean = num.mean(axis=0)
    std = num.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)              # cột hằng số: chia cho 1 thay vì 0
    return mean.astype(np.float32), std.astype(np.float32)


def apply_standardizer(X, mean, std):
    """Trả về bản sao của X, trong đó 10 cột đầu được (x - mean) / std; 44 cột nhị phân giữ nguyên."""
    out = np.array(X, dtype=np.float32, copy=True)    # không sửa X gốc tại chỗ
    out[:, :N_NUMERIC] = (out[:, :N_NUMERIC] - mean) / std
    return out


def prepare_data(device, val_fraction: float = 0.2, seed: int = 42,
                 processed_dir: str = "data/processed", verbose: bool = True) -> dict:
    """Gộp các bước trên và đưa TOÀN BỘ dữ liệu lên `device` một lần (không dùng DataLoader).

    Trả về dict gồm các tensor trên device:
        X_tr, y_tr, X_val, y_val, X_eval, y_eval        (X float32, y int64)
    cùng các mảng numpy: eval_row_id, mean, std, majority_class, majority_val_acc
    """
    X_full, y_full, X_eval, y_eval, eval_row_id = load_split(processed_dir)
    X_tr, y_tr, X_val, y_val = make_val_split(X_full, y_full, val_fraction, seed)

    mean, std = fit_standardizer(X_tr)                # chỉ X_tr
    X_tr_s = apply_standardizer(X_tr, mean, std)      # cùng mean/std cho cả ba tập
    X_val_s = apply_standardizer(X_val, mean, std)
    X_eval_s = apply_standardizer(X_eval, mean, std)

    # mốc thấp nhất: luôn đoán lớp phổ biến nhất CỦA TRAIN, đo trên val
    majority_class = int(np.bincount(y_tr, minlength=N_CLASSES).argmax())
    majority_val_acc = float((y_val == majority_class).mean())

    def to_x(a): return torch.tensor(a, dtype=torch.float32, device=device)
    def to_y(a): return torch.tensor(a, dtype=torch.int64, device=device)

    data = {
        "X_tr": to_x(X_tr_s), "y_tr": to_y(y_tr),
        "X_val": to_x(X_val_s), "y_val": to_y(y_val),
        "X_eval": to_x(X_eval_s), "y_eval": to_y(y_eval),
        "eval_row_id": eval_row_id, "mean": mean, "std": std,
        "majority_class": majority_class, "majority_val_acc": majority_val_acc,
    }
    if verbose:
        for k in ("tr", "val", "eval"):
            print(f"X_{k:<4s} {tuple(data['X_' + k].shape)} {data['X_' + k].dtype} | "
                  f"y_{k:<4s} {tuple(data['y_' + k].shape)} {data['y_' + k].dtype} | device {data['X_' + k].device}")
        print(f"Đoán luôn lớp đa số (lớp {majority_class}) -> val accuracy = {majority_val_acc:.4f}")
    return data


def iterate_batches(X, y, batch_size: int, generator: torch.Generator | None = None,
                    shuffle: bool = True, drop_last: bool = False):
    """Generator trả về từng cặp (xb, yb), thay cho DataLoader.

    Lô cuối: mặc định GIỮ lô cuối nhỏ hơn batch_size (drop_last=False), để mỗi epoch dùng đủ mọi mẫu train.
    Với 371 847 mẫu và batch 512: 726 lô đủ + 1 lô 135 mẫu = 727 bước/epoch. Loss lấy trung bình trên lô nên
    lô nhỏ không làm sai thang đo, chỉ có gradient nhiễu hơn một chút ở một bước trong 727.
    Xáo lại mỗi lần gọi (mỗi epoch). Truyền `generator` cùng device với X để thứ tự lặp lại được theo seed.
    """
    n = len(X)
    if shuffle:
        perm = torch.randperm(n, generator=generator, device=X.device)
    else:
        perm = torch.arange(n, device=X.device)
    stop = (n // batch_size) * batch_size if drop_last else n
    for i in range(0, stop, batch_size):
        idx = perm[i:i + batch_size]
        yield X[idx], y[idx]
