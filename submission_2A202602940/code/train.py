"""train.py — đặt seed, đánh giá, vòng huấn luyện `run_experiment(cfg, data)`, dự đoán và ghi file nộp.

Mọi thí nghiệm chỉ là *đổi dict cfg* rồi gọi lại run_experiment (xem GUIDE, Part 2).
Mọi chỉ số (loss, accuracy, macro-F1) dùng cùng định nghĩa với scripts/evaluate.py.
"""
from __future__ import annotations

import math
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params
from optimizer import build_optimizer, build_scheduler, clip_gradients

N_CLASSES = 7
TRAIN_EVAL_SUBSET = 50_000   # train loss mỗi epoch đo trên tập con CỐ ĐỊNH này của train (eval mode)
TRAIN_EVAL_SEED = 0          # seed chọn tập con trên — cố định cho MỌI thí nghiệm, không phụ thuộc cfg["seed"]

# Cấu hình mặc định = BASELINE (M-base). `lr` được chọn bằng val ở Part 2 của notebook.
DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base",
    loss="ce",                 # "ce" | "mse"
    optimizer="sgd_momentum",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=None,                   # chọn bằng val, không dùng eval
    weight_decay=0.0, momentum=0.9,
    batch=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init="he",
    clip_norm=None,            # None = không clip; hoặc số, ví dụ 1.0
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    scheduler=None,            # None | "cosine" (tuỳ chọn, ghi vào notes nếu dùng)
    seed=1,
)


def set_seed(seed: int) -> None:
    """Đặt seed cho random, numpy, torch (và torch.cuda nếu có)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình cộng F1 của 7 lớp; F1_c = 2PR/(P+R), bằng 0 nếu P+R = 0.

    cm: ma trận nhầm lẫn (7, 7), hàng = nhãn thật, cột = dự đoán. Giống hệt scripts/evaluate.py.
    """
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diag(cm)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return float(f1.mean())


def confusion_matrix(y: torch.Tensor, pred: torch.Tensor, k: int = N_CLASSES) -> np.ndarray:
    """Ma trận nhầm lẫn k x k (hàng = thật, cột = dự đoán), tính bằng bincount ngay trên device."""
    return torch.bincount(y * k + pred, minlength=k * k).reshape(k, k).cpu().numpy()


def compute_loss(logits, y, loss_name: str, reduction: str = "mean"):
    """"ce"  : cross-entropy nhận logit thô và nhãn int64 (F.cross_entropy).
       "mse" : MSE giữa logit và one-hot của y, như nn.MSELoss: KHÔNG có hệ số 1/2, lấy trung bình trên
               MỌI phần tử (B x 7). Với reduction="sum" trả tổng trên mọi phần tử (evaluate tự chia N*7).
    Loss luôn tính bằng float32 (kể cả khi forward chạy trong autocast FP16/BF16).
    """
    logits = logits.float()
    if loss_name == "ce":
        return F.cross_entropy(logits, y, reduction=reduction)
    if loss_name == "mse":
        target = F.one_hot(y, num_classes=logits.shape[1]).float()
        return F.mse_loss(logits, target, reduction=reduction)
    raise ValueError(f"loss phải là 'ce' hoặc 'mse', nhận {loss_name!r}")


@torch.no_grad()
def predict(model, X, batch_size: int = 8192) -> torch.Tensor:
    """Trả về nhãn dự đoán int64 (N,) = argmax của logits (eval mode, không xáo)."""
    model.eval()
    return torch.cat([model(X[i:i + batch_size]).argmax(dim=1) for i in range(0, len(X), batch_size)])


@torch.no_grad()
def evaluate(model, X, y, loss_name: str = "ce", batch_size: int = 8192) -> dict:
    """Trả về dict(loss, acc, macro_f1) ở chế độ eval() (dropout tắt) và no_grad.

    Loss = tổng loss từng lô (reduction="sum") chia cho N (CE) hoặc N*7 (MSE) — cùng thang với loss lúc train.
    Dùng hàm này cho: train loss (tập con cố định của train), val, và eval cuối cùng.
    """
    model.eval()
    total, preds = 0.0, []
    for i in range(0, len(X), batch_size):
        logits = model(X[i:i + batch_size])
        total += compute_loss(logits, y[i:i + batch_size], loss_name, reduction="sum").item()
        preds.append(logits.argmax(dim=1))
    pred = torch.cat(preds)
    n_terms = len(X) * (N_CLASSES if loss_name == "mse" else 1)
    cm = confusion_matrix(y, pred)
    return dict(loss=total / n_terms, acc=float(np.trace(cm) / cm.sum()), macro_f1=macro_f1_from_confusion(cm))


def _train_eval_subset(data: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Tập con cố định TRAIN_EVAL_SUBSET mẫu của train để đo train loss (giống nhau cho mọi thí nghiệm)."""
    if "_tr_sub" not in data:
        n = len(data["X_tr"])
        g = torch.Generator().manual_seed(TRAIN_EVAL_SEED)
        idx = torch.randperm(n, generator=g)[:min(TRAIN_EVAL_SUBSET, n)].to(data["X_tr"].device)
        data["_tr_sub"] = (data["X_tr"][idx], data["y_tr"][idx])
    return data["_tr_sub"]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_experiment(cfg: dict, data: dict, verbose: bool = True) -> dict:
    """Huấn luyện một cấu hình và trả về lịch sử + tóm tắt.

    Args:
        cfg : dict cấu hình (khoá thiếu lấy từ DEFAULT_CFG)
        data: kết quả của data.prepare_data (tensor X_tr, y_tr, X_val, y_val trên device). X_eval KHÔNG được dùng.

    Trả về dict:
        {"cfg": cfg,
         "history": {"epoch", "train_loss", "val_loss", "val_acc", "val_macro_f1",
                     "grad_norm" (trung bình mỗi epoch, TRƯỚC clip), "grad_norm_max", "clip_frac", "epoch_time_s"},
         "summary": {"step0_loss", "best_val_loss", "best_epoch", "final_train_loss", "final_val_loss",
                     "val_acc", "val_macro_f1", "time_per_epoch_s", "peak_mem_MB", "diverged", ...},
         "best_state": state_dict (trên CPU) của epoch có val_loss thấp nhất — giữ trong RAM, không ghi JSON}

    Quy ước đo:
      - train_loss: eval mode, tập con cố định 50 000 mẫu train (TRAIN_EVAL_SUBSET), cùng thang với val_loss.
      - epoch_time_s: chỉ thời gian của vòng cập nhật trên train (không tính phần đánh giá), có synchronize.
      - best epoch chọn theo val_loss thấp nhất; val_acc / val_macro_f1 trong summary lấy tại epoch đó.
      - diverged: loss của một lô thành NaN/inf (hoặc val_loss không hữu hạn) -> dừng ngay.
    """
    cfg = {**DEFAULT_CFG, **cfg}
    cfg["hidden"] = tuple(cfg["hidden"])
    X_tr, y_tr, X_val, y_val = data["X_tr"], data["y_tr"], data["X_val"], data["y_val"]
    device = X_tr.device
    X_sub, y_sub = _train_eval_subset(data)

    # ---- 0. model, optimizer, (scheduler), scaler
    set_seed(cfg["seed"])
    model = MLP(hidden=cfg["hidden"], dropout=cfg["dropout"], init=cfg["init"])
    n_params = count_params(model)
    if cfg["hidden"] in EXPECTED_PARAMS:
        assert n_params == EXPECTED_PARAMS[cfg["hidden"]], f"số tham số {n_params} != {EXPECTED_PARAMS[cfg['hidden']]}"
    model.to(device)
    optimizer = build_optimizer(cfg["optimizer"], model.parameters(), cfg["lr"],
                                weight_decay=cfg["weight_decay"], momentum=cfg["momentum"])
    steps_per_epoch = math.ceil(len(X_tr) / cfg["batch"])
    scheduler = build_scheduler(optimizer, cfg.get("scheduler"), steps_per_epoch * cfg["epochs"])

    precision = cfg["precision"]
    amp_dtype = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]
    use_scaler = precision == "fp16" and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_scaler else None
    gen = torch.Generator(device=device).manual_seed(cfg["seed"])   # thứ tự lô lặp lại được theo seed

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # ---- 1. loss bước 0 trên val, TRƯỚC bước cập nhật đầu tiên (kỳ vọng ≈ ln 7 với CE)
    step0 = evaluate(model, X_val, y_val, cfg["loss"])
    hist = {k: [] for k in ("epoch", "train_loss", "val_loss", "val_acc", "val_macro_f1",
                            "grad_norm", "grad_norm_max", "clip_frac", "epoch_time_s")}
    best_val, best_epoch, best_state = math.inf, 0, None
    diverged, diverged_at, skipped_steps = False, None, 0

    if verbose:
        print(f"[{cfg['exp_id']}] {cfg['optimizer']} lr={cfg['lr']} batch={cfg['batch']} loss={cfg['loss']} "
              f"init={cfg['init']} dropout={cfg['dropout']} clip={cfg['clip_norm']} {precision} seed={cfg['seed']} "
              f"| {n_params:,} tham số | step0 val_loss={step0['loss']:.4f}")

    # ---- 2. vòng huấn luyện
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        _sync(device)
        t0 = time.perf_counter()
        norms, n_clipped = [], 0
        for step, (xb, yb) in enumerate(iterate_batches(X_tr, y_tr, cfg["batch"], generator=gen), start=1):
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(xb)
            loss = compute_loss(logits, yb, cfg["loss"])          # loss luôn ở float32
            if not torch.isfinite(loss):
                diverged, diverged_at = True, f"epoch {epoch}, bước {step}"
                break
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)                         # đưa gradient về thang thật TRƯỚC khi đo/cắt
            else:
                loss.backward()
            gn = clip_gradients(model.parameters(), cfg["clip_norm"])   # chuẩn TRƯỚC khi cắt
            if math.isfinite(gn):
                norms.append(gn)
                n_clipped += cfg["clip_norm"] is not None and gn > cfg["clip_norm"]
            elif scaler is None:                                   # FP32/BF16: gradient inf/NaN = phân kỳ
                diverged, diverged_at = True, f"epoch {epoch}, bước {step} (grad_norm={gn})"
                break
            else:                                                  # FP16: GradScaler sẽ bỏ qua bước này
                skipped_steps += 1
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
        _sync(device)
        epoch_time = time.perf_counter() - t0

        # ---- cuối epoch: đo ở eval mode, không gradient
        tr = evaluate(model, X_sub, y_sub, cfg["loss"]) if not diverged else dict(loss=math.nan)
        va = evaluate(model, X_val, y_val, cfg["loss"]) if not diverged else dict(loss=math.nan, acc=math.nan, macro_f1=math.nan)
        if not diverged and not math.isfinite(va["loss"]):
            diverged, diverged_at = True, f"epoch {epoch} (val_loss={va['loss']})"
        hist["epoch"].append(epoch)
        hist["train_loss"].append(tr["loss"])
        hist["val_loss"].append(va["loss"])
        hist["val_acc"].append(va["acc"])
        hist["val_macro_f1"].append(va["macro_f1"])
        hist["grad_norm"].append(float(np.mean(norms)) if norms else math.nan)
        hist["grad_norm_max"].append(float(np.max(norms)) if norms else math.nan)
        hist["clip_frac"].append(n_clipped / max(len(norms), 1))
        hist["epoch_time_s"].append(epoch_time)

        if not diverged and va["loss"] < best_val:
            best_val, best_epoch = va["loss"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if verbose:
            msg = (f"  ep {epoch:2d} | train {tr['loss']:.4f} | val {va['loss']:.4f} acc {va['acc']:.4f} "
                   f"F1 {va['macro_f1']:.4f} | gn {hist['grad_norm'][-1]:.3f} (max {hist['grad_norm_max'][-1]:.2f})")
            if cfg["clip_norm"] is not None:
                msg += f" clip {hist['clip_frac'][-1]:.0%}"
            print(msg + f" | {epoch_time:.1f}s" + (" *" if best_epoch == epoch else ""))
        if diverged:
            if verbose:
                print(f"  -> PHÂN KỲ tại {diverged_at}; dừng lần chạy.")
            break

    # ---- 3. tóm tắt tại best epoch
    bi = best_epoch - 1
    peak_mem = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else math.nan
    finite_times = [t for t in hist["epoch_time_s"] if math.isfinite(t)]
    summary = dict(
        step0_loss=step0["loss"],
        best_val_loss=best_val if best_epoch else math.nan,
        best_epoch=best_epoch,
        final_train_loss=hist["train_loss"][-1] if hist["train_loss"] else math.nan,
        final_val_loss=hist["val_loss"][-1] if hist["val_loss"] else math.nan,
        val_acc=hist["val_acc"][bi] if best_epoch else math.nan,
        val_macro_f1=hist["val_macro_f1"][bi] if best_epoch else math.nan,
        time_per_epoch_s=float(np.mean(finite_times)) if finite_times else math.nan,
        peak_mem_MB=peak_mem,
        diverged=diverged,
        diverged_at=diverged_at,
        epochs_run=len(hist["epoch"]),
        steps_per_epoch=steps_per_epoch,
        n_params=n_params,
        fp16_skipped_steps=skipped_steps,
    )
    if verbose:
        print(f"  => best epoch {best_epoch}: val_loss {summary['best_val_loss']:.4f} | val_acc {summary['val_acc']:.4f} "
              f"| val_macro_f1 {summary['val_macro_f1']:.4f} | {summary['time_per_epoch_s']:.2f}s/epoch "
              f"| peak {summary['peak_mem_MB']:.0f} MB")
    return {"cfg": cfg, "history": hist, "summary": summary, "best_state": best_state}


def write_predictions(row_id, preds, path: str) -> None:
    """Ghi file nộp cho scripts/evaluate.py: CSV có tiêu đề `row_id,pred` (pred là 0..6, cùng thứ tự với row_id)."""
    row_id = np.asarray(row_id, dtype=np.int64)
    preds = np.asarray(preds, dtype=np.int64)
    assert row_id.shape == preds.shape, f"row_id {row_id.shape} và preds {preds.shape} phải cùng độ dài"
    assert len(np.unique(row_id)) == len(row_id), "row_id bị lặp"
    assert preds.min() >= 0 and preds.max() <= N_CLASSES - 1, "pred phải nằm trong 0..6"
    pd.DataFrame({"row_id": row_id, "pred": preds}).to_csv(path, index=False)


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str) -> np.ndarray:
    """Dùng MỘT LẦN cho cấu hình cuối cùng (và baseline): nạp best_state, dự đoán eval, ghi predictions.

    Sau đó chạy `python scripts/evaluate.py --pred <pred_path> --out ...` (trong notebook) để có điểm chính thức.
    Trả về mảng dự đoán (numpy).
    """
    cfg = {**DEFAULT_CFG, **cfg}
    assert result.get("best_state") is not None, "result không có best_state (lần chạy phân kỳ ngay từ đầu?)"
    model = MLP(hidden=tuple(cfg["hidden"]), dropout=cfg["dropout"], init="default")
    model.load_state_dict(result["best_state"])
    model.to(data["X_eval"].device)
    preds = predict(model, data["X_eval"]).cpu().numpy()      # fp32, eval mode
    write_predictions(data["eval_row_id"], preds, pred_path)
    return preds


def run_and_log(cfg: dict, data: dict, out_dir: str = "..", verbose: bool = True) -> dict:
    """run_experiment + lưu ngay results/<exp_id>.json và figures/<exp_id>.png (mỗi lần chạy một ảnh)."""
    from plots import plot_run
    from results_table import save_result

    result = run_experiment(cfg, data, verbose=verbose)
    save_result(result, f"{out_dir}/results")
    plot_run(result, f"{out_dir}/figures/{result['cfg']['exp_id']}.png")
    return result
