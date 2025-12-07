import os
import json
import math
import argparse
import random
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from ct_model import build_ct_model
from data_utils import apply_pgd_preprocess
import matplotlib.pyplot as plt
import csv
import torch.nn.functional as F
from torch.distributions import Beta



def set_seeds(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

def normalize_event_name(name: str) -> str:
    return name.strip().lower()


def build_aligned_dataset(
    pgd_npz_path: str,
    time_len: int = 200,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """按样本级对齐：直接使用 PGD NPZ 中的 `sample_labels_shifted` 作为 y，
    并使用 `event_ids` 将每个样本映射到唯一事件名。返回 (X, y, event_names, event_coords_deg, station_coords_deg)。
    X 和 y 都被裁剪/填充到 time_len。
    """
    pgd_npz = np.load(pgd_npz_path, allow_pickle=True)

    # PGD 样本级数据与事件映射
    X_all = pgd_npz['pgd_dataset']              # [N_samples, T_pgd, F]
    event_ids = pgd_npz['event_ids']            # [N_samples]
    event_names_unique = pgd_npz['event_names'] # [N_events]
    # 样本级台站坐标（可选）
    has_sta_coords = 'sample_station_coords' in pgd_npz.files
    sta_coords_all = pgd_npz['sample_station_coords'] if has_sta_coords else None  # [N_samples, N, 2]
    # 事件坐标（度）
    # 坐标可能缺失，做兼容处理
    if 'event_lats' in pgd_npz.files and 'event_lons' in pgd_npz.files:
        event_lats = pgd_npz['event_lats']      # [N_events]
        event_lons = pgd_npz['event_lons']      # [N_events]
    else:
        print("警告: NPZ中缺少 event_lats/event_lons，将使用零坐标占位")
        event_lats = np.zeros((len(event_names_unique),), dtype=np.float32)
        event_lons = np.zeros((len(event_names_unique),), dtype=np.float32)
    names_unique_norm = [normalize_event_name(n) for n in event_names_unique.tolist()]

    # 直接使用样本级标签
    if 'sample_labels_shifted' not in pgd_npz.files:
        raise RuntimeError("PGD NPZ 缺少 'sample_labels_shifted' 字段，请使用带样本级标签的数据集")
    y_all = pgd_npz['sample_labels_shifted']    # [N_samples, T_label]

    X_list, y_list, ev_list, ev_coords_list, sta_coords_list = [], [], [], [], []
    n_samples = X_all.shape[0]

    for i in range(n_samples):
        ev_idx = int(event_ids[i])
        if ev_idx < 0 or ev_idx >= len(names_unique_norm):
            continue
        ev_name_norm = names_unique_norm[ev_idx]

        x_i = X_all[i]
        y_i = y_all[i]

        # 裁剪/填充到 time_len
        T_x = x_i.shape[0]
        T_y = y_i.shape[0]
        if T_x < time_len:
            pad_x = np.zeros((time_len - T_x, x_i.shape[1]), dtype=x_i.dtype)
            x_i = np.concatenate([x_i, pad_x], axis=0)
        else:
            x_i = x_i[:time_len]
        if T_y < time_len:
            pad_y = np.full((time_len - T_y,), y_i[-1] if T_y > 0 else 0.0, dtype=y_i.dtype)
            y_i = np.concatenate([y_i, pad_y], axis=0)
        else:
            y_i = y_i[:time_len]

        X_list.append(x_i)
        y_list.append(y_i)
        ev_list.append(ev_name_norm)
        # 记录该样本对应事件的坐标（lat, lon）
        ev_coords_list.append([float(event_lats[ev_idx]), float(event_lons[ev_idx])])
        # 样本级台站坐标（若存在）
        if has_sta_coords:
            sta_coords_list.append(sta_coords_all[i])

    X = np.stack(X_list, axis=0)
    y = np.stack(y_list, axis=0)
    names = np.array(ev_list)
    ev_coords = np.array(ev_coords_list, dtype=np.float32)  # [N_samples_aligned, 2]
    # 对齐后的台站坐标（若原文件未提供，则用零填充以便模型前向兼容）
    if has_sta_coords and len(sta_coords_list) > 0:
        sta_coords = np.array(sta_coords_list, dtype=np.float32)  # [N_samples_aligned, N, 2]
    else:
        # 推断台站数（F//2）并填充为零
        num_stations = X_all.shape[2] // 2
        sta_coords = np.zeros((len(ev_coords_list), num_stations, 2), dtype=np.float32)

    # 对齐统计输出，便于核验
    print(f"PGD samples: {n_samples}, unique events: {len(names_unique_norm)}, aligned samples: {len(names)}")

    return X, y, names, ev_coords, sta_coords


def gaussian_nll(mu: torch.Tensor, log_var: torch.Tensor, y: torch.Tensor, var_floor: float, time_weights: torch.Tensor) -> torch.Tensor:
    # mu, log_var, y: [B, T]; time_weights: [T] or [B, T]
    sigma2 = torch.exp(log_var).clamp_min(var_floor)
    nll = 0.5 * ((y - mu) ** 2 / sigma2 + torch.log(sigma2))
    if time_weights.dim() == 1:
        tw = time_weights.unsqueeze(0).expand_as(nll)
    else:
        tw = time_weights
    loss = torch.sum(nll * tw) / torch.sum(tw)
    return loss


def random_zero_stations_in_batch(
    x: torch.Tensor,
    num_stations: int,
    max_frac: float = 0.5,
    alpha: float = 2.0,
    beta: float = 8.0,
) -> torch.Tensor:
    """随机置零部分台站（PGD与状态两通道均置零）。
    - x: [B, T, N*2]
    - num_stations: N
    - max_frac: 每个样本最多置零比例（0-1），默认不超过0.5
    - alpha, beta: Beta分布参数，倾向小比例，使置零台站多的样本占少数
    返回增强后的新张量，不修改原张量。
    """
    B, T, FEAT = x.shape
    N2 = num_stations * 2
    if FEAT != N2:
        raise ValueError(f"Expected last dim {N2}, got {FEAT}")
    device = x.device
    N = num_stations
    # 保证比例范围
    max_frac = float(max(0.0, min(max_frac, 1.0)))
    # Beta分布采样（偏向小比例）
    dist = Beta(alpha, beta)
    fracs = dist.sample((B,)).to(device) * max_frac
    x_rs = x.view(B, T, N, 2).clone()
    for b in range(B):
        # 计算置零台站数量（四舍五入），并裁剪到 [0, N]
        k = int(torch.clamp((fracs[b] * N).round(), min=0, max=N).item())
        if k <= 0:
            continue
        idx = torch.randperm(N, device=device)[:k]
        x_rs[b, :, idx, :] = 0.0
    return x_rs.view(B, T, FEAT)

def mask_keep_stations_in_batch(x: torch.Tensor, num_stations: int, keep: int) -> torch.Tensor:
    B, T, FEAT = x.shape
    N = num_stations
    K = int(max(0, min(int(keep), N)))
    if K >= N:
        return x
    x2 = x.view(B, T, N, 2).clone()
    device = x2.device
    for b in range(B):
        idx = torch.randperm(N, device=device)[:K]
        mask = torch.ones(N, dtype=torch.bool, device=device)
        mask[idx] = False
        x2[b, :, mask, :] = 0.0
    return x2.view(B, T, FEAT)


def compute_sequence_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    # y_*: [E, T]
    err = y_pred - y_true
    mae_seq = float(np.mean(np.abs(err)))
    rmse_seq = float(np.sqrt(np.mean(err ** 2)))
    mae_final = float(np.mean(np.abs(err[:, -1])))
    rmse_final = float(np.sqrt(np.mean(err[:, -1] ** 2)))
    return {
        'mae_seq': mae_seq,
        'rmse_seq': rmse_seq,
        'mae_final': mae_final,
        'rmse_final': rmse_final,
    }


def plot_loss_curve(train_losses, val_losses, out_path: str):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,4))
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='val')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_batch_log(out_dir: str, epoch: int, train_batches: list, val_batches: list):
    """保存单个epoch的batch级日志到CSV。train_batches项为(loss, grad_norm, lr)，val_batches项为(loss,)"""
    path = os.path.join(out_dir, f'batch_loss_epoch_{epoch:03d}.csv')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['phase', 'batch_idx', 'loss', 'grad_norm', 'lr'])
        for i, (loss, grad, lr) in enumerate(train_batches):
            w.writerow(['train', i, float(loss), float(grad), float(lr)])
        for i, (loss,) in enumerate(val_batches):
            w.writerow(['val', i, float(loss), '', ''])
    return path


def load_station_coords_csv(csv_path: str, num_stations: int) -> torch.Tensor:
    """加载台站坐标CSV，返回形状 [num_stations, 2] 的 (lat_deg, lon_deg)。
    注意：此函数仅作为回退用途，假设CSV的台站顺序与数据集中通道顺序一致。
    CSV格式需包含列名：station, lat_deg, lon_deg。
    """
    coords = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    lat = float(row.get('lat_deg'))
                    lon = float(row.get('lon_deg'))
                except Exception:
                    lat = float(row.get('lat') or 0.0)
                    lon = float(row.get('lon') or 0.0)
                coords.append([lat, lon])
                if len(coords) >= num_stations:
                    break
        if len(coords) < num_stations:
            # 用0填充到指定数量
            coords.extend([[0.0, 0.0]] * (num_stations - len(coords)))
    except Exception as e:
        print(f"读取CSV失败: {e}. 将返回全零坐标作为占位。")
        coords = [[0.0, 0.0]] * num_stations
    return torch.tensor(coords, dtype=torch.float32)



def main():
    parser = argparse.ArgumentParser(description='Train Transformer-based CT Mw(t) model with heteroscedastic loss')
    # Data
    parser.add_argument('--pgd_npz', type=str, default=r'F:\magnitude_prediction\dataset\all_events_pgd_dataset_sta_16_with_labels.npz')
    # 已将标签并入 PGD NPZ（键：sample_labels_shifted），不再单独传 label_npz
    parser.add_argument('--time_len', type=int, default=200)
    # Preprocess
    parser.add_argument('--log10_pgd', action='store_true', help='Apply log10 scaling to PGD (recommended)')
    parser.add_argument('--layernorm_pgd', action='store_true', help='LayerNorm-style normalization over stations')
    # Model
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--ffn_dim', type=int, default=256)
    parser.add_argument('--post_units', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--input_dropout', type=float, default=0.0)
    parser.add_argument('--use_station_attention', action='store_true')
    parser.add_argument('--attn_hidden', type=int, default=16)
    parser.add_argument('--normalize_station_attention', action='store_true', help='对台站注意力进行归一化（Masked Softmax），降低台站数量变化影响')
    parser.add_argument('--attn_temperature', type=float, default=1.0, help='台站注意力Softmax温度参数（>0）')
    parser.add_argument('--hard_mask', action='store_true', help='Hard gate offline stations by status')
    parser.add_argument('--heteroscedastic', action='store_true', help='Predict mean+variance with Gaussian NLL loss')
    parser.add_argument('--var_floor', type=float, default=1e-4)
    parser.add_argument('--use_causal_mask', action='store_true', help='Use causal attention mask for streaming consistency')
    # Geo features (改进版：只使用台站位置信息)
    parser.add_argument('--use_geo', action='store_true', help='启用基于台站位置的地理特征（包括坐标编码和距离统计）')
    parser.add_argument('--coords_csv', type=str, default=None, help='台站坐标CSV路径（可选回退；默认使用PGD NPZ中的样本级坐标）')
    parser.add_argument('--geo_mode', type=str, default='xyz', choices=['xyz', 'raw', 'sin_cos'], help='台站坐标编码方式')
    parser.add_argument('--geo_embed_dim', type=int, default=16, help='台站地理嵌入维度（已增强以补偿移除事件特征）')
    parser.add_argument('--pgd_clip', type=float, default=None, help='对门控加权后的PGD进行对称裁剪以抑制尖峰（默认关闭）')
    # Train
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Adam weight decay')
    parser.add_argument('--train_frac', type=float, default=0.7)
    parser.add_argument('--val_frac', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--early_patience', type=int, default=15)
    parser.add_argument('--min_delta', type=float, default=1e-4, help='validation loss minimal improvement to reset patience')
    parser.add_argument('--use_scheduler', action='store_true', help='use ReduceLROnPlateau scheduler on val loss')
    # Early-time weighting
    parser.add_argument('--early_weight', type=float, default=3.0, help='Weight multiplier for early time steps')
    parser.add_argument('--early_frac', type=float, default=0.5, help='Fraction of sequence considered early')
    # Augmentation: 随机置零台站（训练阶段）
    parser.add_argument('--rand_zero_stations', action='store_true', help='训练时随机置零部分台站（PGD与状态码均置零）')
    parser.add_argument('--zero_max_frac', type=float, default=0.5, help='最多置零比例（0-1，默认0.5）')
    parser.add_argument('--zero_beta_a', type=float, default=2.0, help='Beta分布alpha，偏向较小比例')
    parser.add_argument('--zero_beta_b', type=float, default=8.0, help='Beta分布beta，偏向较小比例')
    # Output
    parser.add_argument('--out_dir', type=str, default=r'f:\magnitude_prediction\model_new\runs')
    parser.add_argument('--test_sta_num', type=int, nargs='+', default=None)

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    set_seeds(args.seed)

    # Load and align
    X_np, y_np, names, ev_coords_np, sta_coords_np = build_aligned_dataset(args.pgd_npz, time_len=args.time_len)
    E, T, F = X_np.shape
    num_stations = F // 2

    # 从 NPZ 读取强制测试样本索引提示（若存在）
    try:
        _npz_meta = np.load(args.pgd_npz, allow_pickle=True)
        if 'test_indices_hint' in _npz_meta.files:
            _forced_hint_raw = _npz_meta['test_indices_hint']
            forced_test_hint = [int(i) for i in (\
                _forced_hint_raw.tolist() if hasattr(_forced_hint_raw, 'tolist') else list(_forced_hint_raw)
            ) if 0 <= int(i) < E]
        else:
            forced_test_hint = []
    except Exception:
        forced_test_hint = []

    # Preprocess PGD
    X_np = apply_pgd_preprocess(X_np, num_stations=num_stations, use_layernorm=args.layernorm_pgd)


    # 测试X_np是否归一化
    # plt.figure(figsize=(6,4))
    # for i in range(num_stations):
    #     plt.plot(X_np[0, :, i], label=f'sta {i}')
    # plt.title('PGD Distribution after Preprocessing')
    # plt.tight_layout()
    # plt.show()

    # 测试y_np是否归一化
    # plt.figure(figsize=(6,4))
    # plt.plot(y_np[0])
    # plt.title('PGD Distribution after Preprocessing')
    # plt.tight_layout()
    # plt.show()
    # Normalize labels
    y_mean = float(np.mean(y_np))
    y_std = float(np.std(y_np) + 1e-8)
    y_norm = (y_np - y_mean) / y_std

    # Split（按事件分组）：
    # - 每个事件的样本数 < 3：全部放入训练集
    # - 每个事件的样本数 ≥ 3：按照 train/val/test 比例划分，且三者至少各有一个
    # 保持可复现：按种子进行分组内打乱
    rng = np.random.default_rng(args.seed)
    # 构建“事件名 → 样本索引”映射
    ev_to_indices = {}
    for i, n in enumerate(names.tolist()):
        ev_to_indices.setdefault(n, []).append(i)

    train_idx_list, val_idx_list, test_idx_list = [], [], []
    train_frac = float(args.train_frac)
    val_frac = float(args.val_frac)
    test_frac = max(0.0, 1.0 - train_frac - val_frac)
    for ev, idxs in ev_to_indices.items():
        idxs = list(idxs)
        rng.shuffle(idxs)
        if len(idxs) < 3:
            # 样本数不足 3，全进训练集
            train_idx_list.extend(idxs)
            continue
        # 先保证三个集合至少各有一个
        base_train = [idxs[0]]
        base_val = [idxs[1]]
        base_test = [idxs[2]]
        remain = idxs[3:]
        L = len(remain)
        # 余下样本按比例分配（向下取整），剩余给 test
        extra_train = int(math.floor(L * train_frac))
        extra_val = int(math.floor(L * val_frac))
        extra_test = L - extra_train - extra_val
        train_idx_list.extend(base_train + remain[:extra_train])
        val_idx_list.extend(base_val + remain[extra_train:extra_train + extra_val])
        test_idx_list.extend(base_test + remain[extra_train + extra_val:])

    # 强制将 NPZ 中的提示样本并入测试集，并从训练/验证集中移除
    if len(forced_test_hint) > 0:
        _forced_set = set(forced_test_hint)
        train_idx_list = [i for i in train_idx_list if i not in _forced_set]
        val_idx_list = [i for i in val_idx_list if i not in _forced_set]
        # 追加到测试集并去重
        test_idx_list = list(dict.fromkeys(test_idx_list + list(_forced_set)))

    train_idx = np.array(train_idx_list, dtype=int)
    val_idx = np.array(val_idx_list, dtype=int)
    test_idx = np.array(test_idx_list, dtype=int)

    X_train = torch.tensor(X_np[train_idx], dtype=torch.float32)
    y_train = torch.tensor(y_norm[train_idx], dtype=torch.float32)
    EV_train = torch.tensor(ev_coords_np[train_idx], dtype=torch.float32)
    ST_train = torch.tensor(sta_coords_np[train_idx], dtype=torch.float32)
    X_val = torch.tensor(X_np[val_idx], dtype=torch.float32)
    y_val = torch.tensor(y_norm[val_idx], dtype=torch.float32)
    EV_val = torch.tensor(ev_coords_np[val_idx], dtype=torch.float32)
    ST_val = torch.tensor(sta_coords_np[val_idx], dtype=torch.float32)
    X_test = torch.tensor(X_np[test_idx], dtype=torch.float32)
    y_test = torch.tensor(y_norm[test_idx], dtype=torch.float32)
    EV_test = torch.tensor(ev_coords_np[test_idx], dtype=torch.float32)
    ST_test = torch.tensor(sta_coords_np[test_idx], dtype=torch.float32)
    # 为各数据集保留事件名以便写出 CSV
    names_train = names[train_idx]
    names_val = names[val_idx]
    names_test = names[test_idx]

    train_loader = DataLoader(TensorDataset(X_train, y_train, EV_train, ST_train), batch_size=args.batch_size, shuffle=True)
    # 评估时使用不打乱的 loader，保证样本顺序与 names_* 对齐
    train_eval_loader = DataLoader(TensorDataset(X_train, y_train, EV_train, ST_train), batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(TensorDataset(X_val, y_val, EV_val, ST_val), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test, EV_test, ST_test), batch_size=args.batch_size, shuffle=False)

    # Build model
    model = build_ct_model(
        num_stations=num_stations,
        features_per_station=2,
        model_dim=args.d_model,
        num_attention_heads=args.nhead,
        num_encoder_layers=args.num_layers,
        feedforward_dim=args.ffn_dim,
        dropout_rate=args.dropout,
        post_hidden_units=args.post_units,
        input_dropout_rate=args.input_dropout,
        enable_station_attention=args.use_station_attention,
        attention_hidden_units=args.attn_hidden,
        normalize_station_attention=args.normalize_station_attention,
        attention_temperature=args.attn_temperature,
        enable_hard_mask=args.hard_mask,
        enable_heteroscedastic_output=args.heteroscedastic,
        enable_causal_mask=args.use_causal_mask,
        enable_geo_features=args.use_geo,
        geo_feature_mode=args.geo_mode,
        geo_embedding_dim=args.geo_embed_dim,
        pgd_value_clip=args.pgd_clip,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # Geo坐标来源：优先使用NPZ中的样本级台站坐标；如提供CSV可作为回退静态嵌入
    if args.use_geo and args.coords_csv:
        try:
            coords = load_station_coords_csv(args.coords_csv, num_stations)
            model.set_station_coords(coords)
            print('已加载静态台站坐标CSV作为回退嵌入；训练/推理仍按样本传入NPZ坐标。')
        except Exception as e:
            print(f'警告：读取CSV台站坐标失败，将仅使用NPZ中的样本级坐标。错误: {e}')

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, threshold=args.min_delta) if args.use_scheduler else None

    # Time weights for early-time emphasis
    time_weights = torch.ones(T, dtype=torch.float32, device=device)
    early_T = int(T * args.early_frac)
    if early_T > 0:
        time_weights[:early_T] *= args.early_weight

    train_losses, val_losses = [], []
    best_val = float('inf')
    best_state = None
    patience = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        batch_log_train = []  # (loss, grad_norm, lr)
        grad_norm_sum = 0.0
        num_batches = 0
        for xb, yb, evb, stb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            evb = evb.to(device)
            stb = stb.to(device)
            # 训练增强：随机置零台站（PGD与状态）
            if args.rand_zero_stations:
                xb = random_zero_stations_in_batch(
                    xb,
                    num_stations=num_stations,
                    max_frac=min(args.zero_max_frac, 0.5),
                    alpha=args.zero_beta_a,
                    beta=args.zero_beta_b,
                )
            optimizer.zero_grad()

            if args.heteroscedastic:
                mu, log_var = model(xb, station_coords_deg=stb if model.use_geo else None)
                loss = gaussian_nll(mu, log_var, yb, args.var_floor, time_weights)
            else:
                yhat = model(xb, station_coords_deg=stb if model.use_geo else None)
                loss = F.mse_loss(yhat, yb, reduction='none')
                # Weighted average across time then mean over batch
                loss = torch.sum(loss * time_weights.unsqueeze(0)) / (loss.size(0) * torch.sum(time_weights))

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running += loss.item()
            lr_now = optimizer.param_groups[0]['lr']
            batch_log_train.append((loss.item(), float(grad_norm), float(lr_now)))
            grad_norm_sum += float(grad_norm)
            num_batches += 1
        train_losses.append(running / len(train_loader))
        avg_grad_norm = grad_norm_sum / max(num_batches, 1)

        model.eval()
        with torch.no_grad():
            val_running = 0.0
            batch_log_val = []  # (loss,)
            for xb, yb, evb, stb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                evb = evb.to(device)
                stb = stb.to(device)
                if args.heteroscedastic:
                    mu, log_var = model(xb, station_coords_deg=stb if model.use_geo else None)
                    vloss = gaussian_nll(mu, log_var, yb, args.var_floor, time_weights)
                else:
                    yhat = model(xb, station_coords_deg=stb if model.use_geo else None)
                    vloss = F.mse_loss(yhat, yb, reduction='none')
                    vloss = torch.sum(vloss * time_weights.unsqueeze(0)) / (vloss.size(0) * torch.sum(time_weights))
                val_running += vloss.item()
                batch_log_val.append((vloss.item(),))
            val_losses.append(val_running / len(val_loader))
            # 保存batch日志
            # try:
            #     path_csv = save_batch_log(args.out_dir, epoch, batch_log_train, batch_log_val)
            #     print(f"Saved batch log: {path_csv} | avg grad norm: {avg_grad_norm:.4f}")
            # except Exception as e:
            #     print(f"Failed to save batch log for epoch {epoch}: {e}")

        # Early stopping
        if (best_val - val_losses[-1]) > args.min_delta:
            best_val = val_losses[-1]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        print(f"Epoch {epoch:03d} | train {train_losses[-1]:.4f} | val {val_losses[-1]:.4f}")
        if scheduler is not None:
            try:
                scheduler.step(val_losses[-1])
                print(f"LR now: {optimizer.param_groups[0]['lr']:.6f}")
            except Exception:
                pass

        # 每10轮保存一次当前模型权重快照
        if (epoch % 10) == 0:
            try:
                snap_path = os.path.join(args.out_dir, f'ct_transformer_epoch_{epoch:03d}.pt')
                torch.save({
                    'state_dict': model.state_dict(),
                    'config': {
                        'num_stations': num_stations,
                        'feature_dim': 2,
                        'd_model': args.d_model,
                        'nhead': args.nhead,
                        'num_layers': args.num_layers,
                        'ffn_dim': args.ffn_dim,
                        'dropout': args.dropout,
                        'post_units': args.post_units,
                        'input_dropout': args.input_dropout,
                        'use_station_attention': args.use_station_attention,
                        'attn_hidden': args.attn_hidden,
                        'normalize_station_attention': args.normalize_station_attention,
                        'attn_temperature': float(args.attn_temperature),
                        'hard_mask': args.hard_mask,
                        'heteroscedastic': args.heteroscedastic,
                        'use_causal_mask': args.use_causal_mask,
                        'use_geo': args.use_geo,
                        'geo_mode': args.geo_mode,
                        'geo_embed_dim': args.geo_embed_dim,
                        'pgd_clip': None if args.pgd_clip is None else float(args.pgd_clip),
                        'rand_zero_stations': args.rand_zero_stations,
                        'zero_max_frac': float(args.zero_max_frac),
                        'zero_beta_a': float(args.zero_beta_a),
                        'zero_beta_b': float(args.zero_beta_b),
                    },
                    'epoch': epoch,
                    'train_loss': train_losses[-1],
                    'val_loss': val_losses[-1],
                    'y_mean': y_mean,
                    'y_std': y_std,
                    'time_len': T,
                    'var_floor': args.var_floor,
                    'seed': args.seed,
                    'train_frac': args.train_frac,
                    'val_frac': args.val_frac,
                }, snap_path)
                print(f"保存周期快照: {snap_path}")
            except Exception as e:
                print(f"保存周期快照失败（epoch {epoch}）: {e}")
        if patience >= args.early_patience:
            print("Early stopping triggered.")
            break

    # Save checkpoint
    ckpt_path = os.path.join(args.out_dir, 'best_ct_transformer.pt')
    torch.save({
        'state_dict': best_state if best_state is not None else model.state_dict(),
        'config': {
            'num_stations': num_stations,
            'feature_dim': 2,
            'd_model': args.d_model,
            'nhead': args.nhead,
            'num_layers': args.num_layers,
            'ffn_dim': args.ffn_dim,
            'dropout': args.dropout,
            'post_units': args.post_units,
            'input_dropout': args.input_dropout,
            'use_station_attention': args.use_station_attention,
            'attn_hidden': args.attn_hidden,
            'normalize_station_attention': args.normalize_station_attention,
            'attn_temperature': float(args.attn_temperature),
            'hard_mask': args.hard_mask,
            'heteroscedastic': args.heteroscedastic,
            'use_geo': args.use_geo,
            'geo_mode': args.geo_mode,
            'geo_embed_dim': args.geo_embed_dim,
            'pgd_clip': None if args.pgd_clip is None else float(args.pgd_clip),
            'rand_zero_stations': args.rand_zero_stations,
            'zero_max_frac': float(args.zero_max_frac),
            'zero_beta_a': float(args.zero_beta_a),
            'zero_beta_b': float(args.zero_beta_b),
        },
        'y_mean': y_mean,
        'y_std': y_std,
        'time_len': T,
        'var_floor': args.var_floor,
        'seed': args.seed,
        'train_frac': args.train_frac,
        'val_frac': args.val_frac,
    }, ckpt_path)

    # Plot loss curve
    plot_loss_curve(train_losses, val_losses, os.path.join(args.out_dir, 'loss_curve.png'))

    # Evaluate on test/val/train 并保存为 CSV
    model.load_state_dict(torch.load(ckpt_path)['state_dict'])
    model.to(device)
    model.eval()

    # ----- Test -----
    test_sta_nums = args.test_sta_num if args.test_sta_num is not None else [num_stations]
    metrics_by_sta_num = {}
    for _keep in test_sta_nums:
        preds = []
        vars_ = []
        trues = []
        used_stations_all = []
        with torch.no_grad():
            for xb, yb, evb, stb in test_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                evb = evb.to(device)
                stb = stb.to(device)
                if int(_keep) < num_stations:
                    xb = mask_keep_stations_in_batch(xb, num_stations=num_stations, keep=int(_keep))
                try:
                    B, T_b, FEAT_b = xb.shape
                    x_rs = xb.view(B, T_b, num_stations, 2)
                    for b in range(B):
                        ssum = x_rs[b].abs().sum(dim=0).sum(dim=1)  # [N]
                        used_idx = (ssum > 0).nonzero().view(-1).cpu().numpy().tolist()
                        used_stations_all.append(used_idx)
                except Exception:
                    pass
                if args.heteroscedastic:
                    mu, log_var = model(xb, station_coords_deg=stb if model.use_geo else None)
                    preds.append(mu.cpu().numpy())
                    sigma2 = torch.exp(log_var).clamp_min(args.var_floor)
                    vars_.append(sigma2.cpu().numpy())
                    trues.append(yb.cpu().numpy())
                else:
                    yhat = model(xb, station_coords_deg=stb if model.use_geo else None)
                    preds.append(yhat.cpu().numpy())
                    trues.append(yb.cpu().numpy())
        y_pred_norm = np.concatenate(preds, axis=0)
        y_true_norm = np.concatenate(trues, axis=0)
        y_pred = y_pred_norm * y_std + y_mean
        y_true = y_true_norm * y_std + y_mean
        if args.heteroscedastic and len(vars_) > 0:
            y_var_norm = np.concatenate(vars_, axis=0)
            y_sigma = np.sqrt(y_var_norm) * y_std
        else:
            y_sigma = None
        m = compute_sequence_metrics(y_true, y_pred)
        metrics_by_sta_num[str(int(_keep))] = m
        print('Test metrics:', {'sta': int(_keep), **m})
        csv_path = os.path.join(args.out_dir, f'test_sequences_sta{int(_keep)}.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            header = ['sample_index', 'event_name', 't', 'y_pred', 'y_true', 'abs_error']
            if y_sigma is not None:
                header.append('y_sigma')
            header.append('used_stations')
            writer.writerow(header)
            for s_idx in range(y_pred.shape[0]):
                ev_name = str(names_test[s_idx]) if s_idx < len(names_test) else ''
                for t in range(y_pred.shape[1]):
                    err = float(abs(y_pred[s_idx, t] - y_true[s_idx, t]))
                    row = [int(test_idx[s_idx]) if s_idx < len(test_idx) else int(s_idx), ev_name, t, float(y_pred[s_idx, t]), float(y_true[s_idx, t]), err]
                    if y_sigma is not None:
                        row.append(float(y_sigma[s_idx, t]))
                    stations_str = ''
                    if t == 0:
                        try:
                            stations = used_stations_all[s_idx] if s_idx < len(used_stations_all) else []
                            stations_str = ','.join(str(i) for i in stations)
                        except Exception:
                            stations_str = ''
                    row.append(stations_str)
                    writer.writerow(row)

    # ----- Validation -----
    preds_v, vars_v, trues_v = [], [], []
    with torch.no_grad():
        for xb, yb, evb, stb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            evb = evb.to(device)
            stb = stb.to(device)
            if args.heteroscedastic:
                mu, log_var = model(xb, station_coords_deg=stb if model.use_geo else None)
                preds_v.append(mu.cpu().numpy())
                sigma2 = torch.exp(log_var).clamp_min(args.var_floor)
                vars_v.append(sigma2.cpu().numpy())
                trues_v.append(yb.cpu().numpy())
            else:
                yhat = model(xb, station_coords_deg=stb if model.use_geo else None)
                preds_v.append(yhat.cpu().numpy())
                trues_v.append(yb.cpu().numpy())
    y_pred_v = np.concatenate(preds_v, axis=0) * y_std + y_mean
    y_true_v = np.concatenate(trues_v, axis=0) * y_std + y_mean
    if args.heteroscedastic and len(vars_v) > 0:
        y_sigma_v = np.sqrt(np.concatenate(vars_v, axis=0)) * y_std
    else:
        y_sigma_v = None

    csv_path_v = os.path.join(args.out_dir, 'val_sequences.csv')
    with open(csv_path_v, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ['sample_index', 'event_name', 't', 'y_pred', 'y_true', 'abs_error']
        if y_sigma_v is not None:
            header.append('y_sigma')
        writer.writerow(header)
        for s_idx in range(y_pred_v.shape[0]):
            ev_name = str(names_val[s_idx]) if s_idx < len(names_val) else ''
            for t in range(y_pred_v.shape[1]):
                err = float(abs(y_pred_v[s_idx, t] - y_true_v[s_idx, t]))
                row = [int(val_idx[s_idx]) if s_idx < len(val_idx) else int(s_idx), ev_name, t, float(y_pred_v[s_idx, t]), float(y_true_v[s_idx, t]), err]
                if y_sigma_v is not None:
                    row.append(float(y_sigma_v[s_idx, t]))
                writer.writerow(row)

    # ----- Train -----
    preds_tr, vars_tr, trues_tr = [], [], []
    with torch.no_grad():
        for xb, yb, evb, stb in train_eval_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            evb = evb.to(device)
            stb = stb.to(device)
            if args.heteroscedastic:
                mu, log_var = model(xb, station_coords_deg=stb if model.use_geo else None)
                preds_tr.append(mu.cpu().numpy())
                sigma2 = torch.exp(log_var).clamp_min(args.var_floor)
                vars_tr.append(sigma2.cpu().numpy())
                trues_tr.append(yb.cpu().numpy())
            else:
                yhat = model(xb, station_coords_deg=stb if model.use_geo else None)
                preds_tr.append(yhat.cpu().numpy())
                trues_tr.append(yb.cpu().numpy())
    y_pred_tr = np.concatenate(preds_tr, axis=0) * y_std + y_mean
    y_true_tr = np.concatenate(trues_tr, axis=0) * y_std + y_mean
    if args.heteroscedastic and len(vars_tr) > 0:
        y_sigma_tr = np.sqrt(np.concatenate(vars_tr, axis=0)) * y_std
    else:
        y_sigma_tr = None

    csv_path_tr = os.path.join(args.out_dir, 'train_sequences.csv')
    with open(csv_path_tr, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ['sample_index', 'event_name', 't', 'y_pred', 'y_true', 'abs_error']
        if y_sigma_tr is not None:
            header.append('y_sigma')
        writer.writerow(header)
        for s_idx in range(y_pred_tr.shape[0]):
            ev_name = str(names_train[s_idx]) if s_idx < len(names_train) else ''
            for t in range(y_pred_tr.shape[1]):
                err = float(abs(y_pred_tr[s_idx, t] - y_true_tr[s_idx, t]))
                row = [int(train_idx[s_idx]) if s_idx < len(train_idx) else int(s_idx), ev_name, t, float(y_pred_tr[s_idx, t]), float(y_true_tr[s_idx, t]), err]
                if y_sigma_tr is not None:
                    row.append(float(y_sigma_tr[s_idx, t]))
                writer.writerow(row)

    # Summary JSON
    with open(os.path.join(args.out_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'metrics': metrics_by_sta_num.get(str(num_stations), list(metrics_by_sta_num.values())[-1] if len(metrics_by_sta_num)>0 else {}),
            'metrics_by_sta_num': metrics_by_sta_num,
            'test_sta_nums': [int(n) for n in (test_sta_nums if isinstance(test_sta_nums, (list, tuple)) else [num_stations])],
            'num_test': int(X_test.shape[0]),
            'ckpt_path': ckpt_path,
            'stopped_epoch': int(epoch),
            'seed': args.seed,
            'train_frac': args.train_frac,
            'val_frac': args.val_frac,
            'optimizer': {
                'type': 'Adam',
                'lr': args.lr,
                'weight_decay': args.weight_decay,
            },
            'early_stopping': {
                'patience': args.early_patience,
                'min_delta': args.min_delta,
            },
            'scheduler': 'ReduceLROnPlateau' if args.use_scheduler else 'none',
            'loss': 'GaussianNLL' if args.heteroscedastic else 'WeightedMSE',
            'split_indices': {
                'train': [int(i) for i in train_idx],
                'val': [int(i) for i in val_idx],
                'test': [int(i) for i in test_idx],
            },
            'forced_test_hint': [int(i) for i in forced_test_hint],
            'augmentation': (
                {
                    'rand_zero_stations': True,
                    'zero_max_frac': float(args.zero_max_frac),
                    'zero_beta_a': float(args.zero_beta_a),
                    'zero_beta_b': float(args.zero_beta_b),
                }
                if args.rand_zero_stations else 'none'
            ),
        }, f, indent=2)

    print('Done. Saved to:', args.out_dir)


if __name__ == '__main__':
    main()
