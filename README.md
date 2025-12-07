# CTSequence Model 说明文档

## 项目概览
- CTSequence Model：面向时间序列的震级 `Mw(t)` 预测模型。
- 采用 Transformer 编码器，结合台站状态门控与台站注意力权重，支持异方差输出（均值+方差）。
- 输入为 GNSS PGD 与台站状态，可选使用台站地理特征（坐标编码+台站间距离统计）。

## 目录结构
- `dataset/`：训练/评估所需的 NPZ 数据集（示例：`all_events_pgd_dataset_sta_16_with_labels.npz`）。
- `mag_real_stf/catalog_all/`：地震事件的时序文件（`.mr`）。
- `model_new/ct_model.py`：模型实现与构建函数。
- `model_new/train_ct.py`：训练脚本与 CLI。
- `model_new/stream_infer.py`：流式推理脚本与 CLI。

## 数据格式
- 样本张量 `X` 形状：`[E, T, N*2]`，其中 `N` 为台站数。
- 最后维度为两个通道：`PGD` 与 `status`（在线=1，离线=0）。
- 标签 `y` 形状：`[E, T]`，表示每秒的 `Mw(t)`。
- 可选台站坐标 `station_coords_deg`：形状 `[B, N, 2]`，分别为 `(lat_deg, lon_deg)`。

## 模型结构与关键特性
- 编码器：`TransformerEncoder`（Pre-LN，`GELU` 激活）。
- 台站门控：根据 `PGD` 与 `status` 生成软/硬门控因子，抑制离线台站与异常值。
- 台站注意力：按台站维度聚合权重，支持温度缩放与归一化（Masked Softmax）。
- 地理特征（可选）：台站坐标编码（`xyz/raw/sin_cos`）+ 台站间距离统计特征（最小/最大/均值/标准差）。
- 异方差输出：同时预测均值与对数方差，损失为高斯 NLL。

## 标准化参数名（新）
- 模型结构
  - `features_per_station`（每台站特征数，原 `feature_dim`）
  - `model_dim`（模型嵌入维度，原 `d_model`）
  - `num_attention_heads`（注意力头数，原 `nhead`）
  - `num_encoder_layers`（编码层数，原 `num_layers`）
  - `feedforward_dim`（前馈层维度，原 `ffn_dim`）
- 正则与投影
  - `dropout_rate`（整体 dropout，原 `dropout`）
  - `post_hidden_units`（后置 MLP 隐藏单元，原 `post_units`）
  - `input_dropout_rate`（输入层 dropout，原 `input_dropout`）
- 注意力与门控
  - `enable_station_attention`（启用台站注意力，原 `use_station_attention`）
  - `attention_hidden_units`（注意力/门控 MLP 隐层，原 `attn_hidden`）
  - `normalize_station_attention`（注意力归一化，保持不变）
  - `attention_temperature`（注意力温度，原 `attn_temperature`）
  - `enable_hard_mask`（硬门控，原 `hard_mask`）
- 输出与掩码
  - `enable_heteroscedastic_output`（启用异方差输出，原 `heteroscedastic`）
  - `enable_causal_mask`（启用因果掩码，原 `use_causal_mask`）
- 地理特征
  - `enable_geo_features`（启用地理特征，原 `use_geo`）
  - `geo_feature_mode`（坐标编码模式，原 `geo_mode`）
  - `geo_embedding_dim`（地理嵌入维度，原 `geo_embed_dim`）
- 其他
  - `pgd_value_clip`（PGD对称裁剪阈值，原 `pgd_clip`）

## 旧参数兼容
- `build_ct_model(...)` 自动兼容旧参数名并映射到新参数名。
- 旧脚本（如 `stream_infer.py`）仍可继续使用旧命名，不需要改动。
- 参考实现：`model_new/ct_model.py:316` 的 `alias` 映射与统一传参。

## 关键代码位置
- 模型类定义：`model_new/ct_model.py:25`（`CTSequenceModel`）。
- 构建函数：`model_new/ct_model.py:316`（`build_ct_model`）。
- 训练脚本构建调用：`model_new/train_ct.py:428`。

## 训练使用
- 基本命令：
```bash
python model_new/train_ct.py \
  --pgd_npz f:\magnitude_prediction\dataset\all_events_pgd_dataset_sta_16_with_labels.npz \
  --time_len 200 \
  --d_model 128 --nhead 4 --num_layers 2 --ffn_dim 256 \
  --post_units 128 --dropout 0.2 --input_dropout 0.0 \
  --use_station_attention --attn_hidden 16 \
  --normalize_station_attention --attn_temperature 1.0 \
  --hard_mask --heteroscedastic --use_causal_mask \
  --use_geo --geo_mode xyz --geo_embed_dim 16 \
  --batch_size 32 --epochs 50 --lr 1e-3 --weight_decay 0.0 \
  --early_weight 3.0 --early_frac 0.5 \
  --rand_zero_stations --zero_max_frac 0.5 --zero_beta_a 2.0 --zero_beta_b 8.0 \
  --out_dir f:\magnitude_prediction\model_new\runs
```


## 流式推理
- 基本命令：
```bash
python model_new/stream_infer.py \
  --ckpt f:\magnitude_prediction\model_new\runs\best_ct_transformer.pt \
  --pgd_npz f:\magnitude_prediction\dataset\all_events_pgd_dataset_sta_16_with_labels.npz \
  --sample_index 0 --max_window 200 --causal \
  --normalize_station_attention --attn_temperature 1.0 --pgd_clip 2.0
```
- 输出：若启用异方差，打印 `Mw ± 2σ`；否则打印 `Mw`。

## 地理特征使用
- 启用：训练时添加 `--use_geo` 与 `--geo_mode`（`xyz/raw/sin_cos`）。
- 输入：在 `forward(inputs, station_coords_deg=...)` 传入 `[B, N, 2]` 形状的台站坐标（度）。
- 备用：若不传入批次坐标，可通过 `CTSequenceModel.set_station_coords(...)` 预注册静态台站嵌入。

## 大文件管理（Git LFS）
- GitHub 对单文件大小限制为 100MB；建议使用 Git LFS 管理大数据文件（如 `*.npz`）。
- 推荐步骤：
```bash
git lfs install
git lfs track "*.npz"
git add .gitattributes
git commit -m "Track *.npz with Git LFS"
# 如需将历史中的 *.npz 统一迁移到 LFS：
git lfs migrate import --include="*.npz" --everything
git push -u origin master --force-with-lease
```

## 注意事项
- 路径含空格时需加引号或使用 `--` 分隔（例如：`"CTSequence model"`）。
- Git 不跟踪空目录，如需保留结构可添加占位文件 `.gitkeep`。
- 输入形状与台站数需一致：`[B, T, N*2]` 与坐标 `[B, N, 2]` 中的 `N` 必须匹配。

## 版本兼容与迁移建议
- 新代码统一使用标准化参数名；旧脚本/检查点仍可正常加载。
- 建议逐步将外部脚本的构造函数调用迁移到新参数名，提升可读性与一致性。

