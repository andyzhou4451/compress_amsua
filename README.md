# compress_amsua 项目说明

## 项目概览
该项目围绕 **AMSU-A 遥感观测数据** 的整理、建模与压缩评估，包含从 NetCDF 数据整理为神经网络输入、训练/评估 VAEformer 轻量模型、以及对重建结果做统计与可视化分析的一整套脚本工具。

核心目标：
- 将 AMSU-A 原始 NetCDF 数据整理为可训练的张量数据（含 mask/统计量）。
- 训练 **VAEformerLite** 变体进行压缩与重建。
- 执行评估与可视化，输出误差统计与 NetCDF 结果。

## 目录结构（根目录脚本一览）

| 文件 | 作用 | 关键输出/要点 |
| --- | --- | --- |
| `read_amsua.py` | 单文件数据整理示例 | 生成 `network_ready_feature.nc/.npz` 与 `*_meta.json`，构建 (time, feature, lat, lon) 张量。 |
| `batch_nc_to_npz.py` | **批量** NetCDF → NPZ | 输出 `X/M/feature_names/lat/lon/time`，可选标准化统计。 |
| `run_batch_nc_to_npz.slurm` | Slurm 批处理封装 | 批量执行 `batch_nc_to_npz.py`。 |
| `inspect_npz.py` | NPZ 结构与统计检查 | 输出结构、dtype、分位数、缺测情况等。 |
| `train_amsua_vaeformer_lite.py` | 训练 VAEformerLite | 支持 DDP/SLURM，生成模型与训练日志。 |
| `pretrain.sbatch` | 预训练作业脚本 | Slurm 训练入口。 |
| `finetune_entropy.sbatch` | 熵模型微调脚本 | Slurm 训练入口。 |
| `eval_amsua_split.py` | 按 split 评估模型 | 固定随机 seed 采样 patch，计算重建误差与 bpp。 |
| `amsua_eval_test.sbatch` | 评估作业脚本 | Slurm 评估入口。 |
| `amsua_codec_tiles.py` | **按 tile 编码/解码** | 对指定 tile 编码估计 bpp 与重建。 |
| `amsua_codec_testset.sbatch` | 编解码测试作业 | Slurm 入口脚本。 |
| `analyze_tmbrs_and_convert_nc.py` | 分析 tmbrs 通道并转成 NC | 输出统计与可视化，支持 recon 对齐。 |
| `analyze_groups_plot_blank_and_to_nc.py` | 组级别分析与绘图 | 输出 recon vs orig 的统计、误差图、NC 结果。 |

> 注：多数脚本包含参数解析与注释，适合作为可复用工具或流水线组件。

## 数据处理流程（推荐）

1. **准备 NetCDF 数据**（示例：`1bamua_YYYYMMDD_tHH.nc`）。
2. **转换为 NPZ**（批量或单文件）。
3. **训练 VAEformerLite**。
4. **评估与可视化**：生成误差统计、重建对比或导出 NetCDF。

```
NetCDF (.nc)
   │
   ├─ read_amsua.py / batch_nc_to_npz.py
   ▼
NPZ (X/M/feature_names/...)
   │
   ├─ train_amsua_vaeformer_lite.py
   ├─ eval_amsua_split.py
   └─ amsua_codec_tiles.py
```

## 关键脚本说明与示例

### 1) 数据整理（单文件）
```bash
python read_amsua.py
```
- 读取 `NC_PATH` 指定的 NetCDF。
- 生成 `(time, feature, lat, lon)` 的张量与 mask。
- 输出标准化统计（均值/方差）。

### 2) 批量转换 NetCDF → NPZ
```bash
python batch_nc_to_npz.py --input_dir <nc_dir> --output_dir <npz_dir>
```
- 每个 `.nc` 生成一个 `.npz`。
- 默认保存 `X`（特征张量）与 `M`（有效点 mask）。

### 3) 训练模型
```bash
python train_amsua_vaeformer_lite.py --data_dir <npz_dir>
```
- 支持 DDP 与 SLURM 环境。
- 输入为 `npz`，包含 `X/M/feature_names`。

### 4) 评估与压缩实验
```bash
python eval_amsua_split.py --data_dir <npz_dir> --ckpt <model.pt>
python amsua_codec_tiles.py --input_npz <file.npz> --ckpt <model.pt>
```
- `eval_amsua_split.py` 按 split/patch 抽样评估。
- `amsua_codec_tiles.py` 支持 tile 编码、解码与 bpp 估算。

## 依赖环境
- Python 3.8+
- 核心依赖：`numpy`, `torch`, `xarray`, `matplotlib`
- 若使用 NetCDF：建议安装 `netCDF4` 或 `h5netcdf`

## 产出文件说明

- **NPZ**
  - `X`: float32，(T, F, H, W)
  - `M`: uint8，(T, F, H, W) 有效点 mask
  - `feature_names`: 特征名称（如 `tmbrs-1`、`channels-1` 等）
  - `lat/lon/time`: 坐标
  - `mean/std`（可选）

- **模型输出**
  - 训练产生的模型权重 (`.pt`) 与日志
  - 评估输出的误差统计、图像、NetCDF 文件等

## 备注
- 若仅需要确认 NPZ 内容是否正确，可先运行 `inspect_npz.py` 检查维度与统计分布。
- Slurm 环境下推荐使用 `.sbatch` / `.slurm` 文件批量运行。
