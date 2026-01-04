# AMSU-A 压缩训练与评估

## 修改数据文件夹路径

本项目所有训练/评估脚本都通过 `--data_dir` 指向数据目录。将数据文件夹替换成你的实际路径即可：

- 训练脚本：`train_amsua_vaeformer_lite.py`
- 评估脚本：`eval_amsua_split.py`

示例（将 `/data/amsua_npz` 换成你的路径）：

```bash
python train_amsua_vaeformer_lite.py \
  --data_dir /data/amsua_npz \
  --pattern "*.npz"
```

如果你使用 `sbatch` 脚本（如 `pretrain.sbatch`、`finetune_entropy.sbatch`、`amsua_eval_test.sbatch`），直接把脚本中的 `--data_dir .` 改成你的路径即可。

## 运行模型测试（评估）

评估使用 `eval_amsua_split.py`，需要提供模型权重 `--ckpt`，并指定评估的 split：

```bash
python eval_amsua_split.py \
  --data_dir /data/amsua_npz \
  --split_json splits_amsua.json \
  --split test \
  --ckpt /path/to/checkpoint.pt \
  --stats_json /data/amsua_npz/stats_amsua.json
```

- `--split` 可选：`train` / `val` / `test`
- 如果训练时使用了 `--stats_json` 做归一化，评估时也应传入同一个 stats 文件。

输出会显示 `mse` 和 `bpp`，也可以用 `--out_json` 保存评估结果。
