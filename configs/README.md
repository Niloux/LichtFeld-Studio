# 训练配置

[篮球场.json](篮球场.json) 是完整的训练参数快照，按用途用空行分组。
常用项优先看下面的表，其余参数保留在相应功能组里。

先填写实际存在的 `dataset.data_path` 和 `init_path`，再从仓库根目录运行：

```sh
./build/train/lfs-train --config configs/篮球场.json
```

## 常用修改

| 要调整的内容 | 配置项 | 当前设置 |
| --- | --- | --- |
| 数据集和图片目录 | `dataset.data_path`、`dataset.images` | COLMAP 数据集；图片目录为 `images` |
| 初始化点云 | 顶层 `init_path` | 初始化 PLY 路径 |
| 输出位置和文件名 | `dataset.output_folder`、`dataset.output_name` | `./output/1010/篮球场`；名称留空 |
| 图片分辨率 | `dataset.resize_factor`、`dataset.max_width` | 缩小 2 倍；不额外限制宽度 |
| 训练规模 | `optimization.iterations`、`optimization.max_cap` | 30000 步；最多 300 万个高斯 |
| 策略和相机渲染 | `optimization.strategy`、`optimization.gut` | MRNF；GUT 开启 |
| 颜色表达 | `optimization.sh_degree` | SH0 |
| 天空背景 | `optimization.sky_enabled`、`optimization.sky_num_points` | 关闭；启用时使用 10 万个天空高斯 |
| 外观校正 | `optimization.use_exposure_correction`、`optimization.use_bilateral_grid`、`optimization.use_ppisp` | 曝光校正开启；双边网格和 PPISP 关闭 |
| 正则化与衰减 | `optimization.opacity_reg`、`optimization.opacity_decay`、`optimization.scale_reg`、`optimization.scale_decay` | 0.005、0.005、0.005、0.002 |
| 留出验证图片 | `dataset.use_test_split`、`dataset.test_every` | 不留出；每个相机每 8 张采样评估 |
| 评估指标 | `optimization.enable_eval` | 开启 PSNR、SSIM；权重可用时自动计算 LPIPS |
| 评估和保存时间 | `optimization.eval_steps`、`optimization.save_steps` | 在数组指定的迭代执行 |
| 深度和法线监督 | `optimization.use_depth_loss`、`optimization.use_normal_loss` | 均关闭 |

## 文件中的分组顺序

- `dataset`：路径与名称、图片尺寸与坐标居中、验证采样、掩码、加载缓存、延时摄影。
- `optimization` 开头：策略、步数、点数上限、步数缩放、初始化。
- 渲染与外观：相机处理、SH、背景、天空、曝光校正、双边网格、PPISP。
- 损失：图像与结构、掩码与裁剪框、深度、法线。
- 优化：正则化与衰减、学习率、增密与剪枝、稀疏化。
- 评估与保存：评估开关、空间与精度、评估掩码、保存时间。
- 顶层 `export_formats`：最终导出格式，训练版使用 `ply`。

## 几个容易混淆的设置

- JSON 保持扁平字段和空行分组。JSON 不支持注释，`optimization` 内部的功能组也不能改成嵌套对象。
- `output_name` 留空时，项目使用 `project.licht`，导出使用 `splat_ITER.ply`。填写名称后，项目和导出使用该名称。
- `use_test_split=false` 会用全部图片训练，采样评估的是训练视图；设为 `true` 才会留出采样图片。`eval_all=true` 则评估全部可用视图。
- 只执行不超过 `iterations` 的评估、保存时间点。当前数组保留了 45000、60000 等后续时间点，训练到 30000 步时这些时间点不执行。
- `steps_scaler` 调整步数；`image_count_scaler` 和 `image_count_scaler_total` 记录缩放来源，通常无需手改。
- 各功能先由 `use_*`、`enable_*` 或 `sky_enabled` 控制，其后的参数供相应功能使用。
- 训练版使用预生成的深度、法线图，两个 `*_auto_generate` 都设为 `false`。启用监督后需要相应的图文件。
- LPIPS 没有独立的 JSON 开关。将原生 `.lfw` 权重放到 `~/.lichtfeld/onnx/lpips-vgg16-v0.1.lfw`，或通过 `LFS_LPIPS_WEIGHTS` 指定路径；下载与校验步骤见 [LPIPS 评估](../docs/train-only.md#lpips-evaluation)。
- 命令行可以覆盖 JSON 的对应选项；完整配置用于恢复训练时，也会覆盖其中明确写出的优化选项。仅想延长原训练时，可使用 `--resume` 和 `--iter`。

更多训练、恢复和构建说明见 [训练版文档](../docs/train-only.md)。
