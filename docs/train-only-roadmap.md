# 独立训练版：阶段记录与后续计划

更新日期：2026-09-06。

目标：在 LFS fork 内维护独立的 `lfs-train`，复用训练内核与数据流，
让桌面编辑器从依赖解析、配置、编译和链接阶段退出。构建和安装方法见
[独立训练版使用说明](train-only.md)。

## 1. 当前阶段：基础拆分已完成

- `LFS_BUILD_PROFILE=train` 构建独立 CLI，默认 `studio` profile 保留。
- 保留 FastGS、gsplat/GUT、图像缓存与 GPU 解码、训练策略和已有损失。
- 支持 COLMAP、普通点云或 Gaussian PLY 初始化、PLY 导出和项目恢复。
- 支持读取已有 masks、depth、normal；depth/normal 监督默认关闭。
- 排除桌面查看器、视频、MCP/TCP、插件、完整 Python 运行时和自动先验生成。
- 修复 Q16 静态缓存退出时通过失效 CUDA 流释放显存的问题。
- 保留 OpenMesh 数据层和通用 Tensor/神经算子，暂不继续拆分这些依赖。

已提交：

| 提交 | 内容 |
| --- | --- |
| `bef1e784` | 独立训练版、Q16 清理修复、测试、CI 定义和说明文档 |
| `3bb7cf7c` | 忽略本地 CodeGraph 索引和仓库根目录的 `.deb` 安装包 |

已完成的验证：

- 本机 Release 构建和安装通过；编译单元、vcpkg 包及运行依赖审计通过。
- 3 项 CTest 和 9 项生成数据 GPU 回归通过，包括缺失监督数据报错。
- contextcapture 使用 COLMAP 初始点、RGB/GUT 完成 20 步，安装后的程序恢复到 24 步并导出。
- Q16 修复后完成 1000 步 RGB/GUT 训练，高斯数从 25,094 增长到 35,186，保存、导出和退出无 CUDA 错误。

这些验证不代表完整画质或性能验收。完整 studio、Debug 构建及新 CI 工作流尚未验证。

## 天空辅助建模

已实现独立固定几何天空高斯：+Z 上半球、固定位置/尺度/旋转/opacity，
仅训练 SH0 颜色；原始 sky mask 约束前景 alpha，不额外腐蚀或对重建损失边界降权。
天空几何、颜色与 Adam 状态进入 checkpoint，PLY 仅导出前景。
旧 cubemap checkpoint 不直接迁移，需要新开训练。详见 [天空训练](train-sky.md)。

本机验证：GPU 数值梯度与 Adam 恢复检查通过；两种渲染器各完成
20→24 步保存恢复；held-out 评估、9 项集成用例和原有 3 项 CTest 通过。
contextcapture 的 252 张鱼眼图像及 sky mask 校验通过，并以宽度 512、
1,488,579 点 LiDAR 初始化和 100,000 天空高斯完成 5 步功能验证；
导出仍为 1,488,579 个前景点。用户试用反馈：相较 cubemap，漂浮点更少、结果更干净；
这是实际使用反馈，尚无同条件定量画质对比。

## 2. 当前推荐训练配置

数据：`/home/wuyou/3dgeer/data/contextcapture`，252 张鱼眼图像。
`lidar.ply` 包含 1,488,579 个 XYZ+RGB 点；该点云与相机的坐标和尺度对齐仍需验证。

| 项目 | 配置 |
| --- | --- |
| 初始化 | `--init .../lidar.ply`，替换 COLMAP 初始点云 |
| 鱼眼相机 | `--gut` |
| 分辨率 | `-r 4 --max-width 0`，宽高各缩小到 1/4，不附加宽度上限 |
| SH 阶数 | `--sh-degree 0` |
| 曝光矫正 | `--exposure-correction` |
| 遮罩 | `--mask-mode ignore`，白色参与损失，黑色忽略 |
| 天空 | `--sky --sky-mask-dir sky_masks --sky-num-points 100000` |
| depth / normal | 保持默认关闭 |
| 训练目标 | 30,000 步，导出 PLY |

从仓库根目录启动新训练：

```bash
./build/train/lfs-train \
  -d /home/wuyou/3dgeer/data/contextcapture \
  --init /home/wuyou/3dgeer/data/contextcapture/lidar.ply \
  -o ./output/contextcapture_lidar_gaussian_sky \
  --gut \
  -r 4 \
  --max-width 0 \
  --sh-degree 0 \
  --exposure-correction \
  --mask-mode ignore \
  --sky --sky-mask-dir sky_masks --sky-num-points 100000 \
  --iter 30000 \
  --export ply
```

注意以下参数语义：

- `masks/L/`、`masks/R/` 自动按图像匹配；黑白含义尚未核实。
  如果白色表示排除区域，应加 `--invert-masks`。
- `ignore` 负责有效区域的损失遮罩；独立 sky mask 通过 alpha 惩罚约束天空区前景趋于透明。
  当前天空组件支持 `none`/`ignore`，不与 `segment` 组合。
- `--max-cap` 小于初始点数时会随机抽样初始点云；保留全部 LiDAR 点需要至少约 149 万的容量上限。
- SH 0 只使用与观察方向无关的颜色；全分辨率需显式设 `-r 1 --max-width 0`。
- 使用新的输出目录，避免覆盖此前训练。当前已完成宽度 512 的联合短跑；`-r 4 --max-width 0` 长训尚未做定量验收。

恢复同一训练时使用项目状态：

```bash
./build/train/lfs-train \
  --resume ./output/contextcapture_lidar_gaussian_sky/project.licht \
  --iter 30000 \
  --export ply
```

`--iter` 是目标总步数；恢复时不需要重新传入 LiDAR 初始化参数。

## 3. 后续验证

功能链路已通过短跑，后续重点验证配置恢复和重建质量：

- [ ] 检查若干图像与遮罩，确认黑白含义、左右相机对应和缩放后的对齐。
- [ ] 检查 LiDAR 点云与相机坐标、尺度及投影是否一致。
- [x] 宽度 512 联合短跑确认初始化点数、天空与有效区域遮罩加载和曝光矫正。
- [x] 生成数据在 GUT/FastGS 下完成独立进程恢复，确认天空配置和优化状态恢复正确。
- [ ] 补充实际 LiDAR 长训 checkpoint 的恢复验证。
- [ ] 完成长训练，记录 loss、模型规模、显存占用和导出结果，检查重建质量。
- [ ] 在相同数据、初始化、分辨率、SH 和训练配置下，对比曝光矫正开关的效果。

比较结果时记录提交号、完整命令和输出目录；吞吐统计排除编译与预热，
不要用不同高斯数量或分辨率的运行直接比较性能。耗时 GPU 评测开始前先确认范围。

## 4. 后续阶段：自定义 loss

具体新增哪种 loss 尚未确定，接口设计前先明确监督来源和优化目标。

- 优先提供小型 C++/CUDA 损失接口，同时接入前向损失和正确的反向梯度。
- 验证权重为零时回到基线；权重非零时确实影响预期参数的梯度。
- 用小规模数值检查和训练对照验证梯度及损失尺度。
- 如确有 Python 实验需求，再设计独立、轻量的 Python loss bridge，避免重新引入编辑器绑定。

目前没有实现独立的自定义 loss 插件接口。
`LFS_TRAIN_PYTHON_LOSS=ON` 会明确报未实现，不能仅打开开关获得该能力。

## 5. 后续维护：按需求补齐验证与精简

- 补跑 Debug、CI 和完整 studio 的兼容性检查。
- 做与原版同配置的训练质量和性能对照。
- 根据实际维护成本再决定是否拆除 OpenMesh 等剩余数据层依赖。
- 当前鱼眼数据依赖 GUT，继续保留对应训练后端，不进行 FastGS-only 裁剪。
- 自动 depth/normal 生成继续作为外部预处理；需要时读取预生成结果。
  `LFS_TRAIN_PREPROCESS=ON` 当前同样未实现。

下一次工作从第 3 节开始；本文件记录计划，不表示未勾选项目已经执行。
