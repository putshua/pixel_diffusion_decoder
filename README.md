# Pixel Diffusion Video Decoder

用 H3 video latent 作为条件，采用 PiD / PixelDiT 空间主干、轻量时序模块和 **直接 x0 prediction**，训练同分辨率视频解码器。

当前实验：**512×320、56 帧、24 FPS，4×L20Z，全局 BS=64，每卡 2、GA=8**。从官方空间预训练权重开始；这是视频适配实验，不是官方 PiD 的原样复现。**1-step 蒸馏尚未实现**，目前 callback 使用 8 步 Euler。

## 方案

- 官方空间主干：PixelDiT_finetune_2kto4k EMA，约 **1.366B 参数**，约 1.344B 可训练。
- 输入：干净 H3 posterior-mean latent，经 post_quant_conv，24 通道，不添加条件 latent 噪声。
- 逐帧空间注意力：14 个 patch blocks + 2 个 pixel blocks；每隔一个 patch block 加入时序残差，瓶颈 128、Conv1d kernel=3、输出零初始化。
- H3 的 17 个 latent 时间位置按原始编码块规则映射到 56 个真实帧，不做简单线性插值。
- 新建零初始化 RGB readout，网络直接预测干净像素 x0；空间主干、latent adapter、时序模块与 readout 一起训练。H3 encoder 冻结。
- 不使用字幕编码器；一个可学习 null token。像素采用 ImageNet mean/std 归一化。

加噪与训练目标（本代码 t=0 是纯噪声，t=1 是干净视频）：

```text
x_t = t*x0 + (1-t)*noise
x0_pred = model(latent, x_t, t)
loss = mean(((x0_pred - x0) / max(1-t, 0.05))²)
```

这是 **x0 输出参数化 + RF 时间加权 loss**。日志 `x_mse` 是未加时间权重的像素 MSE。官方 PiD 主方案输出 velocity；这里保留空间预训练权重，重新初始化 x0 输出头。

| 训练项 | 当前设置 |
|---|---|
| 主干 LR | 2e-6 |
| latent adapter / 时序模块 / null token LR | 2e-5 |
| x0 输出头 LR | 1e-4 |
| AdamW | betas=(0.9,0.95)，weight_decay=0.01 |
| warmup / clip | 200 updates / grad norm 1.0 |
| t 采样 | sigmoid(N(-0.8, 0.8²)) |
| batch | 4 × 2 × GA 8 = 64 条完整视频 |
| 最大步数 | 30,000 optimizer updates |
| callback / checkpoint | 每 200 / 2000 optimizer updates |
| 分布式 | root FSDP FULL_SHARD，block activation checkpoint，FP32 参数/梯度，BF16 autocast |

梯度按累积次数取平均，只在完整 global batch 后 clip 和更新。FSDP 每个 micro-step 同步并保持梯度分片；没有使用会保留完整梯度的 `no_sync`。原生 VAE 逐条视频编码，控制显存。

## 目录

```text
pid_video/       模型、目标、训练、数据抽取、checkpoint、显存/LR 对照工具
scripts/         本机/PAI worker 启动脚本
configs/h3_vae/  H3 组件和架构配置（不含权重）
tests/           CPU 合约测试和可选四卡 tiny FSDP 恢复测试
third_party/    固化 PiD、H3 VAE 源码，保留来源及许可证
vendor_archives/ diffusers 固化源码压缩包与逐文件 SHA；bootstrap 后展开到 third_party
docs/           设计、实验快照、数据与运行说明
examples/       manifest / 固定验证样本格式
```

只提交源码、模板和汇总指标。不包含视频、训练数据清单、权重、checkpoint 或凭据。独立目录不依赖原 h3_refiner 业务代码。请使用 editable install，并保留仓库内 third_party；尚未提供可脱离仓库目录的 wheel。

## 环境

已用 Python 3.10.20、PyTorch 2.8.0+cu128 的训练环境验证原实验。CPU 测试不能替代完整预训练模型的 GPU 验证。

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python scripts/bootstrap_vendor.py
pip install -r requirements.txt
pip install --no-deps ./third_party/diffusers-sol
pip install -e . --no-deps
```

需要系统 FFmpeg（含 libx264）。训练和 VAE 诊断入口显式优先加载 `third_party/diffusers-sol/src` 的 **0.40.0.dev0** 固化源码；环境中 pip 显示的 diffusers 版本不是这里实际采用的版本。

## 准备权重

PiD 权重下载固定 Hugging Face revision `1b9b0872cef786e527ecc5f12b5ab2e6a02d55a4`，转换为 DCP，并记录文件 SHA256：

```bash
python -m pid_video.fetch_prior --destination assets/pid_official
```

H3 权重须由使用者合法取得。将对应 FL2VA video VAE 的 `model.safetensors` 放到 `configs/h3_vae/source/model.safetensors`（git 已忽略）；架构配置已包含。其他位置可复制组件配置，并修改其 `source_path`，然后通过 `--vae` 指定组件目录。缺少/不兼容权重会报错，不随机回退。

## 准备数据与 run

`sources.jsonl` 每行一个 `source_id` 和原视频 `path`，格式见 `examples/`。source_id 唯一，条数须能被 world size 整除。显式选一个固定验证视频，并从训练清单中排除；没有划分大规模独立测试集。

每个源视频采用三个时间区间分层采样，24 FPS 抽取 56 帧，保持比例 resize + 中心 crop 到 512×320。尺寸/时长不足或读取失败会记录到 `rejections_rank*.jsonl` 并换下一个候选；连续 64 个不可用来源则明确退出。

```bash
python -m pid_video.prepare \
  --run runs/experiment_001 \
  --sources /data/sources.jsonl \
  --validation-json /data/validation.json \
  --vae configs/h3_vae \
  --prior assets/pid_official/checkpoints/PixelDiT_finetune_2kto4k/model \
  --ffmpeg ffmpeg --world-size 4 --global-batch 64 --micro-batch 2
export PID_RUN="$PWD/runs/experiment_001"
bash scripts/run_pai.sh train
```

准备器把输入路径解析为绝对路径，冻结 manifest SHA 和源码 SHA。不要修改已开始训练的配置/源码；重新启动时保持相同挂载路径。首次启动自动做两步完整训练、保存 DCP，再由新进程恢复并继续。平台重启从最新完整 checkpoint 续训，保留 optimizer、step、每 rank 的数据游标和 CPU/CUDA RNG。

SIGTERM/SIGINT 在 optimizer update 边界请求保存并非零退出。直接向**训练 rank 进程**发送信号可触发此机制；launcher 或平台硬终止不能保证保存完成。只认可具有 `complete.json` 的 checkpoint，`latest.json` 在全部 rank 保存成功后更新。视频 callback 不等于权重保存。

## Callback

`$PID_RUN/gallery.html` 和 `callbacks/<step>/comparison.mp4`，从左到右：

1. GT。
2. H3 直接 decoder 输出（历史有网格的基线）。
3. 相同 latent 的 H3 官方空间/时间切块融合解码。
4. PiD x0 的 8 步 Euler 解码。

`metrics.json` 包含固定 t=0.2/0.5/0.8 的去噪 RF loss、8 步完整解码的 `sample_mse`。训练 `x_mse` 与完整采样 `sample_mse` 不可混作同一指标。当前去噪 loss 大幅降低，但 8 步完整解码 MSE 仍有明显波动；不能据此宣称已达到可用解码质量。详见 [实验记录](docs/experiments.md)。

## PAI

单个 worker，4×80GB GPU，共享挂载代码、源视频、权重和 run 目录。启动命令同上，`PID_PYTHON` 可指定解释器。本仓库不自动调用 DLC API 提交任务。

该项目的 DLC 设置：闲时资源、priority=8、AIMaster 自动重启，显式设置 `--enable-job-restart=True --max-num-of-job-restart=100`；workspace、资源 ID、镜像须在提交时验证，登记对应台账。业务失败必须返回非零，不能用保卡循环掩盖失败。

如部署环境要求任务完成后保卡，通过 `PID_HOLDER_GUARD=/absolute/path/guard.py` 指定已有 guard；成功后执行 `--ensure`。此独立导出不硬编码原机器路径；未配置 guard 时成功完成即退出。不要停止其他任务。

## 验证

```bash
OMP_NUM_THREADS=2 python -m pytest -q
# 以下命令需要四张空闲 GPU；仅测试基础设施，不代表预训练质量：
torchrun --standalone --nproc-per-node=4 tests/pid_fsdp_worker.py runs/tiny_fsdp_check
```

导出包的 CPU 测试、源码独立性及扫描结果见 `docs/export_validation.json`。本机原实验的完整四卡验证与独立导出的验证分别记录，导出过程中没有重启或占用当前训练 GPU。

## 来源与后续

- [官方 PiD](https://github.com/nv-tlabs/PiD)，固定 commit `2c8814c2b91cc41a2be7809962c891e0d0ccff5f`。
- H3 VAE、diffusers 的来源与 SHA 清单见各 third_party 子目录。
- 保留第三方原许可证；模型权重及数据的使用条件独立于代码。此私有项目未额外授予本项目新增代码的开源许可。
- 后续待验证：高噪声条件学习、采样步数消融、latent 条件依赖、多个固定验证源、视频时序质量，之后再考虑 1-step 蒸馏。当前没有实现蒸馏训练入口。
