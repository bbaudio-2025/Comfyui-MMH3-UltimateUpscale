# Comfyui-MMH3-UltimateUpscale

**在显存有限的条件下，通过单节点实现长时间、高分辨率 MiniMax H3 视频放大。**

本节点在**显存紧张**的约束下，对已经去噪完成的 MiniMax H3 AV 潜变量（视频+音频）进行重新采样（增强 / 放大）：采用 **时间分块（支持任意长视频） + 空间分块（支持任意高分辨率）** 的处理方式，**峰值显存仅占用一个 tile**，同时完整保留音频轨道。

MiniMax H3 生成的视频是一个"嵌套潜变量"——把 24 通道的视频和 32 通道的音频打包在同一个张量里。标准 ComfyUI 放大节点无法理解这种结构。`MMH3 Ultimate Upscale` 把整条 `时间分块 → latent 放大 → 空间分块 → 逐块采样 → 空间拼接 → 时间拼接` 流程封装进一个节点，让你能像放大普通 latent 一样放大一段已经生成的 H3 视频——既不破坏音频，也**不会在小显存显卡上爆显存**。

---

## 更新信息

- **20260905 - latent 放大模型现在只支持 `.safetensors`。** 不再列出、也不再加载 `.pth` / `.pt` / `.ckpt` 等 pickle 格式：`torch.load()` 加载 pickle 权重时可以执行任意代码，而在 torch < 2.6 上 `weights_only=True` 也并非可靠防护（[CVE-2025-32434](https://nvd.nist.gov/vuln/detail/CVE-2025-32434)）。如果你之前在用 `minimax_h3_latent_upscaler_3d_fp32.pth`，请按下方[放大模型格式](#放大模型格式仅支持-safetensors)转换一次，或改用 fp16 / bf16 的 `.safetensors` 版本。扫描逻辑同时改为遍历所有已注册的 `latent_upscale_models` 目录（含 `extra_model_paths.yaml`），而不再只看第一个。
- **20260831 - 整理代码结构，修复bugs，添加一些新功能。** 添加`dynamic fade`改善多步数下接缝过渡，添加`brightness match`改善亮度闪烁问题。
- **20260829 - 增加Fun ControlNet相关节点以提升插件性能表现。** 新加入两个节点 `MMH3 Fun Controlnet Inpaint` 和 `MMH3 Spatial Inpaint Params`, 前者有助于消除画面分块的缝隙，后者有助于在较高降噪幅度下保持画面原有内容。要使用这些节点你需要把 [kijai的提交申请](https://github.com/Comfy-Org/ComfyUI/pull/15860) 合并进你的comfyui。
- **20260825 - 新增实验性质的 `LTX25 Ultimate Upscale` 节点。** 在原有 MMH3 流程基础上，扩展支持 LTX2.5 的嵌套 AV latent（视频 `[B,128,T,H,W]` + 音频 `[B,C,time,freq]`）：单节点完成 时间分块 → latent 放大（固定 2x 模型放大后再插值到目标宽高） → 空间分块 → 逐 tile 采样 → 拼接。提供 `LTX25 Latent Upscale Params` / `LTX25 Temporal Split Params` / `LTX25 Spatial Split Params` 三个可选参数节点。音频目前有问题，你应该用原始音频latent。这些节点目前是高度实验性质的，不要依赖它们。

---

## 功能介绍

- **显存有限也能放大长时间、高分辨率视频。** 这是本节点的核心设计目标：时间分块让任意长的视频都能放进显存，空间分块让任意高的分辨率都能放进显存，且同一时刻只采样一个 tile——因此无论视频多长、输出分辨率多高，峰值显存都只占"一个 tile"。
- **单节点，全流程。** 时间分块（外循环）、可选的 latent 放大、空间分块（内循环）、逐块扩散采样、空间拼接与时间拼接，全部由一个 `MMH3 Ultimate Upscale` 节点驱动。
- **支持长视频的时间分块。** 长片段被切成带重叠的时间块，逐块独立处理后再拼回。
- **每块两种放大方式：**
  - **H3 3D 模型放大**（`MMH3 Latent Upscale with Model Params`）——使用 `latent_upscale_models` 目录下的 `minimax_h3_latent_upscaler_3d_*.safetensors` 权重（H3 3D 放大模型）。
  - **无模型插值放大**（`MMH3 Latent Upscale Params`）——仅对视频 latent 做空间插值（nearest / bilinear / area / bicubic），无需额外模型，音频保持不变。等价于 ComfyUI 的 *Upscale Latent* 节点，但保留嵌套的 AV 结构。
- **空间分块，显存可控。** 每个时间块再切成 tile，同一时刻只采样一个 tile，因此峰值显存只占"一个 tile"，而非整帧。
- **音频完整保留。** latent 中的音频部分在每一块、每一次拼接中都原样携带，从不重新采样。
- **各阶段均可选。** `latent_upscale_param`、`temporal_split_param`、`spatial_split_param` 全部可选，不连接即跳过该阶段（不放大 / 整段单块 / 整块采样）。
- **友好的显存管理。** 3D 放大模型在每次用完后卸载回 CPU；放大进行时把扩散模型也卸载，因此 H3 与放大模型不会同时驻留显存（下一次采样时 H3 会自动重载）。
- **逐块的 conditioning 处理。** conditioning 按时间块重新锚定、按 tile 裁切关键帧；关键帧视频 latent 会被缩放到（可能已放大的）块的网格尺寸。

---

## 优点

### 时序一致性 & 过渡顺滑
- **首帧锚定（frame-0 anchor）。** 除第一块外，每个时间块的首帧关键帧会被替换为上一块已重采样边界帧（`anchor_conditioning`，由 `anchor_strength` 控制，默认 `0.999`，对应 *Anchor MiniMax H3 Latent* 节点的行为），消除块与块接缝处的细节错位。
- **交叉淡入淡出拼接（cross-fade）。** 重叠的时间块在重叠区用线性交叉淡变（`temporal_append` / `_crossfade`）混合，块间接缝是平滑过渡而非硬切。

### 像素空间一致性 & 过渡顺滑
- **冻结重叠遮罩（frozen overlap mask）。** 每个 tile 都在其真实尺寸上采样，但它与已拼接邻居共享的重叠条带会从累计结果中预填充，并用 `noise_mask`（`spatial_fade_mask`）锁定——重采样只能改变"自由内部"，共享接缝内容被精确保留。
- **遮罩写回。** 采样完成后，用 `torch.where(band, stitched, tile)` 把冻结的接缝区域写回，保证已经一致的接缝绝不会被覆盖。
- **可配置的接缝混合。** tile 之间的重叠带可用 `overlap_blend`（`linear` / `smoothstep` / `overwrite` / `midpoint`）配合 `overlap_mode`（`earlier` 优先 / `later` 优先）混合，完全控制相邻 tile 的过渡方式——平滑而不块状。

### 其他
- **峰值显存仅一个 tile**，得益空间分块与模型卸载。
- **音频从不重采样**——无音频伪影、无额外开销。
- **不强制下载模型**——可在"H3 3D 模型放大"与"无模型插值放大"两条路径间自由选择。

---

## 为什么分块重采样优于动态显存交换

新版 PyTorch/CUDA 支持按需在内存与显存之间搬运权重，理论上即使工作集超出显存也能不加分块直接生成。但实际上这会**慢得多**，根源是一到两个数量级的带宽鸿沟：

| 路径 | 典型带宽 |
|------|----------|
| GPU 显存（GDDR/HBM） | ~1000 GB/s |
| PCIe 4.0 x16（内存 ↔ 显卡） | ~32 GB/s |
| PCIe 5.0 x16 | ~64 GB/s |

扩散采样要串行跑几十个去噪步，而 transformer 每一步都按**相同的循环顺序**访问各 block——这对任何驻留缓存都是最坏情况：一旦权重放不下，几乎所有非驻留字节都要在*每一步*重新走一遍 PCIe：

```
T(动态交换) ≈ 步数 × 非驻留字节数 / PCIe带宽        ← 受带宽限制
T(分块)     ≈ 分片数 × 单片FLOPs / GPU算力           ← 受算力限制
```

统一内存的缺页处理还会串行化 CUDA 流，导致 GPU 利用率塌缩到个位数。分块重采样则把峰值工作集限定为"**全部权重 + 单个分片的激活**"，让每次 forward 都保持在算力受限区间，代价只是一点可预期的冗余：

- 空间方向：≈ ((tile + overlap) / tile)² —— 如 512px tile 配 128px overlap → ×1.56
- 时间方向：≈ (chunk + overlap) / chunk —— 如 136 帧配 17 帧 overlap → ×1.13

一个 ~1.2–1.8× 的常数代价，永远好过每字节 15–60× 的惩罚。内存↔显存交换只有在"超出量小且每个被搬字节复用率高"时才划算，大 DiT 的多步去噪两条都不满足。

---

## 按显存容量调节 chunk 与 tile 尺寸

单片采样的显存 ≈ **模型权重 + 激活**，激活部分大致正比于 `chunk_length × tile_width × tile_height`：

- **太大** → 该分片落入上文所述的流式交换区（每字节慢 15–60 倍）。
- **太小** → 固定的 overlap 开销占主导：每个方向都要付出 `(尺寸 + overlap) / 尺寸` 的冗余，比如把 tile 缩到 256px 而固定 128px overlap 时，冗余像素已达 ×2.25。

目标是找到**让峰值显存恰好贴着容量上限的最大分片**（采样第一个 tile 时盯着 ComfyUI 的显存表），并把总冗余控制在约 ×1.3–1.8。H3 的推荐起点：

| 显存 | tile_width × tile_height | chunk_length（17 像素帧的倍数） |
|------|--------------------------|--------------------------------|
| 8 GB | 320–384                  | 34–68                          |
| 12 GB | 384–512                 | 51–102                         |
| 16 GB | 512–576                 | 102–153                        |
| 24 GB | 576–768                 | 136–170                        |

说明：
- 以上数值假设使用量化/裁剪版 H3 权重；请以自己实测峰值为准，每次只调一档。
- tile 不要低于 256px，否则固定 overlap 会吃掉预算。
- LTX25 节点在 32px 网格上遵循同样原则；如果权重无论如何都无法常驻，更小的分片依然能通过减少溢出流量带来收益。

---

## 节点列表

| 节点 | 作用 |
|------|------|
| **MMH3 Ultimate Upscale** | 主节点，运行整个流程。输入：`latent`、`conditioning`、`model`、`noise`、`sampler`、`sigmas`、可选 `negative` + `cfg`，以及三个可选参数输入。 |
| **MMH3 Temporal Split Params** | `chunk_length`（像素帧，17 的倍数）、`temporal_overlap`（17 的倍数）、`anchor_strength`。 |
| **MMH3 Spatial Split Params** | `tile_width` / `tile_height`（像素，32 的倍数）、`spatial_w_overlap` / `spatial_h_overlap`（像素，32 的倍数）、`fade_width` / `fade_height`（接缝遮罩淡变）、`min_tile_size`、`overlap_mode`、`overlap_blend`。 |
| **MMH3 Latent Upscale with Model Params** | H3 3D 模型放大：`model_name`、`width`、`height`（吸附到 32 的倍数）、`device`、`precision`。 |
| **MMH3 Latent Upscale Params** | 无模型插值：`method`、`width`、`height`（吸附到 32 的倍数）。 |

### 典型工作流
1. 用 MiniMax H3 生成一段 AV latent（视频+音频在同一个 latent 中）。
2.（可选）`MMH3 Temporal Split Params` → 连到 `temporal_split_param`。
3.（可选）`MMH3 Latent Upscale with Model Params` **或** `MMH3 Latent Upscale Params` → 连到 `latent_upscale_param`。
4.（可选）`MMH3 Spatial Split Params` → 连到 `spatial_split_param`。
5. 把 `latent`、`conditioning`、`model`、`noise`、`sampler`、`sigmas` 接入 `MMH3 Ultimate Upscale`。
6. 用 H3 VAE 解码输出 latent。

> 你设置的放大 width/height 必须与 **conditioning 的生成尺寸**一致（即视频被放大后、conditioning 所基于的尺寸）。

### 放大模型格式：仅支持 `.safetensors`

`MMH3 Latent Upscale with Model Params` 只列出并加载 **`.safetensors`** 检查点（同时接受 `.sft`，与 ComfyUI 核心一致）。`.pth` / `.pt` / `.ckpt` 等 pickle 格式在扫描阶段被跳过，并在文件被读取之前直接拒绝。

**原因：** 用 `torch.load()` 加载 pickle 权重时，选中的那一刻就可能执行任意代码。`weights_only=True` 只能降低风险，在 torch < 2.6 上并不可靠（存在 pickle 阶段绕过，[CVE-2025-32434](https://nvd.nist.gov/vuln/detail/CVE-2025-32434)）。彻底拒绝 pickle 才能消除这个攻击面。

**如果你手上有信任的 `.pth` 权重**（例如 `minimax_h3_latent_upscaler_3d_fp32.pth`），转换一次即可：

```python
import torch, safetensors.torch
sd = torch.load('model.pth', map_location='cpu', weights_only=True)
sd = sd['model'] if isinstance(sd, dict) and 'model' in sd else sd
safetensors.torch.save_file(sd, 'model.safetensors')
```

如果 `weights_only=True` 报错，说明这个文件不是纯 state dict，不应被信任——不要靠改回 `weights_only=False` 来绕过。仍留在目录里被忽略的文件，会在启动时打印到控制台。

---

## 参考项目

本节点构建于以下现有社区项目之上，单独列出：

- **Latent 分割（时间 / 空间分块、锚定、拼接逻辑）：**  
  https://github.com/bbaudio-2025/Comfyui-MiniMax-H3-LatentSplit
- **Latent 模型放大（H3 3D 放大模型权重与推理）：**  
  https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler

其中 H3 3D 放大模型的网络代码与归一化统计量改编自第二个项目；时间/空间分块、锚定与拼接逻辑沿用第一个项目。

---

## 额外信息

本项目使用AI编写，如果你运行遇到任何问题，最好问AI。😂
