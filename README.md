# Sol-Attn v5 for MiniMax-H3 — 部署与使用

ComfyUI 自定义节点 `Patch Sol-Attn (MiniMax)`(v5)+ **comfy-kitchen int8 Sol-Attn 内核**编译方法。
为 MiniMax-H3 提供训练无关的块稀疏注意力加速(官方 PR:Comfy-Org/comfy-kitchen#117,已合并进 main)。

## 仓库内容

| 文件 | 作用 | 放哪 |
|---|---|---|
| `sol_attn_minimax_v5.py` | v5 节点(自适应 tau / top-k SLA / VSA 三种模式) | `ComfyUI/custom_nodes/` |
| `rebuild_kitchen.bat` | comfy-kitchen 编译脚本(改动其中两行路径即可用) | 任意位置,双击运行 |

## 前置环境(不满足会编译失败/回退)

| 项 | 要求 | 说明 |
|---|---|---|
| 系统 | Windows + ComfyUI ≥ 0.33(需要 `comfy_api.latest`) | |
| Python | ComfyUI 的 `.venv`,建议 3.13 | `_C.abi3.pyd` 为 abi3,但 torch/CUDA 需要一致 |
| PyTorch | **2.12+cu130(与编译机一致最好)** | CUDA 扩展链接了 cu130 runtime |
| 编译工具链 | **VS 2022 BuildTools**(Desktop C++ 工作负载)+ **CUDA Toolkit ≥ 12.8**(实测 13.0) | `vcvars64.bat` 路径见脚本 |
| 网络 | 可访问 GitHub(可配 gh-proxy 代理) | 见 `rebuild_kitchen.bat` 注释 |

要省去自己编译:也可以从同 torch/cuda 版本的机器**直接拷贝 `site-packages/comfy_kitchen/` 整个目录**(含 `backends/cuda/_C.abi3.pyd`)到目标机对应位置。

## 编译步骤(rebuild_kitchen.bat)

1. 准备 comfy-kitchen 源码(合并了 PR117 的版本,建议 **Comfy-Org main**):
   ```
   git clone https://github.com/Comfy-Org/comfy-kitchen.git F:\ComfyUi\ComfyUI\comfy-kitchen
   cd comfy-kitchen && git switch main
   ```
2. 编辑 `rebuild_kitchen.bat`,改两处:
   - `vcvars64.bat` 路径 → 你机器上 VS BuildTools 的实际路径
   - `cd /d` 的源码目录、`.venv` 路径 → 你的实际路径
3. 双击运行(约 10~20 分钟,全架构 `75-real;80-real;86-real;89;120`,含 **sm120/5060 Ti**;只需当前卡可删掉其他架构提速)
4. 验证:
   ```
   .venv\Scripts\python -c "from comfy_kitchen.backends import cuda; print(hasattr(cuda,'sol_attn_chunked'))"
   ```
   输出 `True` 即成功(分段 QKV 生产者可用)。

## 节点用法

工作流里放在 UNET 之后(自动链到既有 attention override):

| 参数 | 建议 |
|---|---|
| `selection` | `adaptive tau`:通用,tau 1.3(≈16% 稀疏);`top-k (SLA)`:配 lightx2v SLA LoRA,keep **15**(蒸馏值)/10(更快);`VSA (FastVideo)`:配 FastH3-VSA 检查点,vsa_keep **10** |
| `start_percent` / `end_percent` | 普通检查点 0.2 / 0.9;**few-step(4 步)检查点必须 0 / 1** |
| `min_tokens` | 12288(默认;短序列稀疏不划算) |
| `sink_conditioning` | `exact_kv_and_rows`(默认,保护 H3 条件行) |
| `verbose` | 排查时开,确认内核是否生效 |

## 内核是否生效的判定(日志)

```
[sol_attn] chunked qkv producer on 50 blocks            ← producer 已挂
[sol_attn] producer path: N tokens, chunked qkv, topk=…  ← 稀疏在跑
[sol_attn] dense …: …, 原因                              ← 回退 dense(看原因)
```

没有任何 `[sol_attn]` 日志 = 节点没挂上或 verbose 关闭;出现 `kernel failed (...) falling back` = 内核未生效(检查编译产物/架构)。

## 配套 LoRA(可选,top-k SLA 模式专用)

**`minimax_h3_fl2v_turbo_4step_v0.1_768p_sla_comfyui_bf16.safetensors`**(≈2GB,lightx2v 官方 SLA 蒸馏 LoRA)

- HF 首页:https://huggingface.co/lightx2v/Minimax-h3-Turbo-SLA
- 直连下载:
  ```
  https://huggingface.co/lightx2v/Minimax-h3-Turbo-SLA/resolve/main/minimax_h3_fl2v_turbo_4step_v0.1_768p_sla_comfyui_bf16.safetensors
  ```
- 国内镜像(hf-mirror,速度快):
  ```
  https://hf-mirror.com/lightx2v/Minimax-h3-Turbo-SLA/resolve/main/minimax_h3_fl2v_turbo_4step_v0.1_768p_sla_comfyui_bf16.safetensors
  ```
- 放置: `ComfyUI/models/loras/`
- 搭配:节点 `selection=top-k (SLA)`、`keep_percent=15`(蒸馏对齐值;10 更快)、采样 `steps=4`、`start_percent=0 / end_percent=1`、任务走 **fl2v/首尾帧**(FL2V 系蒸馏),参考音频 44.1kHz 立体声

## 常见问题

- **编译失败 `CMake configuration failed`**:先删源码目录下的 `build/` 缓存再重编;确认 `cl.exe` 已就绪(脚本会走 vcvars64)
- **回退 dense / `no attribute 'sol_attn_chunked'`**:kernel 包太旧(缺 PR117 合并)——更新 main 重编;或节点文件与内核不配套
- **换机器**:推荐直接用相同 torch/CUDA 的机器拷贝 `comfy_kitchen` 包;版本不一致请在本机重编
- **与 MiniMaxH3 Director 共存**:Director 的多段引导会识别本节点的 layout 观察器(详见 Director fork `r2v-source` 的兼容补丁)

## 致谢

- [Comfy-Org/comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen)(PR #117 int8 Sol-Attn)
- Sol-Attn:training-free block-sparse attention(arXiv 2607.24027)
- 节点本体来源:PR #117 附带的临时测试节点(已做跨版本适配,见提交记录)