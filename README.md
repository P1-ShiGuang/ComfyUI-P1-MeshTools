# P1_时光 · Pixal3D 3D 打印工作流分享包

一个能从**一张图**跑出**可打印 3D 模型**的 ComfyUI 工作流，加上它用到的那一个自研节点
**P1 只保留最大体**（`P1KeepLargestBody`）。

> 本仓库对应分享包 `Pixal3D模型和工作流：P1_时光.zip` 里**除模型权重以外**的全部内容：
> 工作流 json、自研节点、三份说明文字。
> **模型权重（13.85 GB）不在仓库里** —— GitHub 单文件上限 100 MB，装不下，请按第三节清单自行放置。

---

## 一、仓库里有什么

| 路径 | 是什么 |
|---|---|
| `workflows/3D模型工作流B站：P1_时光.json` | ComfyUI 工作流（面板里 Load，或直接拖进画布） |
| `__init__.py` + `requirements.txt` | 自研节点 `P1KeepLargestBody`（仓库根目录 = 直接可克隆的插件目录） |
| `docs/模型放置说明.txt` | 原包里的模型放置说明（目录结构、每个模型对应哪个节点号） |
| `docs/插件安装说明.txt` | 原包里的插件安装说明（4 个第三方包怎么装、节点变红怎么查） |
| `docs/把qwen3vl_8b放这里.txt` | 文本编码器放置说明 |

---

## 二、安装本节点

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/P1-ShiGuang/ComfyUI-P1-MeshTools.git
pip install scipy      # 可选，大网格快很多；不装自动退化纯 numpy
# 重启 ComfyUI
```

节点路径：`P1_时光 / 网格 → P1 只保留最大体`

| 输入 | 类型 | 说明 |
|---|---|---|
| `mesh` | MESH | 上游任意网格节点 |
| `mode` | 枚举 | **默认「丢小碎屑(按比例)」** |
| `min_ratio` | FLOAT | 碎屑判据，默认 `0.001`（顶点数 < 最大体×0.1% 才丢） |

| mode | 行为 |
|---|---|
| `丢小碎屑(按比例)` | 只丢小于 最大体×`min_ratio` 的连通体（默认，安全；保留合理独立件） |
| `只保留最大体` | 只留顶点数最多的连通体（最干净；多独立件模型会被一起丢掉） |
| `关闭` | 直通，不处理 |

**为什么需要它**：PixelArtistry 打印链官方 README 要求人工在 Blender 里做这一步
（Edit Mode → Ctrl+L → Ctrl+I → X → Vertices）。本节点把它变成一根线。
实测一条 LOD 之后的网格：3,993,640 顶点、97 个连通体，最大体占 99.956%，
其余 96 个碎屑合计仅 0.0441% —— 不清理会全部打进打印件。

**实现重点**：删顶点后必须把面索引重映射到新编号，否则面索引越界，这是这类节点最容易踩的坑。
uv / 顶点色 / 法线 / 切线也一并按新顶点号重排。

日志示例：

```
[P1-碎屑清理] 第1件: 体 97→1 | 顶点 3993640→3985182 | 面 7952416→7949068 | 最大体 3985182 顶点 | 丢碎屑 96 个/8458 顶点(0.2118%)
```

---

## 三、模型权重（需自行下载，本仓库不包含）

放进 `ComfyUI/models/`，按下面目录结构：

| 文件 | 大小 | 放到 | 对应节点 |
|---|---|---|---|
| `pixal3d_int8_convrot.safetensors` | 5.20 GB | `diffusion_models/` | #948 基础几何 |
| `pixal3d_multiview_int8_convrot.safetensors` | 5.20 GB | `diffusion_models/` | #319 多视图细化 |
| `trellis_2_shape_vae_bf16.safetensors` | 1.02 GB | `vae/` | #117 结构 VAE |
| `trellis_2_texture_vae_bf16.safetensors` | 0.88 GB | `vae/` | #906 贴图 VAE |
| `dino_v3_L_naf_fp32.safetensors` | 1.13 GB | `clip_vision/` | #15 图像特征 |
| `birefnet.safetensors` | 0.41 GB | `background_removal/` | #193 去背景 |

合计 **13.85 GB**。若你的 ComfyUI 用 `unet/` 而不是 `diffusion_models/`，两个 pixal3d 放 `unet/` 也能识别。

另外还要自己下载 3 个（本包未含，「一图生四视角」那一段用）：

| 文件（文件名必须完全一致） | 大小 | 放到 | 对应节点 |
|---|---|---|---|
| `qwen_image_2.1_int8_convrot.safetensors` | 约 7.3 GB | `diffusion_models/` | #928 |
| `qwen3vl_8b_int8_convrot.safetensors` | 约 9.4 GB | `text_encoders/` | #929 |
| `qwen_image_2.1_vae_bf16.safetensors` | 约 0.7 GB | `vae/` | #930 |

缺这三个，工作流从 #928 开始就加载失败。

---

## 四、软件环境

| 项 | 要求 |
|---|---|
| ComfyUI | 需带 3D 节点的版本（`comfy_extras` 里有 `nodes_trellis2` / `nodes_mesh_postprocess` / `nodes_mesh_io` / `nodes_save_3d` / `nodes_load_3d` / `nodes_hunyuan3d`）；本包在 **0.39.0** 上验证通过 |
| Python | ≥ 3.10（推荐 3.12） |
| PyTorch | ≥ 2.1（int8 模型靠 `torch._int_mm`，低于 2.1 直接报错）；推荐 2.6+，CUDA cu124 以上 |
| GPU | NVIDIA，Turing(sm75) 起，显存 ≥ 20 GB |
| 内存 | ≥ 48 GB（LOD 那一步常驻 40~55 GB） |
| Blender | 4.5（LOD 降面用），路径写在节点 #333，装完改成你自己的 |

**第三方自定义节点（4 个，需自己装，作者 Mstafa-awad）**

- `WTiVo-WatertightVoxel-ComfyuiNode` — 节点 #327 体素重建+水密
- `WTiVo-FastMergeByDistance` — 节点 #330 按距离合并
- `LODTailor-The-Mesh-Trimmer-ComfyuiNode` — 节点 #328 Blender 降面
- `ComfyUI-MemoryCleaner` — 显存清理（工作流里 7 处）

WTiVo 需要自己的运行环境：把 `python_wtivo`（Python 3.12 + torch 2.8 + cu128）放在 ComfyUI 同级目录。

---

## 五、跑之前必须改两处

1. **节点 #122 LoadImage** → 换成你自己的图（单人、全身、脚底离画面边缘留白）。
   若生成的模型「没有脚/没有腿」，问题不在分辨率 —— 是 WTiVo 的 `faithc_component_mode` 用了 `largest`，
   把独立的腿当碎壳删了。改成 `keep_all`（本工作流已改好），并保留 #924 `P1KeepLargestBody(0.001)` 只清极小碎屑。
2. **节点 #333 PrimitiveString** → Blender exe 路径，改成你机器上的。

跑法：面板里 Load 这个 json → Queue。**中途不要点「保存」** —— 一保存，面板里的旧参数会把文件覆盖回去。
建议串行跑，不要同时提交两个任务。

耗时参考（RTX 3080 20GB / 16 核 / 64GB）：1536 档一轮约 22~40 分钟
（WTiVo ≈2 分钟 → LOD 降面 5~10 分钟 → 打印件保存 → 彩色链）。
输出在 `ComfyUI/output/3d/`，打印件约 240 MB，彩色件约 107 MB。

---

## License

MIT（仅覆盖本仓库自研节点 `P1-MeshTools` 与工作流文件；工作流里引用的第三方节点版权归各自作者）
