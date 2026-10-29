# ComfyUI-P1-MeshTools

**P1 只保留最大体**（`P1KeepLargestBody`）—— 一个把「删网格碎屑」自动化的小节点。

用于 3D 打印链（PixelArtistry / TRELLIS.2 / Pixal3D / WTiVo 体素水密化）出片后的最后一步：
一个网格里通常只有 1 个主体，其余是几十上百个顶点数极小的游离碎块，切片软件会被它们拖慢甚至报错。

## 为什么需要它

PixelArtistry 打印链官方 README 要求**人工**做这一步：

> Look for loose fragments. If `bodies` is greater than 1 in the console:
> Edit Mode → click the model → Ctrl+L → Ctrl+I → X → Vertices.

本节点把这套 Blender 手动操作变成一次连线。实测一条 LOD 之后的网格：

| 指标 | 数值 |
|---|---|
| 总顶点 | 3,993,640 |
| 连通体 | 97 个 |
| 最大体占比 | 99.956% |
| 其余 96 个碎屑 | 每个 12 ~ 417 顶点，合计 0.0441% |

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/P1-ShiGuang/ComfyUI-P1-MeshTools.git
# 重启 ComfyUI
```

依赖：`numpy`、`torch`（ComfyUI 已自带）。`scipy` **可选** —— 装了走 `scipy.sparse.csgraph.connected_components`（大网格快很多），没装自动退化为纯 numpy 传播，结果一致。

```bash
pip install scipy        # 可选但推荐
```

## 节点

路径：`P1_时光 / 网格 → P1 只保留最大体`

| 输入 | 类型 | 说明 |
|---|---|---|
| `mesh` | MESH | 上游任意网格节点 |
| `mode` | 枚举 | 见下表，**默认「丢小碎屑(按比例)」** |
| `min_ratio` | FLOAT | 碎屑判据，默认 `0.001` |

| mode | 行为 |
|---|---|
| `丢小碎屑(按比例)` | 只丢掉顶点数 < 最大体顶点数 × `min_ratio` 的连通体（默认，安全；保留合理的独立件） |
| `只保留最大体` | 只留顶点数最多的连通体（最干净；模型本身是多独立件时会被一起丢掉） |
| `关闭` | 直通，不处理（等于不装这个节点） |

输出：`mesh`（顶点/面/uv/顶点色/法线/切线一并按新顶点号重映射）。

**关键实现点**：删顶点后必须把面索引重映射到新编号，否则面索引越界——这是这类节点最容易踩的坑，本实现已处理。

## 日志示例

```
[P1-碎屑清理] 第1件: 体 97→1 | 顶点 3993640→3985182 | 面 7952416→7949068 | 最大体 3985182 顶点 | 丢碎屑 96 个/8458 顶点(0.2118%)
```

## License

MIT
