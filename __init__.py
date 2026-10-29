"""P1 网格碎屑清理 —— 丢掉网格里的游离碎片（小连通体）。

由来：PixelArtistry 打印链官方 README 明确要求人工做这一步——
  "Look for loose fragments. If `bodies` is greater than 1 in the console:
   Edit Mode -> click the model -> Ctrl+L -> Ctrl+I -> X -> Vertices."
实测本链 LOD 之后：3,993,640 顶点里共 97 个连通体，除最大体（占 99.956%）外
其余 96 个都是 12~417 顶点的碎渣，合计仅占 0.0441%。
本节点把这一步自动化，免得每次出片都手动进 Blender 删。

三种模式（关键后处理可关闭）：
  "丢小碎屑(按比例)" : 只丢掉小于 最大体×min_ratio 的碎屑（默认，安全；保留合理独立件）
  "只保留最大体"     : 只留顶点数最多的连通体（最干净；模型若本身有多个独立件会被一起丢掉）
  "关闭"             : 直通，不处理
"""
import copy

import numpy as np
import torch

try:
    import scipy.sparse as _sp
    from scipy.sparse.csgraph import connected_components as _cc
except Exception:  # 没有 scipy 时退化为纯 numpy 传播
    _sp = None
    _cc = None


MODE_TINY = "丢小碎屑(按比例)"
MODE_LARGEST = "只保留最大体"
MODE_OFF = "关闭"
MODES = [MODE_TINY, MODE_LARGEST, MODE_OFF]

PER_VERTEX_ATTRS = ("uvs", "vertex_colors", "normals", "tangents")


def body_labels(n_verts, faces_np):
    """按顶点-边连通性分解连通体。返回 (labels[N] int64, sizes[K] int64)。"""
    f = np.asarray(faces_np, dtype=np.int64)
    if f.size:
        e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=0)
        e = np.unique(np.sort(e, axis=1), axis=0)
        e = e[(e[:, 0] >= 0) & (e[:, 1] >= 0) & (e[:, 0] != e[:, 1])]
    else:
        e = np.zeros((0, 2), dtype=np.int64)

    if _sp is not None:
        g = _sp.coo_matrix((np.ones(len(e), dtype=np.int8), (e[:, 0], e[:, 1])),
                           shape=(n_verts, n_verts)).tocsr()
        k, lab = _cc(g, directed=False)
        lab = lab.astype(np.int64)
    else:
        lab = np.arange(n_verts, dtype=np.int64)
        if len(e):
            a, b = e[:, 0], e[:, 1]
            for _ in range(64):
                m = np.minimum(lab[a], lab[b])
                before = lab.copy()
                np.minimum.at(lab, a, m)
                np.minimum.at(lab, b, m)
                lab = lab[lab]
                if np.array_equal(lab, before):
                    break
    sizes = np.bincount(lab, minlength=int(lab.max()) + 1 if lab.size else 0)
    return lab, sizes.astype(np.int64)


def pick_bodies(lab, sizes, mode, min_ratio):
    """按模式挑要保留的连通体编号。"""
    if sizes.size == 0:
        return np.zeros(0, dtype=np.int64)
    biggest = int(sizes.max())
    if mode == MODE_LARGEST:
        return np.array([int(np.argmax(sizes))], dtype=np.int64)
    thr = max(1.0, biggest * float(min_ratio))
    keep = np.nonzero(sizes >= thr)[0]
    return keep.astype(np.int64)


def clean_arrays(vertices, faces, mode, min_ratio, extra=None):
    """核心算法（纯 numpy/torch）。返回 (v_new, f_new, stats, extra_sliced)。

    注意：删顶点后必须把面重映射到新编号，否则面索引会越界（这是最容易踩的坑）。
    """
    n = int(vertices.shape[0])
    f_np = faces.detach().cpu().numpy() if isinstance(faces, torch.Tensor) else np.asarray(faces)
    lab, sizes = body_labels(n, f_np)
    keep_bodies = pick_bodies(lab, sizes, mode, min_ratio)

    keep_v = np.isin(lab, keep_bodies)
    v_idx = np.nonzero(keep_v)[0]
    f_np = np.asarray(f_np, dtype=np.int64)
    f_idx = np.nonzero(keep_v[f_np[:, 0]] & keep_v[f_np[:, 1]] & keep_v[f_np[:, 2]])[0] if len(f_np) else np.zeros(0, dtype=np.int64)

    # 旧顶点号 -> 新顶点号
    remap = np.full(n, -1, dtype=np.int64)
    remap[v_idx] = np.arange(v_idx.size, dtype=np.int64)
    f_new_np = remap[f_np[f_idx]] if f_idx.size else np.zeros((0, 3), dtype=np.int64)

    dev = vertices.device
    v_new = vertices[torch.as_tensor(v_idx, device=dev)]
    f_new = torch.as_tensor(f_new_np, device=faces.device if isinstance(faces, torch.Tensor) else dev,
                            dtype=faces.dtype if isinstance(faces, torch.Tensor) else torch.int64)

    stats = {
        "bodies_before": int(sizes.size),
        "bodies_after": int(len(keep_bodies)),
        "verts_before": n,
        "verts_after": int(v_idx.size),
        "faces_before": int(len(f_np)),
        "faces_after": int(f_idx.size),
        "largest": int(sizes.max()) if sizes.size else 0,
        "dropped_bodies": int(sizes.size - len(keep_bodies)),
        "dropped_verts": int(n - v_idx.size),
    }
    out_extra = None
    if extra:
        out_extra = {}
        for k, t in extra.items():
            if isinstance(t, torch.Tensor) and t.dim() >= 2 and int(t.shape[0]) == n:
                out_extra[k] = t[torch.as_tensor(v_idx, device=t.device)]
            else:
                out_extra[k] = t
    return v_new, f_new, stats, out_extra


class P1KeepLargestBody:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH",),
                "mode": (MODES, {"default": MODE_TINY,
                                 "tooltip": "丢小碎屑=按比例丢；只保留最大体=最干净；关闭=直通"}),
                "min_ratio": ("FLOAT", {"default": 0.001, "min": 0.0, "max": 1.0, "step": 0.0001,
                                        "tooltip": "碎屑判据：顶点数 < 最大体顶点数×该比例 才丢（0.001=0.1%）"}),
            },
        }

    RETURN_TYPES = ("MESH",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "clean"
    CATEGORY = "P1_时光/网格"

    def clean(self, mesh, mode=MODE_TINY, min_ratio=0.001):
        if mode == MODE_OFF:
            print("[P1-碎屑清理] 关闭（直通）", flush=True)
            return (mesh,)

        v_all = mesh.vertices
        f_all = mesh.faces
        squeeze = v_all.dim() == 2
        if squeeze:
            v_all = v_all.unsqueeze(0)
            f_all = f_all.unsqueeze(0)

        counts_v = getattr(mesh, "vertex_counts", None)
        counts_f = getattr(mesh, "face_counts", None)
        b = int(v_all.shape[0])

        out_v, out_f, out_attrs, totals = [], [], {k: [] for k in PER_VERTEX_ATTRS}, None
        for i in range(b):
            nv = int(counts_v[i]) if counts_v is not None else int(v_all.shape[1])
            nf = int(counts_f[i]) if counts_f is not None else int(f_all.shape[1])
            v_i, f_i = v_all[i][:nv], f_all[i][:nf]
            extra = {}
            for k in PER_VERTEX_ATTRS:
                t = getattr(mesh, k, None)
                if isinstance(t, torch.Tensor) and t.dim() >= 2 and int(t.shape[0]) == b and int(t.shape[1]) >= nv:
                    extra[k] = t[i][:nv]
            vi_new, fi_new, stats, sliced = clean_arrays(v_i, f_i, mode, min_ratio, extra)
            v_new, f_new = vi_new, fi_new
            out_v.append(v_new)
            out_f.append(f_new)
            for k, t in (sliced or {}).items():
                out_attrs[k].append(t)
            totals = stats
            print("[P1-碎屑清理] 第%d件: 体 %d→%d | 顶点 %d→%d | 面 %d→%d | 最大体 %d 顶点 | 丢碎屑 %d 个/%d 顶点(%.4f%%)"
                  % (i + 1, stats["bodies_before"], stats["bodies_after"],
                     stats["verts_before"], stats["verts_after"],
                     stats["faces_before"], stats["faces_after"], stats["largest"],
                     stats["dropped_bodies"], stats["dropped_verts"],
                     100.0 * stats["dropped_verts"] / max(1, stats["verts_before"])), flush=True)

        out = copy.copy(mesh)
        if b == 1:
            out.vertices, out.faces = out_v[0].unsqueeze(0), out_f[0].unsqueeze(0)
            out.vertex_counts = None
            out.face_counts = None
        else:
            # 变长批：补齐后重新打包
            maxv = max(int(t.shape[0]) for t in out_v)
            maxf = max(int(t.shape[0]) for t in out_f)
            pad_v = [torch.nn.functional.pad(t, (0, 0, 0, maxv - t.shape[0])) for t in out_v]
            pad_f = [torch.nn.functional.pad(t, (0, 0, 0, maxf - t.shape[0])) for t in out_f]
            out.vertices, out.faces = torch.stack(pad_v), torch.stack(pad_f)
            out.vertex_counts = torch.tensor([t.shape[0] for t in out_v], device=v_all.device)
            out.face_counts = torch.tensor([t.shape[0] for t in out_f], device=v_all.device)
        for k, lst in out_attrs.items():
            if lst and len(lst) == b:
                out.__dict__[k] = lst[0].unsqueeze(0) if b == 1 else torch.stack(lst)
        return (out,)


NODE_CLASS_MAPPINGS = {"P1KeepLargestBody": P1KeepLargestBody}
NODE_DISPLAY_NAME_MAPPINGS = {"P1KeepLargestBody": "P1 只保留最大体"}
