# dataset_roi3d.py
# ============================
# Legacy: Case3DDataset (old structure)
# Multiscale: MultiScaleCase3DDataset (new structure: small/large/global)
#
# Scheme C style:
# - NO resize
# - optional center-crop if exceeds cap
# - dynamic padding handled in collate
# ============================

import os
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


# ----------------------------
# helpers
# ----------------------------
def ensure_dhw(arr: np.ndarray) -> np.ndarray:
    """
    Ensure volume is in [D,H,W].
    Heuristic: if input looks like [H,W,D] (D small <=64), transpose.
    """
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape={arr.shape}")
    a, b, c = arr.shape
    if c <= 64 and a >= 64 and b >= 64:
        return np.transpose(arr, (2, 0, 1))
    return arr


def safe_zscore(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean = x.mean()
    std = x.std(unbiased=False)
    return (x - mean) / (std + eps)


def center_crop_if_gt(
    src: np.ndarray,
    cap_h: int,
    cap_w: int,
    cap_d: int,
    cap_d_if_gt: int,
) -> np.ndarray:
    """
    Scheme C:
    - NEVER resize
    - Only center-crop if dimension exceeds cap
    src is [D,H,W]
    """
    d, h, w = src.shape

    if cap_d and cap_d_if_gt and d > cap_d_if_gt and d > cap_d:
        s = (d - cap_d) // 2
        src = src[s : s + cap_d]
        d = cap_d

    if cap_h and h > cap_h:
        s = (h - cap_h) // 2
        src = src[:, s : s + cap_h, :]
        h = cap_h

    if cap_w and w > cap_w:
        s = (w - cap_w) // 2
        src = src[:, :, s : s + cap_w]
        w = cap_w

    return src


def _pick_mask_path(dir_path: str, kind: str) -> str:
    """
    kind in:
      - "target" -> roi_mask_target.npy
      - "all"    -> roi_mask_all.npy
      - "breast" -> breast_mask.npy
      - "none"   -> no mask
    """
    kind = (kind or "none").lower()
    if kind == "target":
        p = os.path.join(dir_path, "roi_mask_target.npy")
    elif kind == "all":
        p = os.path.join(dir_path, "roi_mask_all.npy")
    elif kind == "breast":
        p = os.path.join(dir_path, "breast_mask.npy")
    else:
        return ""
    return p if os.path.isfile(p) else ""


def _load_one_volume_with_optional_mask(
    vol_path: str,
    mask_path: str,
    use_mask: bool,
    dynamic_pad: bool,
    cap_h: int,
    cap_w: int,
    cap_d: int,
    cap_d_if_gt: int,
    fail_safe: bool,
    expected_dhw: Tuple[int, int, int],
) -> torch.Tensor:
    """
    Return tensor [C,D,H,W], C=1 (no mask) or 2 (with mask)
    """
    exp_d, exp_h, exp_w = expected_dhw
    try:
        vol = ensure_dhw(np.load(vol_path)).astype(np.float32)  # [D,H,W]
        if dynamic_pad:
            vol = center_crop_if_gt(vol, cap_h, cap_w, cap_d, cap_d_if_gt)
        vol_t = torch.from_numpy(vol)[None].float()  # [1,D,H,W]
        vol_t = safe_zscore(vol_t)

        if use_mask:
            if (not mask_path) or (not os.path.isfile(mask_path)):
                raise FileNotFoundError(f"mask missing: {mask_path}")
            mask = ensure_dhw(np.load(mask_path)).astype(np.float32)
            if dynamic_pad:
                mask = center_crop_if_gt(mask, cap_h, cap_w, cap_d, cap_d_if_gt)
            mask_t = torch.from_numpy(mask)[None].float().clamp(0, 1)
            x = torch.cat([vol_t, mask_t], dim=0)  # [2,D,H,W]
        else:
            x = vol_t  # [1,D,H,W]
        return x.float()

    except Exception as e:
        if not fail_safe:
            raise
        # return zeros
        d = cap_d if (dynamic_pad and cap_d > 0) else exp_d
        h = cap_h if (dynamic_pad and cap_h > 0) else exp_h
        w = cap_w if (dynamic_pad and cap_w > 0) else exp_w
        c = 2 if use_mask else 1
        x = torch.zeros((c, d, h, w), dtype=torch.float32)
        return x


# ============================================================
# Legacy dataset (old structure)
# data_root/
#   dcis/CASE/roi_xxxx/{image.npy, roi_mask_target.npy}
#   dcis+/CASE/roi_xxxx/{image.npy, roi_mask_target.npy}
# ============================================================
@dataclass
class CaseSample:
    roi_vol_paths: List[str]
    roi_mask_paths: List[str]
    roi_names: List[str]
    label: int
    case_id: str
    cls_name: str
    case_dir: str


class Case3DDataset(Dataset):
    """
    Legacy Scheme C dataset:
    - ROI spatial size NOT unified here
    - optional center-crop (cap) in dataset if dynamic_pad=True
    - padding ONLY in collate_mil_dynamic_pad
    """

    def __init__(
        self,
        data_root: str,
        indices: Optional[List[int]] = None,
        use_mask: bool = False,
        cache_in_ram: bool = False,  # kept for compatibility (not used)
        fail_safe: bool = True,
        seed: int = 42,
        enable_augment: bool = False,  # kept for compatibility (not used)
        expected_d: int = 10,
        expected_h: int = 256,
        expected_w: int = 256,
        mil: bool = False,
        max_rois: int = 4,
        roi_sort: str = "maskvox",  # kept for compatibility (not used here)
        roi_mask_thr: float = 0.1,  # kept for compatibility (not used here)
        dynamic_pad: bool = False,
        cap_h: int = 256,
        cap_w: int = 256,
        cap_d: int = 10,
        cap_d_if_gt: int = 10,
    ):
        self.data_root = str(data_root)
        self.use_mask = bool(use_mask)
        self.fail_safe = bool(fail_safe)
        self.seed = int(seed)
        self.enable_augment = bool(enable_augment)

        self.expected_d = int(expected_d)
        self.expected_h = int(expected_h)
        self.expected_w = int(expected_w)

        self.mil = bool(mil)
        self.max_rois = int(max_rois)

        self.dynamic_pad = bool(dynamic_pad)
        self.cap_h = int(cap_h)
        self.cap_w = int(cap_w)
        self.cap_d = int(cap_d)
        self.cap_d_if_gt = int(cap_d_if_gt)

        self.samples: List[CaseSample] = []
        self._scan()
        self.indices = indices if indices is not None else list(range(len(self.samples)))

    def _scan(self):
        classes = [("dcis", 0), ("dcis+", 1)]
        for cls_name, y in classes:
            cls_dir = os.path.join(self.data_root, cls_name)
            if not os.path.isdir(cls_dir):
                continue

            for case_id in sorted(os.listdir(cls_dir)):
                case_dir = os.path.join(cls_dir, case_id)
                if not os.path.isdir(case_dir):
                    continue

                vols, masks, names = [], [], []
                for roi_name in sorted(os.listdir(case_dir)):
                    roi_dir = os.path.join(case_dir, roi_name)
                    img = os.path.join(roi_dir, "image.npy")
                    if not os.path.isfile(img):
                        continue
                    m = os.path.join(roi_dir, "roi_mask_target.npy")
                    vols.append(img)
                    masks.append(m if os.path.isfile(m) else "")
                    names.append(roi_name)

                if vols:
                    self.samples.append(CaseSample(vols, masks, names, y, case_id, cls_name, case_dir))

    def __len__(self):
        return len(self.indices)

    def _load_one_roi(self, vol_path: str, mask_path: str) -> torch.Tensor:
        return _load_one_volume_with_optional_mask(
            vol_path=vol_path,
            mask_path=mask_path,
            use_mask=self.use_mask,
            dynamic_pad=self.dynamic_pad,
            cap_h=self.cap_h,
            cap_w=self.cap_w,
            cap_d=self.cap_d,
            cap_d_if_gt=self.cap_d_if_gt,
            fail_safe=self.fail_safe,
            expected_dhw=(self.expected_d, self.expected_h, self.expected_w),
        )

    def __getitem__(self, idx):
        s = self.samples[self.indices[idx]]

        meta = {
            "case_id": s.case_id,
            "cls_name": s.cls_name,
            "case_dir": s.case_dir,
        }

        if not self.mil:
            x = self._load_one_roi(s.roi_vol_paths[0], s.roi_mask_paths[0])
            y = torch.tensor([float(s.label)], dtype=torch.float32)
            return x, y, meta

        x_list: List[torch.Tensor] = []
        # IMPORTANT: slice BOTH lists
        for vp, mp in zip(s.roi_vol_paths[: self.max_rois], s.roi_mask_paths[: self.max_rois]):
            x_list.append(self._load_one_roi(vp, mp))

        if not x_list:
            if not self.fail_safe:
                raise RuntimeError("No valid ROI")
            # failsafe: one dummy roi
            c = 2 if self.use_mask else 1
            x_list = [torch.zeros((c, self.expected_d, self.expected_h, self.expected_w), dtype=torch.float32)]

        y = torch.tensor([float(s.label)], dtype=torch.float32)
        return x_list, y, meta


# ============================================================
# Multiscale dataset (new structure)
# data_root/
#   small/dcis/CASE/roi_0000/{image.npy, ...}
#   large/dcis/CASE/{image.npy, ...}
#   global/dcis/CASE/{image.npy, ...}
# and dcis+ similarly
# ============================================================
@dataclass
class MultiScaleCaseSample:
    small_roi_dirs: List[str]
    large_dir: str
    global_dir: str
    label: int
    case_id: str
    cls_name: str


class MultiScaleCase3DDataset(Dataset):
    """
    Returns:
      small_list: List[tensor [C,D,H,W]]  (len <= max_rois_small)
      x_large: tensor [C,D,H,W]
      x_global: tensor [C,D,H,W]
      y: tensor [1]
      meta: dict
    """

    def __init__(
        self,
        data_root: str,
        indices: Optional[List[int]] = None,
        use_mask: bool = True,
        fail_safe: bool = True,
        seed: int = 42,
        enable_augment: bool = False,  # reserved
        max_rois_small: int = 6,
        allow_empty_small: bool = False,

        # mask selection per stream
        mask_kind_small: str = "target",
        mask_kind_large: str = "all",
        mask_kind_global: str = "breast",

        # require existence of large/global
        require_large: bool = True,
        require_global: bool = True,

        # crop (optional) per stream
        dynamic_pad: bool = True,
        cap_small_d: int = 10,
        cap_small_h: int = 128,
        cap_small_w: int = 128,
        cap_large_d: int = 15,
        cap_large_h: int = 360,
        cap_large_w: int = 360,
        cap_global_d: int = 0,      # 0 means don't crop depth by cap_d
        cap_global_h: int = 0,      # 0 means don't crop H
        cap_global_w: int = 0,      # 0 means don't crop W
        cap_d_if_gt: int = 0,       # for depth crop condition; set 0 to disable condition

        # expected shapes for fail_safe zeros
        expected_small_dhw: Tuple[int, int, int] = (10, 128, 128),
        expected_large_dhw: Tuple[int, int, int] = (15, 360, 360),
        expected_global_dhw: Tuple[int, int, int] = (30, 409, 332),
    ):
        self.data_root = str(data_root)
        self.use_mask = bool(use_mask)
        self.fail_safe = bool(fail_safe)
        self.seed = int(seed)
        self.enable_augment = bool(enable_augment)

        self.max_rois_small = int(max_rois_small)
        self.allow_empty_small = bool(allow_empty_small)

        self.mask_kind_small = str(mask_kind_small)
        self.mask_kind_large = str(mask_kind_large)
        self.mask_kind_global = str(mask_kind_global)

        self.require_large = bool(require_large)
        self.require_global = bool(require_global)

        self.dynamic_pad = bool(dynamic_pad)

        self.cap_small_d = int(cap_small_d)
        self.cap_small_h = int(cap_small_h)
        self.cap_small_w = int(cap_small_w)

        self.cap_large_d = int(cap_large_d)
        self.cap_large_h = int(cap_large_h)
        self.cap_large_w = int(cap_large_w)

        self.cap_global_d = int(cap_global_d)
        self.cap_global_h = int(cap_global_h)
        self.cap_global_w = int(cap_global_w)

        self.cap_d_if_gt = int(cap_d_if_gt)

        self.expected_small_dhw = tuple(map(int, expected_small_dhw))
        self.expected_large_dhw = tuple(map(int, expected_large_dhw))
        self.expected_global_dhw = tuple(map(int, expected_global_dhw))

        self.samples: List[MultiScaleCaseSample] = []
        self._scan()
        self.indices = indices if indices is not None else list(range(len(self.samples)))

    def _scan(self):
        classes = [("dcis", 0), ("dcis+", 1)]
        small_root = os.path.join(self.data_root, "small")
        large_root = os.path.join(self.data_root, "large")
        global_root = os.path.join(self.data_root, "global")

        for cls_name, y in classes:
            small_cls = os.path.join(small_root, cls_name)
            large_cls = os.path.join(large_root, cls_name)
            global_cls = os.path.join(global_root, cls_name)

            if not os.path.isdir(small_cls):
                continue

            for case_id in sorted(os.listdir(small_cls)):
                case_small = os.path.join(small_cls, case_id)
                if not os.path.isdir(case_small):
                    continue

                # small rois
                roi_dirs = []
                for roi_name in sorted(os.listdir(case_small)):
                    roi_dir = os.path.join(case_small, roi_name)
                    if not os.path.isdir(roi_dir):
                        continue
                    if not roi_name.lower().startswith("roi_"):
                        continue
                    if os.path.isfile(os.path.join(roi_dir, "image.npy")):
                        roi_dirs.append(roi_dir)

                if (not roi_dirs) and (not self.allow_empty_small):
                    continue

                # large/global case dirs
                case_large = os.path.join(large_cls, case_id)
                case_global = os.path.join(global_cls, case_id)

                if self.require_large and (not os.path.isfile(os.path.join(case_large, "image.npy"))):
                    continue
                if self.require_global and (not os.path.isfile(os.path.join(case_global, "image.npy"))):
                    continue

                self.samples.append(
                    MultiScaleCaseSample(
                        small_roi_dirs=roi_dirs,
                        large_dir=case_large,
                        global_dir=case_global,
                        label=y,
                        case_id=case_id,
                        cls_name=cls_name,
                    )
                )

    def __len__(self):
        return len(self.indices)

    def _load_from_dir(
        self,
        dir_path: str,
        mask_kind: str,
        cap_d: int,
        cap_h: int,
        cap_w: int,
        expected_dhw: Tuple[int, int, int],
    ) -> torch.Tensor:
        vol_path = os.path.join(dir_path, "image.npy")
        mask_path = _pick_mask_path(dir_path, mask_kind) if self.use_mask else ""
        return _load_one_volume_with_optional_mask(
            vol_path=vol_path,
            mask_path=mask_path,
            use_mask=self.use_mask,
            dynamic_pad=self.dynamic_pad,
            cap_h=cap_h,
            cap_w=cap_w,
            cap_d=cap_d,
            cap_d_if_gt=self.cap_d_if_gt if self.cap_d_if_gt > 0 else cap_d,
            fail_safe=self.fail_safe,
            expected_dhw=expected_dhw,
        )

    def __getitem__(self, idx):
        s = self.samples[self.indices[idx]]

        meta = {
            "case_id": s.case_id,
            "cls_name": s.cls_name,
            "small_case_dir": os.path.dirname(s.small_roi_dirs[0]) if s.small_roi_dirs else "",
            "large_case_dir": s.large_dir,
            "global_case_dir": s.global_dir,
        }

        # small list (MIL bag)
        small_dirs = s.small_roi_dirs[: self.max_rois_small]
        small_list: List[torch.Tensor] = []
        for rd in small_dirs:
            small_list.append(
                self._load_from_dir(
                    dir_path=rd,
                    mask_kind=self.mask_kind_small,
                    cap_d=self.cap_small_d,
                    cap_h=self.cap_small_h,
                    cap_w=self.cap_small_w,
                    expected_dhw=self.expected_small_dhw,
                )
            )

        if (not small_list) and self.allow_empty_small:
            # represent empty bag as one dummy roi for stability
            c = 2 if self.use_mask else 1
            d, h, w = self.expected_small_dhw
            small_list = [torch.zeros((c, d, h, w), dtype=torch.float32)]

        # large
        x_large = self._load_from_dir(
            dir_path=s.large_dir,
            mask_kind=self.mask_kind_large,
            cap_d=self.cap_large_d,
            cap_h=self.cap_large_h,
            cap_w=self.cap_large_w,
            expected_dhw=self.expected_large_dhw,
        )

        # global
        x_global = self._load_from_dir(
            dir_path=s.global_dir,
            mask_kind=self.mask_kind_global,
            cap_d=self.cap_global_d,
            cap_h=self.cap_global_h,
            cap_w=self.cap_global_w,
            expected_dhw=self.expected_global_dhw,
        )

        y = torch.tensor([float(s.label)], dtype=torch.float32)
        return small_list, x_large, x_global, y, meta


# ----------------------------
# collate functions
# ----------------------------
def collate_keep_meta(batch):
    """
    For non-MIL single-tensor batches:
      return x[B,C,D,H,W], y[B], metas[list]
    """
    xs, ys, metas = [], [], []
    for x, y, m in batch:
        ys.append(float(y.item() if torch.is_tensor(y) else y))
        xs.append(x.float())
        metas.append(m or {})
    x_stack = torch.stack(xs, dim=0)
    y_stack = torch.tensor(ys, dtype=torch.float32)
    return x_stack, y_stack, metas


def collate_mil_dynamic_pad(batch, pad_min_d=7, pad_min_h=96, pad_min_w=96):
    """
    For legacy MIL:
      input: (x_list, y, meta)
      output:
        x_flat [N,C,D,H,W], y_case [B], metas, splits
    """
    flat: List[torch.Tensor] = []
    splits: List[int] = []
    ys: List[float] = []
    metas: List[dict] = []

    for x_list, y, m in batch:
        ys.append(float(y.item() if torch.is_tensor(y) else y))
        metas.append(m or {})
        splits.append(len(x_list))
        for x in x_list:
            flat.append(x.float())

    if not flat:
        # dummy
        x_flat = torch.zeros((1, 1, pad_min_d, pad_min_h, pad_min_w), dtype=torch.float32)
        y_case = torch.tensor(ys, dtype=torch.float32)
        return x_flat, y_case, metas, splits

    max_d = max(int(t.shape[1]) for t in flat)
    max_h = max(int(t.shape[2]) for t in flat)
    max_w = max(int(t.shape[3]) for t in flat)

    max_d = max(max_d, pad_min_d)
    max_h = max(max_h, pad_min_h)
    max_w = max(max_w, pad_min_w)

    padded = []
    for t in flat:
        _, d, h, w = t.shape
        pd = max_d - d
        ph = max_h - h
        pw = max_w - w
        t2 = F.pad(t, (0, pw, 0, ph, 0, pd), value=0.0)
        padded.append(t2)

    x_flat = torch.stack(padded, dim=0)  # [N,C,D,H,W]
    y_case = torch.tensor(ys, dtype=torch.float32)
    return x_flat, y_case, metas, splits


def collate_multiscale_mil_dynamic_pad(
    batch,
    pad_min_small: Tuple[int, int, int] = (7, 96, 96),
    pad_min_large: Tuple[int, int, int] = (7, 360, 360),
    pad_min_global: Tuple[int, int, int] = (7, 384, 384),
):
    """
    For multiscale:
      input: (small_list, x_large, x_global, y, meta)

      output:
        x_small_flat [N,C,D,H,W]
        x_large      [B,C,D,H,W]
        x_global     [B,C,D,H,W]
        y_case       [B]
        metas        list
        splits       list[int]  (#small rois per case)
    """
    small_flat: List[torch.Tensor] = []
    splits: List[int] = []
    ys: List[float] = []
    metas: List[dict] = []
    large_list: List[torch.Tensor] = []
    global_list: List[torch.Tensor] = []

    for small_list, x_large, x_global, y, m in batch:
        ys.append(float(y.item() if torch.is_tensor(y) else y))
        metas.append(m or {})
        splits.append(len(small_list))
        for x in small_list:
            small_flat.append(x.float())
        large_list.append(x_large.float())
        global_list.append(x_global.float())

    # --- pad small flat ---
    min_sd, min_sh, min_sw = map(int, pad_min_small)
    if not small_flat:
        # dummy one roi
        x_small_flat = torch.zeros((1, 1, min_sd, min_sh, min_sw), dtype=torch.float32)
    else:
        max_d = max(int(t.shape[1]) for t in small_flat)
        max_h = max(int(t.shape[2]) for t in small_flat)
        max_w = max(int(t.shape[3]) for t in small_flat)
        max_d = max(max_d, min_sd)
        max_h = max(max_h, min_sh)
        max_w = max(max_w, min_sw)

        padded = []
        for t in small_flat:
            _, d, h, w = t.shape
            t2 = F.pad(t, (0, max_w - w, 0, max_h - h, 0, max_d - d), value=0.0)
            padded.append(t2)
        x_small_flat = torch.stack(padded, dim=0)  # [N,C,D,H,W]

    # --- pad large ---
    min_ld, min_lh, min_lw = map(int, pad_min_large)
    max_ld = max([int(t.shape[1]) for t in large_list] + [min_ld])
    max_lh = max([int(t.shape[2]) for t in large_list] + [min_lh])
    max_lw = max([int(t.shape[3]) for t in large_list] + [min_lw])
    padded_large = []
    for t in large_list:
        _, d, h, w = t.shape
        t2 = F.pad(t, (0, max_lw - w, 0, max_lh - h, 0, max_ld - d), value=0.0)
        padded_large.append(t2)
    x_large = torch.stack(padded_large, dim=0)  # [B,C,D,H,W]

    # --- pad global ---
    min_gd, min_gh, min_gw = map(int, pad_min_global)
    max_gd = max([int(t.shape[1]) for t in global_list] + [min_gd])
    max_gh = max([int(t.shape[2]) for t in global_list] + [min_gh])
    max_gw = max([int(t.shape[3]) for t in global_list] + [min_gw])
    padded_global = []
    for t in global_list:
        _, d, h, w = t.shape
        t2 = F.pad(t, (0, max_gw - w, 0, max_gh - h, 0, max_gd - d), value=0.0)
        padded_global.append(t2)
    x_global = torch.stack(padded_global, dim=0)  # [B,C,D,H,W]

    y_case = torch.tensor(ys, dtype=torch.float32)
    return x_small_flat, x_large, x_global, y_case, metas, splits
