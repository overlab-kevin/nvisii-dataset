from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class _FrozenList:
    items: tuple

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

    def to_list(self) -> List[str]:
        return list(self.items)


class PartRegistry:
    """
    Single source of truth for part ordering and conversions.

    - Constructed from a canonical ordered list of part names (dataset phase list).
    - Provides conversions between names, indices, and binary/multi-hot masks (numpy and torch).
    - Offers helpers to interpret part logits and refine masks.

    This class is intentionally strict: names must be unique and non-empty; all
    conversions validate shapes and contents to avoid silent misalignment.
    """

    def __init__(self, names: Iterable[str]):
        names_list = [str(n) for n in names]
        if any(n == "" for n in names_list):
            raise ValueError("PartRegistry: empty part names are not allowed")
        # Enforce uniqueness while preserving order
        seen = set()
        uniq_names: List[str] = []
        for n in names_list:
            if n in seen:
                raise ValueError(f"PartRegistry: duplicate part name detected: '{n}'")
            seen.add(n)
            uniq_names.append(n)

        object.__setattr__(self, "_names", _FrozenList(tuple(uniq_names)))
        object.__setattr__(self, "_index", {n: i for i, n in enumerate(uniq_names)})

    @classmethod
    def from_names(cls, names: Iterable[str]) -> "PartRegistry":
        return cls(names)

    # ------------- basic info -------------
    @property
    def names(self) -> List[str]:
        return self._names.to_list()

    def size(self) -> int:
        return len(self._names)

    def has(self, name: str) -> bool:
        return name in self._index

    # ------------- index conversions -------------
    def to_index(self, name: str) -> int:
        try:
            return self._index[name]
        except KeyError as e:
            raise KeyError(f"Unknown part name '{name}'") from e

    def to_indices(self, names: Iterable[str]) -> np.ndarray:
        idx = [self.to_index(n) for n in names]
        return np.asarray(idx, dtype=np.int64)

    def names_from_indices(self, indices: Iterable[int]) -> List[str]:
        out: List[str] = []
        P = self.size()
        for i in indices:
            ii = int(i)
            if ii < 0 or ii >= P:
                raise IndexError(f"Part index {ii} out of range [0, {P})")
            out.append(self._names[ii])
        return out

    # ------------- mask conversions (numpy) -------------
    def mask_from_names_np(self, names: Iterable[str], *, dtype=np.float32) -> np.ndarray:
        P = self.size()
        mask = np.zeros((P,), dtype=dtype)
        for n in names:
            if not self.has(n):
                # strict
                raise KeyError(f"Unknown part name '{n}' in mask_from_names_np")
            mask[self._index[n]] = 1.0
        return mask

    def names_from_mask_np(self, mask: np.ndarray, *, threshold: float = 0.5) -> List[str]:
        if mask.ndim != 1 or mask.shape[0] != self.size():
            raise ValueError(f"names_from_mask_np expects shape ({self.size()},), got {tuple(mask.shape)}")
        present_idx = np.nonzero(mask >= threshold)[0].tolist()
        return [self._names[i] for i in present_idx]

    # ------------- mask conversions (torch) -------------
    def mask_from_names_t(
        self,
        names: Iterable[str],
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        P = self.size()
        mask = torch.zeros((P,), device=device, dtype=dtype)
        for n in names:
            if not self.has(n):
                raise KeyError(f"Unknown part name '{n}' in mask_from_names_t")
            mask[self._index[n]] = 1.0
        return mask

    def names_from_mask_t(self, mask: torch.Tensor, *, threshold: float = 0.5) -> List[str]:
        if mask.ndim != 1 or mask.shape[0] != self.size():
            raise ValueError(f"names_from_mask_t expects shape ({self.size()},), got {tuple(mask.shape)}")
        present_idx = (mask >= threshold).nonzero(as_tuple=False).flatten().tolist()
        return [self._names[i] for i in present_idx]

    # ------------- logits/mask helpers (torch) -------------
    @staticmethod
    def logits_to_offsets(part_logits: torch.Tensor) -> torch.Tensor:
        """
        Map logits (B,P,3) to offsets in {-1,0,+1} via argmax and center at 0.
        """
        if part_logits.ndim != 3 or part_logits.shape[-1] != 3:
            raise ValueError(f"logits_to_offsets expects (B,P,3), got {tuple(part_logits.shape)}")
        cls = part_logits.argmax(dim=-1)            # (B,P) in {0,1,2}
        offsets = cls.to(torch.int64) - 1           # {-1,0,1}
        return offsets

    @staticmethod
    def refine_mask(initial_mask: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        """
        Apply {-1,0,+1} offsets to a binary mask and clamp to [0,1].
        Shapes:
            initial_mask: (B,P) or (P,)
            offsets:      (B,P) or (P,)
        """
        if initial_mask.shape != offsets.shape:
            # broadcast attempt with explicit error if incompatible
            try:
                refined = initial_mask + offsets
            except Exception as e:
                raise ValueError(f"refine_mask: shape mismatch {tuple(initial_mask.shape)} vs {tuple(offsets.shape)}") from e
        else:
            refined = initial_mask + offsets
        return torch.clamp(refined, 0, 1)

    # ------------- validation -------------
    def assert_conf_length(self, conf: Sequence[float] | np.ndarray | torch.Tensor, *, what: str = "assm_conf") -> None:
        length = len(conf) if not isinstance(conf, torch.Tensor) else int(conf.shape[0])
        if length != self.size():
            raise ValueError(f"{what} has length {length}, expected {self.size()} (registry order mismatch)")
