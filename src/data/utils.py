from typing import Optional, Tuple

import polars as pl
import torch
from torch import Tensor
from torch.nn import functional as F


def _as_batched_lengths(lengths: Tensor) -> Tensor:
    """Ensure a length tensor has a batch dimension."""
    if lengths.dim() == 1:
        return lengths.unsqueeze(0)
    if lengths.dim() != 2:
        raise ValueError(f"lengths must be 1D or 2D, got shape: {tuple(lengths.shape)}")
    return lengths


def downsample_lengths(lengths: Tensor, stride: int) -> Tensor:
    """Convert segment lengths to downsampled frame lengths."""
    lengths = _as_batched_lengths(lengths.to(dtype=torch.long))
    sid = torch.cumsum(F.pad(lengths[:, :-1], (1, 0), value=0), dim=1)
    eid = sid + lengths - 1
    new_sid = (sid + stride - 1) // stride
    new_eid = eid // stride
    return torch.clamp(new_eid - new_sid + 1, min=0)


def lengths_to_sid_eid(lengths: Tensor) -> Tuple[Tensor, Tensor]:
    """Convert segment lengths to start and end frame indices."""
    lengths = _as_batched_lengths(lengths.to(dtype=torch.long))
    sid = torch.cumsum(F.pad(lengths[:, :-1], (1, 0), value=0), dim=1)
    eid = sid + lengths - 1
    return sid, eid


def lengths_to_seq_idx(lengths: Tensor, seq_len: Optional[int] = None) -> Tensor:
    """Expand segment lengths into per-frame segment indices."""
    lengths = _as_batched_lengths(lengths.to(dtype=torch.long))
    rows = []
    for b in range(lengths.shape[0]):
        seg_ids = torch.arange(lengths.shape[1], device=lengths.device, dtype=torch.int32)
        row = torch.repeat_interleave(seg_ids, lengths[b])
        if seq_len is not None:
            if row.numel() > seq_len:
                raise ValueError(f"Expanded seq_idx length {row.numel()} exceeds seq_len {seq_len}")
            if row.numel() < seq_len:
                pad = torch.full((seq_len - row.numel(),), -1, device=lengths.device, dtype=torch.int32)
                row = torch.cat([row, pad], dim=0)
        rows.append(row)
    return torch.stack(rows, dim=0)


def time_to_idx(time_sec: Tuple[float, float], sample_rate: int = 16000, downsample_factor: int = 320) -> Tensor:
    """Convert a time span in seconds to downsampled frame indices."""
    start_sec, end_sec = time_sec

    if start_sec < 0 or end_sec < 0:
        raise ValueError(f'start_sec and end_sec must be non-negative, got: {time_sec}')
    if end_sec < start_sec:
        raise ValueError(f'end_sec must be greater than or equal to start_sec, got: {time_sec}')

    start_idx = int(start_sec * sample_rate / downsample_factor)
    end_idx = int(end_sec * sample_rate / downsample_factor)

    return torch.tensor([start_idx, end_idx], dtype=torch.int32)


def to_frame_idx_list(expr: pl.Expr, sample_rate: int = 16000, downsample_factor: int = 320) -> pl.Expr:
    """Build a Polars expression that converts time lists to frame indices."""
    return expr.list.eval(
        ((pl.element() * sample_rate / downsample_factor).floor().cast(pl.Int64))
    )
