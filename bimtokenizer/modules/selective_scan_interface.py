"""Selective Scan Interface for BiMamba.

CUDA-accelerated operations for Mamba1-based bidirectional SSM.
Provides fused forward/backward autograd functions and high-level BiMamba wrappers.

Based on: https://github.com/hustvl/Vim
Updated for mamba-ssm >= 2.2.5 and causal-conv1d >= 1.4
"""

import torch
import torch.nn.functional as F
from typing import Optional
from torch import Tensor
from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None

try:
    from causal_conv1d.cpp_functions import (
        causal_conv1d_fwd_function,
        causal_conv1d_bwd_function,
        causal_conv1d_update_function,
    )
except ImportError:
    causal_conv1d_fwd_function = None
    causal_conv1d_bwd_function = None
    causal_conv1d_update_function = None

try:
    from mamba_ssm.utils.torch import custom_bwd, custom_fwd
except ImportError:
    from torch.cuda.amp import custom_bwd, custom_fwd

from mamba_ssm.ops.triton.layer_norm import _layer_norm_fwd

import selective_scan_cuda


# ============================================================================
# Utilities
# ============================================================================

def build_valid_time_flip_indices(
    lengths: Tensor,
    seq_len: int,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Build reusable ``(B, seq_len)`` indices that reverse valid time only."""
    if device is None:
        device = lengths.device
    lengths = lengths.to(device=device, dtype=torch.long)
    arange = torch.arange(seq_len, device=device).unsqueeze(0).expand(lengths.shape[0], -1)
    lengths_expanded = lengths.unsqueeze(1)
    return torch.where(
        arange < lengths_expanded,
        lengths_expanded - 1 - arange,
        arange,
    )


def flip_valid_time(
    x: Tensor,
    lengths: Optional[Tensor] = None,
    channel_first: bool = False,
    flip_indices: Optional[Tensor] = None,
) -> Tensor:
    """Flip only valid time steps; padding stays at the end.

    Without lengths, falls back to full time-dimension flip (legacy behavior).

    Args:
        x: (B, L, D) if channel_first=False, or (B, D, L) if channel_first=True
        lengths: (B,) valid lengths along the time dimension; None = full flip
        flip_indices: Optional precomputed indices from
            :func:`build_valid_time_flip_indices`.
    """
    time_dim = 2 if channel_first else 1
    if lengths is None and flip_indices is None:
        return x.flip([time_dim])

    if flip_indices is not None:
        idx = flip_indices.to(device=x.device, dtype=torch.long)
    else:
        lengths = lengths.to(device=x.device, dtype=torch.long)
    if channel_first:
        b, d, seq = x.shape
        if flip_indices is None:
            idx = build_valid_time_flip_indices(lengths, seq, x.device)
        return x.gather(2, idx.unsqueeze(1).expand(-1, d, -1))

    b, seq, d = x.shape
    if flip_indices is None:
        idx = build_valid_time_flip_indices(lengths, seq, x.device)
    return x.gather(1, idx.unsqueeze(-1).expand(-1, -1, d))


def rms_norm_forward(x, weight, bias, eps=1e-6, is_rms_norm=True):
    if x.stride(-1) != 1:
        x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    return _layer_norm_fwd(
        x, weight, bias, eps, None, residual_dtype=None, is_rms_norm=is_rms_norm
    )[0]


# ============================================================================
# Low-level: Selective Scan (autograd wrapper around selective_scan_cuda)
# ============================================================================

class SelectiveScanFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, u, delta, A, B, C, D=None, z=None, delta_bias=None,
                delta_softplus=False, return_last_state=False):
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if z is not None and z.stride(-1) != 1:
            z = z.contiguous()
        if B.dim() == 3:
            B = rearrange(B, "b dstate l -> b 1 dstate l")
            ctx.squeeze_B = True
        if C.dim() == 3:
            C = rearrange(C, "b dstate l -> b 1 dstate l")
            ctx.squeeze_C = True
        out, x, *rest = selective_scan_cuda.fwd(
            u, delta, A, B, C, D, z, delta_bias, delta_softplus
        )
        ctx.delta_softplus = delta_softplus
        ctx.has_z = z is not None
        last_state = x[:, :, -1, 1::2]
        if not ctx.has_z:
            ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
            return out if not return_last_state else (out, last_state)
        else:
            ctx.save_for_backward(u, delta, A, B, C, D, z, delta_bias, x, out)
            out_z = rest[0]
            return out_z if not return_last_state else (out_z, last_state)

    @staticmethod
    def backward(ctx, dout, *args):
        if not ctx.has_z:
            u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
            z = None
            out = None
        else:
            u, delta, A, B, C, D, z, delta_bias, x, out = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
            u, delta, A, B, C, D, z, delta_bias, dout, x, out, None,
            ctx.delta_softplus, False
        )
        dz = rest[0] if ctx.has_z else None
        dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
        dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
        return (
            du, ddelta, dA, dB, dC,
            dD if D is not None else None,
            dz,
            ddelta_bias if delta_bias is not None else None,
            None, None,
        )


def selective_scan_fn(u, delta, A, B, C, D=None, z=None, delta_bias=None,
                      delta_softplus=False, return_last_state=False):
    """Wrapper for SelectiveScanFn.apply.
    If return_last_state is True, returns (out, last_state) where
    last_state has shape (batch, dim, dstate).
    """
    return SelectiveScanFn.apply(
        u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state
    )


def selective_scan_ref(u, delta, A, B, C, D=None, z=None, delta_bias=None,
                       delta_softplus=False, return_last_state=False):
    """Pure PyTorch reference implementation for selective scan (no CUDA)."""
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3
    if A.is_complex():
        if is_variable_B:
            B = torch.view_as_complex(
                rearrange(B.float(), "... (L two) -> ... L two", two=2)
            )
        if is_variable_C:
            C = torch.view_as_complex(
                rearrange(C.float(), "... (L two) -> ... L two", two=2)
            )
    else:
        B = B.float()
        C = C.float()
    x = A.new_zeros((batch, dim, dstate))
    ys = []
    deltaA = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))
    if not is_variable_B:
        deltaB_u = torch.einsum("bdl,dn,bdl->bdln", delta, B, u)
    else:
        if B.dim() == 3:
            deltaB_u = torch.einsum("bdl,bnl,bdl->bdln", delta, B, u)
        else:
            B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
            deltaB_u = torch.einsum("bdl,bdnl,bdl->bdln", delta, B, u)
    if is_variable_C and C.dim() == 4:
        C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
    last_state = None
    for i in range(u.shape[2]):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        if not is_variable_C:
            y = torch.einsum("bdn,dn->bd", x, C)
        else:
            if C.dim() == 3:
                y = torch.einsum("bdn,bn->bd", x, C[:, :, i])
            else:
                y = torch.einsum("bdn,bdn->bd", x, C[:, :, :, i])
        if i == u.shape[2] - 1:
            last_state = x
        if y.is_complex():
            y = y.real * 2
        ys.append(y)
    y = torch.stack(ys, dim=2)
    out = y if D is None else y + u * rearrange(D, "d -> d 1")
    if z is not None:
        out = out * F.silu(z)
    out = out.to(dtype=dtype_in)
    return out if not return_last_state else (out, last_state)


# ============================================================================
# Mid-level: Fused Mamba Inner Functions (conv1d + SSM scan in one autograd op)
# ============================================================================

class MambaInnerFnNoOutProj(torch.autograd.Function):
    """Fused Mamba1 forward without out_proj: in_proj split → causal_conv1d → SSM scan.
    Returns pre-out_proj activation of shape (batch, d_inner, seqlen).
    Used as a building block for bidirectional variants (v2/v3).
    """

    @staticmethod
    @custom_fwd
    def forward(ctx, xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
                A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
                C_proj_bias=None, delta_softplus=True, checkpoint_lvl=1):
        assert causal_conv1d_fwd_function is not None, \
            "causal_conv1d is not available. Please install causal-conv1d."
        assert checkpoint_lvl in [0, 1]
        L = xz.shape[-1]
        delta_rank = delta_proj_weight.shape[1]
        d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
        if torch.is_autocast_enabled():
            x_proj_weight = x_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
            delta_proj_weight = delta_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
        if xz.stride(-1) != 1:
            xz = xz.contiguous()
        conv1d_weight = rearrange(conv1d_weight, "d 1 w -> d w")
        x, z = xz.chunk(2, dim=1)
        conv1d_bias = conv1d_bias.contiguous() if conv1d_bias is not None else None
        conv1d_out = causal_conv1d_fwd_function(
            x, conv1d_weight, conv1d_bias, None, None, None, True
        )
        x_dbl = F.linear(rearrange(conv1d_out, "b d l -> (b l) d"), x_proj_weight)
        delta = rearrange(
            delta_proj_weight @ x_dbl[:, :delta_rank].t(), "d (b l) -> b d l", l=L
        )
        ctx.is_variable_B = B is None
        ctx.is_variable_C = C is None
        ctx.B_proj_bias_is_None = B_proj_bias is None
        ctx.C_proj_bias_is_None = C_proj_bias is None
        if B is None:
            B = x_dbl[:, delta_rank:delta_rank + d_state]
            if B_proj_bias is not None:
                B = B + B_proj_bias.to(dtype=B.dtype)
            if not A.is_complex():
                B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
            else:
                B = rearrange(B, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
        else:
            if B.stride(-1) != 1:
                B = B.contiguous()
        if C is None:
            C = x_dbl[:, -d_state:]
            if C_proj_bias is not None:
                C = C + C_proj_bias.to(dtype=C.dtype)
            if not A.is_complex():
                C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
            else:
                C = rearrange(C, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
        else:
            if C.stride(-1) != 1:
                C = C.contiguous()
        if D is not None:
            D = D.contiguous()
        out, scan_intermediates, out_z = selective_scan_cuda.fwd(
            conv1d_out, delta, A, B, C, D, z, delta_bias, delta_softplus
        )
        ctx.delta_softplus = delta_softplus
        ctx.checkpoint_lvl = checkpoint_lvl
        if checkpoint_lvl >= 1:
            conv1d_out, delta = None, None
        ctx.save_for_backward(
            xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight,
            delta_proj_weight, conv1d_out, delta,
            A, B, C, D, delta_bias, scan_intermediates, out,
        )
        return out_z

    @staticmethod
    @custom_bwd
    def backward(ctx, dout):
        (xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight, delta_proj_weight,
         conv1d_out, delta, A, B, C, D, delta_bias, scan_intermediates, out,
        ) = ctx.saved_tensors
        L = xz.shape[-1]
        delta_rank = delta_proj_weight.shape[1]
        d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
        x, z = xz.chunk(2, dim=1)
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        if ctx.checkpoint_lvl == 1:
            conv1d_out = causal_conv1d_fwd_function(
                x, conv1d_weight, conv1d_bias, None, None, None, True
            )
            delta = rearrange(
                delta_proj_weight @ x_dbl[:, :delta_rank].t(),
                "d (b l) -> b d l", l=L,
            )
        dxz = torch.empty_like(xz)
        dx, dz = dxz.chunk(2, dim=1)
        dconv1d_out, ddelta, dA, dB, dC, dD, ddelta_bias, dz, out_z = selective_scan_cuda.bwd(
            conv1d_out, delta, A, B, C, D, z, delta_bias, dout, scan_intermediates,
            out, dz, ctx.delta_softplus, True,
        )
        dD = dD if D is not None else None
        dx_dbl = torch.empty_like(x_dbl)
        dB_proj_bias = None
        if ctx.is_variable_B:
            if not A.is_complex():
                dB = rearrange(dB, "b 1 dstate l -> (b l) dstate").contiguous()
            else:
                dB = rearrange(dB, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
            dB_proj_bias = dB.sum(0) if not ctx.B_proj_bias_is_None else None
            dx_dbl[:, delta_rank:delta_rank + d_state] = dB
            dB = None
        dC_proj_bias = None
        if ctx.is_variable_C:
            if not A.is_complex():
                dC = rearrange(dC, "b 1 dstate l -> (b l) dstate").contiguous()
            else:
                dC = rearrange(dC, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
            dC_proj_bias = dC.sum(0) if not ctx.C_proj_bias_is_None else None
            dx_dbl[:, -d_state:] = dC
            dC = None
        ddelta = rearrange(ddelta, "b d l -> d (b l)")
        ddelta_proj_weight = torch.einsum("dB,Br->dr", ddelta, x_dbl[:, :delta_rank])
        dx_dbl[:, :delta_rank] = torch.einsum("dB,dr->Br", ddelta, delta_proj_weight)
        dconv1d_out = rearrange(dconv1d_out, "b d l -> d (b l)")
        dx_proj_weight = torch.einsum(
            "Br,Bd->rd", dx_dbl, rearrange(conv1d_out, "b d l -> (b l) d")
        )
        dconv1d_out = torch.addmm(
            dconv1d_out, x_proj_weight.t(), dx_dbl.t(), out=dconv1d_out
        )
        dconv1d_out = rearrange(
            dconv1d_out, "d (b l) -> b d l", b=x.shape[0], l=x.shape[-1]
        )
        dx, dconv1d_weight, dconv1d_bias, *_ = causal_conv1d_bwd_function(
            x, conv1d_weight, conv1d_bias, dconv1d_out, None, None, None, dx, False, True
        )
        dconv1d_bias = dconv1d_bias if conv1d_bias is not None else None
        dconv1d_weight = rearrange(dconv1d_weight, "d w -> d 1 w")
        return (
            dxz, dconv1d_weight, dconv1d_bias, dx_proj_weight, ddelta_proj_weight,
            dA, dB, dC, dD,
            ddelta_bias if delta_bias is not None else None,
            dB_proj_bias, dC_proj_bias, None, None,
        )


class MambaInnerFn(torch.autograd.Function):
    """Fused Mamba1 forward with out_proj and optional B/C/dt RMSNorm."""

    @staticmethod
    @custom_fwd
    def forward(ctx, xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
                out_proj_weight, out_proj_bias,
                A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
                C_proj_bias=None, delta_softplus=True, checkpoint_lvl=1,
                b_rms_weight=None, c_rms_weight=None, dt_rms_weight=None,
                b_c_dt_rms_eps=1e-6):
        assert causal_conv1d_fwd_function is not None, \
            "causal_conv1d is not available. Please install causal-conv1d."
        assert checkpoint_lvl in [0, 1]
        L = xz.shape[-1]
        delta_rank = delta_proj_weight.shape[1]
        d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
        if torch.is_autocast_enabled():
            x_proj_weight = x_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
            delta_proj_weight = delta_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
            out_proj_weight = out_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
            out_proj_bias = (
                out_proj_bias.to(dtype=torch.get_autocast_gpu_dtype())
                if out_proj_bias is not None else None
            )
        if xz.stride(-1) != 1:
            xz = xz.contiguous()
        conv1d_weight = rearrange(conv1d_weight, "d 1 w -> d w")
        x, z = xz.chunk(2, dim=1)
        conv1d_bias = conv1d_bias.contiguous() if conv1d_bias is not None else None
        conv1d_out = causal_conv1d_fwd_function(
            x, conv1d_weight, conv1d_bias, None, None, None, True
        )
        x_dbl = F.linear(rearrange(conv1d_out, "b d l -> (b l) d"), x_proj_weight)
        delta = rearrange(
            delta_proj_weight @ x_dbl[:, :delta_rank].t(), "d (b l) -> b d l", l=L
        )
        ctx.is_variable_B = B is None
        ctx.is_variable_C = C is None
        ctx.B_proj_bias_is_None = B_proj_bias is None
        ctx.C_proj_bias_is_None = C_proj_bias is None
        if B is None:
            B = x_dbl[:, delta_rank:delta_rank + d_state]
            if B_proj_bias is not None:
                B = B + B_proj_bias.to(dtype=B.dtype)
            if not A.is_complex():
                B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
            else:
                B = rearrange(B, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
        else:
            if B.stride(-1) != 1:
                B = B.contiguous()
        if C is None:
            C = x_dbl[:, -d_state:]
            if C_proj_bias is not None:
                C = C + C_proj_bias.to(dtype=C.dtype)
            if not A.is_complex():
                C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
            else:
                C = rearrange(C, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
        else:
            if C.stride(-1) != 1:
                C = C.contiguous()
        if D is not None:
            D = D.contiguous()
        if b_rms_weight is not None:
            B = rearrange(B, "b 1 dstate l -> (b l) dstate", l=L).contiguous()
            B = rms_norm_forward(B, b_rms_weight, bias=None, eps=b_c_dt_rms_eps)
            B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
        if c_rms_weight is not None:
            C = rearrange(C, "b 1 dstate l -> (b l) dstate", l=L).contiguous()
            C = rms_norm_forward(C, c_rms_weight, bias=None, eps=b_c_dt_rms_eps)
            C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
        if dt_rms_weight is not None:
            delta = rearrange(delta, "b d l -> (b l) d", l=L).contiguous()
            delta = rms_norm_forward(delta, dt_rms_weight, bias=None, eps=b_c_dt_rms_eps)
            delta = rearrange(delta, "(b l) d -> b d l", l=L).contiguous()
        out, scan_intermediates, out_z = selective_scan_cuda.fwd(
            conv1d_out, delta, A, B, C, D, z, delta_bias, delta_softplus
        )
        ctx.delta_softplus = delta_softplus
        ctx.out_proj_bias_is_None = out_proj_bias is None
        ctx.checkpoint_lvl = checkpoint_lvl
        ctx.b_rms_weight = b_rms_weight
        ctx.c_rms_weight = c_rms_weight
        ctx.dt_rms_weight = dt_rms_weight
        ctx.b_c_dt_rms_eps = b_c_dt_rms_eps
        if checkpoint_lvl >= 1:
            conv1d_out, delta = None, None
        ctx.save_for_backward(
            xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight,
            delta_proj_weight, out_proj_weight, conv1d_out, delta,
            A, B, C, D, delta_bias, scan_intermediates,
            b_rms_weight, c_rms_weight, dt_rms_weight, out,
        )
        return F.linear(rearrange(out_z, "b d l -> b l d"), out_proj_weight, out_proj_bias)

    @staticmethod
    @custom_bwd
    def backward(ctx, dout):
        (xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight, delta_proj_weight,
         out_proj_weight, conv1d_out, delta, A, B, C, D, delta_bias,
         scan_intermediates, b_rms_weight, c_rms_weight, dt_rms_weight, out,
        ) = ctx.saved_tensors
        L = xz.shape[-1]
        delta_rank = delta_proj_weight.shape[1]
        d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
        x, z = xz.chunk(2, dim=1)
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        if ctx.checkpoint_lvl == 1:
            conv1d_out = causal_conv1d_fwd_function(
                x, conv1d_weight, conv1d_bias, None, None, None, True
            )
            delta = rearrange(
                delta_proj_weight @ x_dbl[:, :delta_rank].t(),
                "d (b l) -> b d l", l=L,
            )
            if dt_rms_weight is not None:
                delta = rearrange(delta, "b d l -> (b l) d", l=L).contiguous()
                delta = rms_norm_forward(delta, ctx.dt_rms_weight, None, ctx.b_c_dt_rms_eps)
                delta = rearrange(delta, "(b l) d -> b d l", l=L).contiguous()
            if b_rms_weight is not None:
                B = rearrange(B, "b 1 dstate l -> (b l) dstate", l=L).contiguous()
                B = rms_norm_forward(B, ctx.b_rms_weight, None, ctx.b_c_dt_rms_eps)
                B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
            if c_rms_weight is not None:
                C = rearrange(C, "b 1 dstate l -> (b l) dstate", l=L).contiguous()
                C = rms_norm_forward(C, ctx.c_rms_weight, None, ctx.b_c_dt_rms_eps)
                C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
        dxz = torch.empty_like(xz)
        dx, dz = dxz.chunk(2, dim=1)
        dout = rearrange(dout, "b l e -> e (b l)")
        dout_y = rearrange(out_proj_weight.t() @ dout, "d (b l) -> b d l", l=L)
        dconv1d_out, ddelta, dA, dB, dC, dD, ddelta_bias, dz, out_z = selective_scan_cuda.bwd(
            conv1d_out, delta, A, B, C, D, z, delta_bias, dout_y,
            scan_intermediates, out, dz, ctx.delta_softplus, True,
        )
        dout_proj_weight = torch.einsum(
            "eB,dB->ed", dout, rearrange(out_z, "b d l -> d (b l)")
        )
        dout_proj_bias = dout.sum(dim=(0, 1)) if not ctx.out_proj_bias_is_None else None
        dD = dD if D is not None else None
        dx_dbl = torch.empty_like(x_dbl)
        dB_proj_bias = None
        if ctx.is_variable_B:
            if not A.is_complex():
                dB = rearrange(dB, "b 1 dstate l -> (b l) dstate").contiguous()
            else:
                dB = rearrange(dB, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
            dB_proj_bias = dB.sum(0) if not ctx.B_proj_bias_is_None else None
            dx_dbl[:, delta_rank:delta_rank + d_state] = dB
            dB = None
        dC_proj_bias = None
        if ctx.is_variable_C:
            if not A.is_complex():
                dC = rearrange(dC, "b 1 dstate l -> (b l) dstate").contiguous()
            else:
                dC = rearrange(dC, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
            dC_proj_bias = dC.sum(0) if not ctx.C_proj_bias_is_None else None
            dx_dbl[:, -d_state:] = dC
            dC = None
        ddelta = rearrange(ddelta, "b d l -> d (b l)")
        ddelta_proj_weight = torch.einsum("dB,Br->dr", ddelta, x_dbl[:, :delta_rank])
        dx_dbl[:, :delta_rank] = torch.einsum("dB,dr->Br", ddelta, delta_proj_weight)
        dconv1d_out = rearrange(dconv1d_out, "b d l -> d (b l)")
        dx_proj_weight = torch.einsum(
            "Br,Bd->rd", dx_dbl, rearrange(conv1d_out, "b d l -> (b l) d")
        )
        dconv1d_out = torch.addmm(
            dconv1d_out, x_proj_weight.t(), dx_dbl.t(), out=dconv1d_out
        )
        dconv1d_out = rearrange(
            dconv1d_out, "d (b l) -> b d l", b=x.shape[0], l=x.shape[-1]
        )
        dx, dconv1d_weight, dconv1d_bias, *_ = causal_conv1d_bwd_function(
            x, conv1d_weight, conv1d_bias, dconv1d_out, None, None, None, dx, False, True
        )
        dconv1d_bias = dconv1d_bias if conv1d_bias is not None else None
        dconv1d_weight = rearrange(dconv1d_weight, "d w -> d 1 w")
        return (
            dxz, dconv1d_weight, dconv1d_bias, dx_proj_weight, ddelta_proj_weight,
            dout_proj_weight, dout_proj_bias,
            dA, dB, dC, dD,
            ddelta_bias if delta_bias is not None else None,
            dB_proj_bias, dC_proj_bias, None, None, None, None, None, None,
        )


# ============================================================================
# Functional wrappers
# ============================================================================

def mamba_inner_fn_no_out_proj(
    xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
    A, B=None, C=None, D=None, delta_bias=None,
    B_proj_bias=None, C_proj_bias=None, delta_softplus=True,
):
    """Single-direction Mamba1 fused forward, returns (batch, d_inner, seqlen)."""
    return MambaInnerFnNoOutProj.apply(
        xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
        A, B, C, D, delta_bias, B_proj_bias, C_proj_bias, delta_softplus,
    )


def mamba_inner_fn(
    xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
    out_proj_weight, out_proj_bias,
    A, B=None, C=None, D=None, delta_bias=None,
    B_proj_bias=None, C_proj_bias=None, delta_softplus=True, checkpoint_lvl=1,
    b_rms_weight=None, c_rms_weight=None, dt_rms_weight=None, b_c_dt_rms_eps=1e-6,
):
    """Single-direction Mamba1 fused forward with out_proj, returns (batch, seqlen, d_model)."""
    return MambaInnerFn.apply(
        xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
        out_proj_weight, out_proj_bias,
        A, B, C, D, delta_bias, B_proj_bias, C_proj_bias, delta_softplus, checkpoint_lvl,
        b_rms_weight, c_rms_weight, dt_rms_weight, b_c_dt_rms_eps,
    )


# ============================================================================
# High-level: Bidirectional Mamba1 fused functions
# ============================================================================

def inn_bimamba_inner_fn(
    xz: Tensor,
    conv1d_weight: Tensor, conv1d_bias: Tensor,
    x_proj_weight: Tensor, dt_proj_weight: Tensor,
    A: Tensor, D: Tensor, dt_bias: Tensor,
    conv1d_b_weight: Tensor, conv1d_b_bias: Tensor,
    x_proj_b_weight: Tensor, dt_proj_b_weight: Tensor,
    A_b: Tensor, D_b: Tensor, dt_bias_b: Tensor,
    out_proj_weight: Tensor, out_proj_bias: Tensor,
    if_devide_out: bool = True,
    delta_softplus: bool = True,
    seq_lens: Optional[Tensor] = None,
) -> Tensor:
    """InnBiMamba (v2): shared in_proj/out_proj, separate conv/SSM params per direction.

    Args:
        xz: (batch, 2*d_inner, seqlen) from shared in_proj
        *_b_*: backward-direction parameters
        if_devide_out: True → average fwd/bwd, False → sum
    Returns:
        (batch, seqlen, d_model) after shared out_proj
    """
    out_f = mamba_inner_fn_no_out_proj(
        xz, conv1d_weight, conv1d_bias, x_proj_weight, dt_proj_weight,
        A, None, None, D, delta_bias=dt_bias, delta_softplus=delta_softplus,
    )
    out_b = mamba_inner_fn_no_out_proj(
        flip_valid_time(xz, seq_lens, channel_first=True),
        conv1d_b_weight, conv1d_b_bias, x_proj_b_weight, dt_proj_b_weight,
        A_b, None, None, D_b, delta_bias=dt_bias_b, delta_softplus=delta_softplus,
    )
    if if_devide_out:
        combined = 0.5 * out_f + 0.5 * flip_valid_time(out_b, seq_lens, channel_first=True)
    else:
        combined = out_f + flip_valid_time(out_b, seq_lens, channel_first=True)
    return F.linear(rearrange(combined, "b d l -> b l d"), out_proj_weight, out_proj_bias)


def ext_bimamba_inner_fn(
    hidden_states: Tensor,
    in_proj_weight: Tensor, in_proj_bias: Tensor,
    conv1d_weight: Tensor, conv1d_bias: Tensor,
    x_proj_weight: Tensor, dt_proj_weight: Tensor,
    A: Tensor, D: Tensor, dt_bias: Tensor,
    out_proj_weight: Tensor, out_proj_bias: Tensor,
    in_proj_b_weight: Tensor, in_proj_b_bias: Tensor,
    conv1d_b_weight: Tensor, conv1d_b_bias: Tensor,
    x_proj_b_weight: Tensor, dt_proj_b_weight: Tensor,
    A_b: Tensor, D_b: Tensor, dt_bias_b: Tensor,
    out_proj_b_weight: Tensor, out_proj_b_bias: Tensor,
    if_devide_out: bool = True,
    delta_softplus: bool = True,
    seq_lens: Optional[Tensor] = None,
    flip_indices: Optional[Tensor] = None,
) -> Tensor:
    """ExtBiMamba (v3): fully independent params per direction.

    Input is flipped BEFORE backward in_proj, matching Algorithm 2 / Fig.1(b).

    Args:
        hidden_states: (batch, seqlen, d_model)
    Returns:
        (batch, seqlen, d_model)
    """
    seqlen = hidden_states.shape[1]

    # Forward direction
    xz_f = rearrange(
        in_proj_weight @ rearrange(hidden_states, "b l d -> d (b l)"),
        "d (b l) -> b d l", l=seqlen,
    )
    if in_proj_bias is not None:
        xz_f = xz_f + rearrange(in_proj_bias.to(dtype=xz_f.dtype), "d -> d 1")
    out_f = mamba_inner_fn_no_out_proj(
        xz_f, conv1d_weight, conv1d_bias, x_proj_weight, dt_proj_weight,
        A, None, None, D, delta_bias=dt_bias, delta_softplus=delta_softplus,
    )
    out_f = F.linear(rearrange(out_f, "b d l -> b l d"), out_proj_weight, out_proj_bias)

    # Backward direction: flip valid region → in_proj_b → scan → out_proj_b → flip back
    hidden_flipped = flip_valid_time(
        hidden_states, seq_lens, flip_indices=flip_indices
    )
    xz_b = rearrange(
        in_proj_b_weight @ rearrange(hidden_flipped, "b l d -> d (b l)"),
        "d (b l) -> b d l", l=seqlen,
    )
    if in_proj_b_bias is not None:
        xz_b = xz_b + rearrange(in_proj_b_bias.to(dtype=xz_b.dtype), "d -> d 1")
    out_b = mamba_inner_fn_no_out_proj(
        xz_b, conv1d_b_weight, conv1d_b_bias, x_proj_b_weight, dt_proj_b_weight,
        A_b, None, None, D_b, delta_bias=dt_bias_b, delta_softplus=delta_softplus,
    )
    out_b = F.linear(rearrange(out_b, "b d l -> b l d"), out_proj_b_weight, out_proj_b_bias)
    out_b = flip_valid_time(out_b, seq_lens, flip_indices=flip_indices)

    if if_devide_out:
        return 0.5 * out_f + 0.5 * out_b
    else:
        return out_f + out_b
