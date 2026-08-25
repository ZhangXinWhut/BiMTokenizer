import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils import parametrize
from torch.nn.utils.rnn import pad_sequence
from .mamba_layers import MambaLayers
from torch.nn.functional import scaled_dot_product_attention
from . import activations
from .alias_free_torch import *
from .alias_free_torch import Activation1d
from einops import rearrange

logger = logging.getLogger(__name__)

def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)

def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))

def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))

class ResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Activation1d(activation=activations.SnakeBeta(dim, alpha_logscale=True)),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Activation1d(activation=activations.SnakeBeta(dim, alpha_logscale=True)),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        return x + self.block(x)

class AttnProj1d(nn.Module):
    """
    纯注意力投影（无 MLP），工作在 (B, C, T) 格式下。
    通过 multi-head attention 实现 in_dim -> out_dim 的维度变换，
    同时让序列中的 token 能互相交互。
    支持 padding mask，防止注意力关注 padding 位置。
    """
    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 8):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        assert out_dim % num_heads == 0, f"out_dim({out_dim}) must be divisible by num_heads({num_heads})"

        self.norm = nn.LayerNorm(in_dim)
        self.q_proj = nn.Linear(in_dim, out_dim, bias=False)
        self.k_proj = nn.Linear(in_dim, out_dim, bias=False)
        self.v_proj = nn.Linear(in_dim, out_dim, bias=False)
        self.out_proj = nn.Linear(out_dim, out_dim)

        self.skip_proj = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, input_length: torch.Tensor = None) -> torch.Tensor:
        """
        x: (B, C_in, T)
        input_length: (B,) 每个样本的有效长度，None 表示全部有效
        """
        x = x.permute(0, 2, 1)  # (B, T, C_in)
        B, T, _ = x.shape

        attn_mask = None
        if input_length is not None:
            # key_padding: True = 需要被屏蔽的位置
            # scaled_dot_product_attention 的 attn_mask: True = 允许关注
            valid = torch.arange(T, device=x.device).unsqueeze(0) < input_length.unsqueeze(1)  # (B, T)
            # (B, 1, 1, T) 广播到 (B, H, T_q, T_k)
            attn_mask = valid.unsqueeze(1).unsqueeze(2)

        h = self.norm(x)
        q = self.q_proj(h).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn_out = scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        attn_out = attn_out.transpose(1, 2).reshape(B, T, self.out_dim)
        attn_out = self.out_proj(attn_out)

        out = self.skip_proj(x) + attn_out
        return out.permute(0, 2, 1)  # (B, C_out, T)


class FrameStackDownConv(nn.Module):
    """
    下采样模块：
    - 时间：通过 frame stacking 做 4x 下采样 (T -> T/stack_factor)
    - 通道：先从 D*stack_factor 压到 hidden_dim，再通过 ResidualUnit 提特征，最后到 latent_dim(32)
    
    Args:
        in_dim:        输入通道数（例如 768，对应 Whisper encoder 输出维度）
        latent_dim:    量化前的瓶颈维度（例如 32，对应 GroupFSQ 输入维度）
        stack_factor:  帧堆叠因子（50Hz -> 12.5Hz 就用 4）
        hidden_dim:    中间隐层通道数
        proj_type:     投影类型，"linear" 使用 WNConv1d，"attention" 使用 AttnProj1d
        num_heads:     注意力投影的头数（仅 proj_type="attention" 时生效）
    """
    def __init__(
        self,
        in_dim: int = 768,
        latent_dim: int = 32,
        stack_factor: int = 4,
        hidden_dim: int = 256,
        dilations = (1, 3, 9),
        proj_type: str = "linear",
        num_heads: int = 8,
    ):
        super().__init__()
        assert in_dim > 0
        assert latent_dim > 0
        assert stack_factor >= 1
        assert proj_type in ("linear", "attention")

        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.stack_factor = stack_factor
        self.hidden_dim = hidden_dim
        self.proj_type = proj_type

        stacked_dim = in_dim * stack_factor

        if proj_type == "attention":
            self.in_proj = AttnProj1d(stacked_dim, hidden_dim, num_heads=num_heads)
        else:
            self.in_proj = WNConv1d(stacked_dim, hidden_dim, kernel_size=1)

        blocks = []
        for d in dilations:
            blocks.append(ResidualUnit(hidden_dim, dilation=d))
        self.res_blocks = nn.Sequential(*blocks)

        self.to_latent = WNConv1d(hidden_dim, latent_dim, kernel_size=1)

        self.reset_parameters()

    def forward(self, x: torch.Tensor, input_length: torch.Tensor):
        """
        Args:
            x:            [B, D_in, T_in]
            input_length: [B] or scalar，表示 T_in（帧数）

        Returns:
            z:            [B, latent_dim, T_out]，T_out = ceil(T_in / stack_factor)
            output_length:[B] or scalar，等于 ceil(input_length / stack_factor)
        """
        B, D, T = x.shape
        s = self.stack_factor

        # 计算输出长度
        output_length = (input_length + s - 1) // s

        # pad 为 stack_factor 的倍数，时间维 pad 在最后
        T_padded = (T + s - 1) // s * s
        if T_padded > T:
            x = F.pad(x, (0, T_padded - T))  # (left, right) pad on time dim

        # Frame stacking: [B, D, T_padded] -> [B, D * s, T_padded / s]
        x = rearrange(x, 'b d (t s) -> b (d s) t', s=s)  # 这里的 t 就是 T_out（整数）

        # 通道侧压缩 + 残差建模
        if self.proj_type == "attention":
            h = self.in_proj(x, output_length)
        else:
            h = self.in_proj(x)
        h = self.res_blocks(h)    # [B, hidden_dim, T_out]

        # 投影到 latent_dim（32），给 GroupFSQ
        z = self.to_latent(h)     # [B, latent_dim, T_out]

        return z, output_length
    
    def reset_parameters(self):
        self.apply(init_weights)

class FrameStackUpConv(nn.Module):
    """
    上采样模块：
    - 输入：量化后的 latent [B, latent_dim, T_12_5]
    - 通道：先从 latent_dim 撑回 hidden_dim，经过若干 ResidualUnit
    - 时间：通过 1x1 conv 生成 out_dim * stack_factor 通道，再 unstack 回时间轴
            [B, out_dim * s, T_12_5] -> [B, out_dim, T_50]

    Args:
        latent_dim:   与 Down 的 latent_dim 一致，比如 32
        out_dim:      输出通道数（例如 768，后面再接 Whisper 对称 decoder / mel decoder）
        stack_factor: 与 Down 的 stack_factor 一致（4）
        hidden_dim:   中间隐层通道数
        proj_type:    投影类型，"linear" 使用 WNConv1d，"attention" 使用 AttnProj1d
        num_heads:    注意力投影的头数（仅 proj_type="attention" 时生效）
    """
    def __init__(
        self,
        latent_dim: int = 32,
        out_dim: int = 768,
        stack_factor: int = 4,
        hidden_dim: int = 256,
        dilations = (1, 3, 9),
        proj_type: str = "linear",
        num_heads: int = 8,
    ):
        super().__init__()
        assert latent_dim > 0
        assert out_dim > 0
        assert stack_factor >= 1
        assert proj_type in ("linear", "attention")

        self.latent_dim = latent_dim
        self.out_dim = out_dim
        self.stack_factor = stack_factor
        self.hidden_dim = hidden_dim
        self.proj_type = proj_type

        # 先从 32 撑回 hidden_dim
        self.from_latent = WNConv1d(latent_dim, hidden_dim, kernel_size=1)

        # 对称的 ResidualUnit 堆叠
        blocks = []
        for d in dilations:
            blocks.append(ResidualUnit(hidden_dim, dilation=d))
        self.res_blocks = nn.Sequential(*blocks)

        # 生成 out_dim * stack_factor 通道，用于 unstack
        if proj_type == "attention":
            self.to_stacked = AttnProj1d(hidden_dim, out_dim * stack_factor, num_heads=num_heads)
        else:
            self.to_stacked = WNConv1d(hidden_dim, out_dim * stack_factor, kernel_size=1)

        self.reset_parameters()

    def forward(self, z_q: torch.Tensor, input_len: torch.Tensor = None):
        """
        Args:
            z_q:       [B, latent_dim, T_12_5] （量化后的 latent）
            input_len: T_12_5（下采样长度），通常就是 Down 的 output_length；
                       输出长度 = input_len * stack_factor

        Returns:
            y:         [B, out_dim, T_50]
            out_len:   = input_len * stack_factor
        """
        s = self.stack_factor

        # latent_dim -> hidden_dim
        h = self.from_latent(z_q)   # [B, hidden_dim, T_12_5]

        # 残差卷积建模
        h = self.res_blocks(h)      # [B, hidden_dim, T_12_5]

        # 生成 stacked 通道
        if self.proj_type == "attention":
            h = self.to_stacked(h, input_len)
        else:
            h = self.to_stacked(h)  # [B, out_dim * s, T_12_5]

        # unstack 到时间维：[B, out_dim * s, T_12_5] -> [B, out_dim, T_50]
        y = rearrange(h, 'b (d s) t -> b d (t s)', s=s)

        # 长度计算
        out_len = None
        if input_len is not None:
            out_len = input_len * s

        return y, out_len
    
    def reset_parameters(self):
        self.apply(init_weights)


class SamplingBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        groups: int = 1,
        upsample_scale: int = 1,
        downsample_scale: int = 1,
    ) -> None:
        super(SamplingBlock, self).__init__()

        self.upsample_scale = upsample_scale
        self.downsample_scale = downsample_scale

        if self.upsample_scale > 1:
            self.de_conv_upsampler = nn.Sequential(
                nn.LeakyReLU(0.2),
                nn.ConvTranspose1d(
                    dim,
                    dim,
                    kernel_size=upsample_scale * 2,
                    stride=upsample_scale,
                    padding=upsample_scale // 2 + upsample_scale % 2,
                    output_padding=upsample_scale % 2,
                    groups=groups,
                ),
            )

        if self.downsample_scale > 1:
            self.conv_downsampler = nn.Sequential(
                nn.LeakyReLU(0.2),
                nn.Conv1d(
                    dim,
                    dim,
                    kernel_size=2 * downsample_scale,
                    stride=downsample_scale,
                    padding=downsample_scale // 2 + downsample_scale % 2,
                    groups=groups,
                ),
            )

    @staticmethod
    def repeat_upsampler(x, upsample_scale):
        return x.repeat_interleave(upsample_scale, dim=2)

    @staticmethod
    def skip_downsampler(x, downsample_scale):
        return F.avg_pool1d(x, kernel_size=downsample_scale, stride=downsample_scale)

    def forward(self, x):
        x = x.transpose(1,2)
        if self.upsample_scale > 1:
            repeat_res = self.repeat_upsampler(x, self.upsample_scale)
            deconv_res = self.de_conv_upsampler(x)
            upmerge_res = repeat_res + deconv_res
        else:
            upmerge_res = x
            repeat_res = x

        if self.downsample_scale > 1:
            conv_res = self.conv_downsampler(upmerge_res)
            skip2_res = self.skip_downsampler(upmerge_res, self.downsample_scale)
            skip1_res = self.skip_downsampler(repeat_res, self.downsample_scale)
        else:
            conv_res = upmerge_res
            skip2_res = upmerge_res
            skip1_res = repeat_res

        final_res = conv_res + skip1_res + skip2_res

        return final_res

class MambaAudioEncoder(nn.Module):
    """
    基于 mel 的双向 Mamba 编码器（Whisper 风格）
    """

    def __init__(
        self,
        num_mel_bins=80,
        stride_size=2,
        kernel_size=3,
        d_model: int = 512,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        d_intermediate: int = 0,
        bias: bool = True,
        fused_add_norm: bool = True,
        rms_norm: bool = False,
        norm_epsilon: float = 1e-5,
        initializer_cfg=None,
        residual_in_fp32: bool = False,
        n_mamba: int = 12,
        bidirectional: bool = True,
        bimamba_type: str = "v3",
        mamba_version: int = 1,
        # Mamba1 only
        dt_rank: str = "auto",
        # Mamba2 only
        headdim: int = 64,
        ngroups: int = 1,
        chunk_size: int = 256,
        use_mem_eff_path: bool = True,
    ):
        super().__init__()
        self.num_mel_bins = num_mel_bins
        self.d_model = d_model
        self.stride_size = stride_size

        # First convolution layer: Convert Mel spectrogram features (num_mel_bins) to hidden dimension (d_model)
        self.conv1 = nn.Conv1d(num_mel_bins, d_model, kernel_size=kernel_size, padding=1)
        # Second convolution layer: Apply downsampling with stride_size
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, stride=stride_size, padding=1)

        self.mamba_blocks = MambaLayers(
            n_mamba=n_mamba,
            mamba_version=mamba_version,
            bidirectional=bidirectional,
            bimamba_type=bimamba_type,
            d_model=d_model,
            d_state=d_state,
            d_intermediate=d_intermediate,
            expand=expand,
            d_conv=d_conv,
            dt_rank=dt_rank,
            headdim=headdim,
            ngroups=ngroups,
            chunk_size=chunk_size,
            use_mem_eff_path=use_mem_eff_path,
            conv_bias=True,
            bias=bias,
            fused_add_norm=fused_add_norm,
            rms_norm=rms_norm,
            norm_epsilon=norm_epsilon,
            initializer_cfg=initializer_cfg,
            residual_in_fp32=residual_in_fp32,
        )

    def forward(self, x: torch.Tensor, input_length: Optional[torch.Tensor] = None):
        """
        Args:
            x: (B, D, T) mel
            input_length: (B,) 原始长度（可选）
        Returns:
            encoded: (B, D, T)
            output_length: (B,) 帧数
        """
        # Get batch size and target sequence length
        bsz, _, tgt_len = x.size()

        # First layer convolution + SiLU activation, Convert Mel spectrogram to hidden states
        inputs_embeds = nn.functional.silu(self.conv1(x))  # (B, D, T)

        # Second layer convolution + SiLU activation, Apply downsampling with stride_size
        inputs_embeds = nn.functional.silu(self.conv2(inputs_embeds))  # (B, D, T)

        if input_length is not None:
            output_length = torch.div(
                input_length, self.stride_size, rounding_mode="floor"
            ).long()
        else:
            output_length = None

        # Mamba 序列建模（透传有效长度，避免反向扫描从 padding 开始）
        h_mamba = self.mamba_blocks(
            inputs_embeds.transpose(1, 2),
            seq_lens=output_length,
        )

        encoded = h_mamba.transpose(1, 2)  # (B, D, T)

        if output_length is None:
            output_length = torch.full((bsz,), tgt_len, dtype=torch.long, device=x.device)

        return encoded, output_length


class MambaAudioDecoder(nn.Module):
    """
    双向 Mamba 解码器（仿 Whisper 结构）
    """

    def __init__(
        self,
        num_mel_bins=80,
        stride_size=2,
        kernel_size=3,
        d_model: int = 512,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        d_intermediate: int = 0,
        bias: bool = True,
        fused_add_norm: bool = True,
        rms_norm: bool = False,
        norm_epsilon: float = 1e-5,
        initializer_cfg=None,
        residual_in_fp32: bool = False,
        n_mamba: int = 12,
        bidirectional: bool = True,
        bimamba_type: str = "v3",
        mamba_version: int = 1,
        # Mamba1 only
        dt_rank: str = "auto",
        # Mamba2 only
        headdim: int = 64,
        ngroups: int = 1,
        chunk_size: int = 256,
        use_mem_eff_path: bool = True,
    ):
        super().__init__()
        self.num_mel_bins = num_mel_bins
        self.d_model = d_model
        self.stride_size = stride_size

        self.mamba_blocks = MambaLayers(
            n_mamba=n_mamba,
            mamba_version=mamba_version,
            bidirectional=bidirectional,
            bimamba_type=bimamba_type,
            d_model=d_model,
            d_state=d_state,
            d_intermediate=d_intermediate,
            expand=expand,
            d_conv=d_conv,
            dt_rank=dt_rank,
            headdim=headdim,
            ngroups=ngroups,
            chunk_size=chunk_size,
            use_mem_eff_path=use_mem_eff_path,
            conv_bias=True,
            bias=bias,
            fused_add_norm=fused_add_norm,
            rms_norm=rms_norm,
            norm_epsilon=norm_epsilon,
            initializer_cfg=initializer_cfg,
            residual_in_fp32=residual_in_fp32,
        )

        # Correct transpose convolution layer to ensure output length close to stride_size times
        self.deconv1 = nn.ConvTranspose1d(
            d_model, 
            d_model, 
            kernel_size=kernel_size, 
            stride=stride_size, 
            padding=0,  # Do not fill input side
            output_padding=0  # Can be adjusted to precisely control length
        )
        self.deconv2 = nn.ConvTranspose1d(
            d_model, 
            num_mel_bins, 
            kernel_size=kernel_size, 
            stride=1,  # Only convert channels, do not change length
            padding=0
        )

    def forward(self, x: torch.Tensor, input_length: Optional[torch.Tensor] = None):
        """
        Args:
            x: (B, D, T)
            input_length: (B,) 帧长度（可选）
        Returns:
            mel: (B, D, T)
            output_length: (B,) 波形长度
        """
        bsz, _, tgt_len = x.size()

        hidden_states = self.mamba_blocks(
            x.transpose(1, 2),
            seq_lens=input_length,
        )  # (B, T, D)

        output_features = nn.functional.silu(self.deconv1(hidden_states.transpose(1, 2))) # (B, D, T)
        output_features = nn.functional.silu(self.deconv2(output_features)) # (B, D, T)

        # If strictly stride_size times length is needed, can trim extra parts
        expected_length = tgt_len * self.stride_size
        if output_features.size(2) > expected_length:
            output_features = output_features[:, :, :expected_length]

        if input_length is not None:
            output_length = input_length * self.stride_size
        else:
            output_length = torch.full((bsz,), output_features.size(2), dtype=torch.long, device=output_features.device)
        # Output shape: [bsz, num_mel_bins, seq_len]
        return output_features, output_length

class PreProjection(nn.Module):
    """
    Pre-quantization projection layer: d_model -> fsq_dim
    Projects Mamba encoder output to FSQ input dimension
    
    Expects input format: (B, d_model, T)
    Returns output format: (B, fsq_dim, T)
    """
    
    def __init__(
        self, 
        d_model: int = 512,
        fsq_dim: int = 8,
    ):
        """
        Args:
            d_model: Input dimension from encoder (512)
            fsq_dim: Output dimension for FSQ (8)
        """
        super().__init__()
        self.d_model = d_model
        self.fsq_dim = fsq_dim
        
        # Linear projection without bias
        # No bias because FSQ handles centering internally
        if d_model != fsq_dim:
            self.proj = weight_norm(nn.Linear(d_model, fsq_dim))
        else:
            self.proj = nn.Identity()
    
    def forward(self, x: torch.Tensor, input_length=None) -> tuple:
        """
        Args:
            x: Encoder output of shape (B, d_model, T)
            input_length: (B,) - input length (可选)

        Returns:
            tuple: (FSQ input, output_length)
                - FSQ input of shape (B, fsq_dim, T)
                - output_length: (B,) - output length (与输入长度相同)
        """
        B, D, T = x.shape
        assert D == self.d_model, f"Expected d_model={self.d_model}, got {D}"

        # Transpose to (B, T, d_model) for Linear and LayerNorm
        x = x.transpose(1, 2)

        # Project to FSQ dimension
        x = self.proj(x)  # (B, T, fsq_dim)

        # Transpose back to (B, fsq_dim, T) for FSQ
        x = x.transpose(1, 2)

        # 输出长度与输入长度相同（投影不改变序列长度）
        if input_length is not None:
            output_length = input_length
        else:
            output_length = torch.tensor([T] * B, device=x.device)

        return x, output_length

    def remove_weight_norm(self):
        """Remove weight normalization for inference optimization."""
        if parametrize.is_parametrized(self.proj, 'weight'):
            parametrize.remove_parametrizations(self.proj, 'weight', leave_parametrized=True)

class PostProjection(nn.Module):
    """
    Post-quantization projection layer: fsq_dim -> d_model
    Projects FSQ output back to Mamba decoder input dimension
    
    Expects input format: (B, fsq_dim, T)
    Returns output format: (B, d_model, T)
    """
    
    def __init__(
        self,
        fsq_dim: int = 8,
        d_model: int = 512,
    ):
        """
        Args:
            fsq_dim: Input dimension from FSQ (8)
            d_model: Output dimension for decoder (512)
        """
        super().__init__()
        self.fsq_dim = fsq_dim
        self.d_model = d_model

        if fsq_dim != d_model:
            self.proj = weight_norm(nn.Linear(fsq_dim, d_model))
        else:
            self.proj = nn.Identity()
        
    def forward(self, x: torch.Tensor, input_length=None) -> tuple:
        """
        Args:
            x: FSQ output of shape (B, fsq_dim, T)
            input_length: (B,) - input length (可选)

        Returns:
            tuple: (Decoder input, output_length)
                - Decoder input of shape (B, d_model, T)
                - output_length: (B,) - output length (与输入长度相同)
        """
        B, D, T = x.shape
        assert D == self.fsq_dim, f"Expected fsq_dim={self.fsq_dim}, got {D}"

        # Transpose to (B, T, fsq_dim) for Linear
        x = x.transpose(1, 2)

        # Project to model dimension
        x = self.proj(x)  # (B, T, d_model)

        # Transpose back to (B, d_model, T) for decoder
        x = x.transpose(1, 2)

        # 输出长度与输入长度相同（投影不改变序列长度）
        if input_length is not None:
            output_length = input_length
        else:
            output_length = torch.tensor([T] * B, device=x.device)

        return x, output_length

    def remove_weight_norm(self):
        """Remove weight normalization for inference optimization."""
        if parametrize.is_parametrized(self.proj, 'weight'):
            parametrize.remove_parametrizations(self.proj, 'weight', leave_parametrized=True)


class GeGluMlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        act_layer = None,
        drop = 0.0,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(in_features, eps=1e-6)
        self.act = nn.GELU(approximate='tanh')
        self.w0 = nn.Linear(in_features, hidden_features)
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        x = self.norm(x)
        x = self.act(self.w0(x)) * self.w1(x)
        x = self.w2(x)
        return x

class PlainAttention(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads):
        super().__init__()
        if in_dim > out_dim:
            # assert in_dim // num_heads == out_dim
            self.head_dim = in_dim // num_heads
            self.qkv = nn.Linear(in_dim, in_dim * 3, bias=False)
            self.q_bias = nn.Parameter(torch.zeros(in_dim))
            self.v_bias = nn.Parameter(torch.zeros(in_dim))
            self.register_buffer('zero_k_bias', torch.zeros(in_dim))
        else:
            # assert out_dim // num_heads == in_dim
            self.head_dim = out_dim // num_heads
            self.qkv = nn.Linear(in_dim, out_dim * 3, bias=False)
            self.q_bias = nn.Parameter(torch.zeros(out_dim))
            self.v_bias = nn.Parameter(torch.zeros(out_dim))
            self.register_buffer('zero_k_bias', torch.zeros(out_dim))

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.scale = self.head_dim ** -0.5
        self.proj = nn.Linear(out_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=torch.cat((self.q_bias, self.zero_k_bias, self.v_bias)))
        q, k, v = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)

        x = scaled_dot_product_attention(q, k, v)

        if self.in_dim > self.out_dim:
            x = torch.mean(x, dim=1)
            if self.in_dim // self.num_heads != self.out_dim:
                x = nn.functional.adaptive_avg_pool1d(x, self.out_dim)
        else:
            x = x.transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        return x


class AttnProjection(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, norm_layer=nn.LayerNorm, mlp_ratio=2):
        super().__init__()
        assert out_dim % in_dim == 0 or in_dim % out_dim == 0
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.norm1 = norm_layer(in_dim)
        self.attn = PlainAttention(in_dim, out_dim, num_heads)
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm3 = norm_layer(in_dim)

        self.norm2 = norm_layer(out_dim)
        hidden_dim = int(out_dim * mlp_ratio)
        self.mlp = GeGluMlp(
            in_features=out_dim,
            hidden_features=hidden_dim
        )

    def forward(self, x):
        x = self.proj(self.norm3(x)) + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
