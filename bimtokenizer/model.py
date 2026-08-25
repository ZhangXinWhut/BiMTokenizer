# -*- coding: utf-8 -*-
import yaml
import logging
import torch
import torch.nn as nn
from torch.nn.utils import parametrize

from bimtokenizer.modules.modules import (
    MambaAudioEncoder,
    MambaAudioDecoder,
    FrameStackDownConv,
    FrameStackUpConv,
)
from bimtokenizer.modules.quantizer.rslq import ResidualSphericalLeech
from bimtokenizer.modules.quantizer.rslq_noscale import ResidualSphericalLeechNoScale
from bimtokenizer.modules.vocos import Vocos
from bimtokenizer.modules.feature_extractor import MelFeatureExtractor


class BiMTokenizer(nn.Module):
    def __init__(self, generator_params):
        super().__init__()
        # Basic parameters
        self.sample_rate = generator_params['sample_rate']
        self.downsample_rate = generator_params['encoder_downsample_rate']
        self.decoder_upsample_rate = generator_params['decoder_upsample_rate']

        ## Codec part
        self.encoder = MambaAudioEncoder(**generator_params['encoder'])
        self.downsample = FrameStackDownConv(**generator_params['downsample'])

        ## Quantizer
        self.quantizer_type = generator_params.get('quantizer_type', 'rslq')
        if self.quantizer_type == 'rslq_noscale':
            self.quantizer = ResidualSphericalLeechNoScale(**generator_params['quantizer'])
        elif self.quantizer_type in (None, 'rslq'):
            self.quantizer = ResidualSphericalLeech(**generator_params['quantizer'])
        else:
            raise ValueError(
                f"不支持的 quantizer_type={self.quantizer_type!r}，"
                f"仅支持 'rslq' / 'rslq_noscale'。"
            )
        self.nq = generator_params['quantizer']['num_codebooks']

        self.upsample = FrameStackUpConv(**generator_params['upsample'])
        self.decoder = MambaAudioDecoder(**generator_params['decoder'])
        self.vocos = Vocos(**generator_params['vocos'])

        ## Feature extractor
        self.feature_extractor = MelFeatureExtractor(**generator_params['feature_extractor'])

    @property
    def input_sample_rate(self):
        return self.sample_rate

    @property
    def output_sample_rate(self):
        return self.sample_rate

    @torch.inference_mode()
    def inference_tokenize(self, x, input_lengths):
        """
            Input:
                x: Waveform tensor # (B, 1, T)
                input_lengths: Valid length for each sample # (B,)
            Output:
                dict: Contains the following key-value pairs
                    "zq": Quantized embeddings # (B, D, T)
                    "codes": Quantization codes # (nq, B, T)
                    "codes_lengths": Quantization code lengths # (B,)
        """
        list_x = [xi[:, :x_len].reshape(-1).cpu().numpy() for xi, x_len in zip(x, input_lengths)]
        features = self.feature_extractor(
            list_x,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            return_attention_mask=True,
            padding="longest",
            truncation=False,
        )
        input_mel = features["input_features"].to(x.device).to(x.dtype)
        mel_lens = features["attention_mask"].sum(dim=-1).long().to(x.device)

        encoder_output, encoder_output_length = self.encoder(input_mel, mel_lens)
        downsample_output, downsample_output_length = self.downsample(
            encoder_output, encoder_output_length
        )

        quantized_output, codes = self.quantizer(
            downsample_output,
            downsample_output_length,
        )

        return {
            "zq": quantized_output,
            "codes": codes,
            "codes_lengths": downsample_output_length,
        }

    @torch.inference_mode()
    def inference_detokenize(self, codes, codes_lengths):
        """
            Input:
                codes: Quantization codes # (nq, B, T)
                codes_lengths: Quantization code lengths for each sample # (B,)
            Output:
                dict: Contains the following key-value pairs
                    "y": Synthesized audio waveform # (B, 1, T)
                    "output_length": Output lengths # (B,)
        """
        zq = self.quantizer.decode(codes)

        upsample_output, upsample_output_length = self.upsample(zq, codes_lengths)
        decoder_output, decoder_output_length = self.decoder(
            upsample_output, upsample_output_length
        )
        y, vocos_output_length = self.vocos(decoder_output, decoder_output_length)

        return {
            "y": y,
            "output_length": vocos_output_length,
        }

    @torch.inference_mode()
    def encode(self, wav_list, device=torch.device("cuda")):
        """
            Input:
                wav_list: List of audio waveforms, each with potentially different length # B * (T,)
            Output:
                dict: Contains the following key-value pairs
                    "codes_list": List of quantization codes # B * (nq, T)
                    "wav_lengths": Original waveform lengths in samples # list[int]
        """
        max_length = max(len(wav) for wav in wav_list)
        batch_size = len(wav_list)
        wav_tensor = torch.zeros(batch_size, 1, max_length, device=device)
        input_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
        for i, wav in enumerate(wav_list):
            wav_tensor[i, 0, :len(wav)] = wav.to(device)
            input_lengths[i] = len(wav)

        result = self.inference_tokenize(wav_tensor, input_lengths)
        chunk_codes = result["codes"]
        chunk_code_lengths = result["codes_lengths"]

        codes_list = [
            chunk_codes[:, i, :int(chunk_code_lengths[i].item())].clone()
            for i in range(batch_size)
        ]

        return {
            "codes_list": codes_list,
            "wav_lengths": [int(l.item()) for l in input_lengths],
        }

    @torch.inference_mode()
    def decode(self, codes_list, device=torch.device("cuda"), wav_lengths=None):
        """
            Input:
                codes_list: List of quantization codes # B * (nq, T)
                wav_lengths: Optional original waveform lengths; trim output to avoid
                             padding artifacts from frame stacking
            Output:
                dict: Contains the following key-value pairs
                    "syn_wav_list": List of synthesized audio waveforms # B * (T,)
        """
        max_code_length = max(codes.shape[-1] for codes in codes_list)
        batch_size = len(codes_list)
        codes_tensor = torch.zeros(
            self.nq, batch_size, max_code_length, device=device, dtype=torch.long
        )
        code_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
        for i, codes in enumerate(codes_list):
            codes_tensor[:, i, :codes.shape[-1]] = codes.to(device)
            code_lengths[i] = codes.shape[-1]

        result = self.inference_detokenize(codes_tensor, code_lengths)
        syn_wav_list = []
        for i in range(batch_size):
            keep = int(code_lengths[i].item()) * self.decoder_upsample_rate
            if wav_lengths is not None:
                keep = min(keep, int(wav_lengths[i]))
            syn_wav_list.append(result["y"][i, 0, :min(keep, result["y"].shape[-1])])

        return {"syn_wav_list": syn_wav_list}

    def remove_weight_norm(self):
        logging.info("Removing weight normalization for inference optimization...")

        def _remove_wn(module):
            for child in module.modules():
                if parametrize.is_parametrized(child, 'weight'):
                    parametrize.remove_parametrizations(child, 'weight', leave_parametrized=True)

        _remove_wn(self.downsample)
        _remove_wn(self.upsample)
        logging.info("Weight normalization removed successfully")

    @classmethod
    def load_from_checkpoint(cls, config_path: str, ckpt_path: str,
                             remove_weight_norm: bool = True):
        logging.info(f"Loading model from {config_path} and {ckpt_path}")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        gen_params = config.get('generator_params', config.get('model', {}))
        model = cls(gen_params)

        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif isinstance(checkpoint, dict) and 'generator' in checkpoint:
            state_dict = checkpoint['generator']
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=False)

        if remove_weight_norm:
            model.remove_weight_norm()

        return model
