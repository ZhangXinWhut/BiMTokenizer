import os
import argparse
import logging
import torch
import yaml

from utils.helpers import set_logging, load_audio, save_audio, find_audio_files
from bimtokenizer.model import BiMTokenizer

if __name__ == "__main__":
    set_logging()
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", type=str,
        default="./config/bimtokenizer_sensevoice_32768_4096_librispeech.yaml",
    )
    parser.add_argument(
        "--checkpoint_path", type=str,
        default="./weights/bimtokenizer_sensevoice_32768_4096_librispeech.pt",
    )
    parser.add_argument("--device", type=str, default="cuda")
    
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--input_dir", type=str, default="input_wavs")
    parser.add_argument("--output_dir", type=str, default="output_wavs")
    parser.add_argument(
        "--n_codebooks",
        type=int,
        default=None,
        help="使用的码本数量；不指定时读取 YAML，仍未配置则使用全部码本",
    )

    args = parser.parse_args()

    device = torch.device(args.device)

    with open(args.config_path, "r") as f:
        inference_config = yaml.safe_load(f) or {}
    configured_n_codebooks = (inference_config.get("inference") or {}).get(
        "n_codebooks"
    )
    n_codebooks = (
        args.n_codebooks if args.n_codebooks is not None else configured_n_codebooks
    )

    generator = BiMTokenizer.load_from_checkpoint(
        config_path=args.config_path,
        ckpt_path=args.checkpoint_path,
        remove_weight_norm=True,
    ).to(device).eval()

    selected_n_codebooks = generator.nq if n_codebooks is None else int(n_codebooks)
    if not 1 <= selected_n_codebooks <= generator.nq:
        raise ValueError(
            f"n_codebooks={selected_n_codebooks} 超出范围，须在 [1, {generator.nq}]"
        )

    model_config = inference_config.get("model") or inference_config.get(
        "generator_params", {}
    )
    codebook_size = int((model_config.get("quantizer") or {}).get("codebook_size", 0))
    bitrate = None
    if codebook_size > 1:
        bitrate = 12.5 * selected_n_codebooks * torch.log2(
            torch.tensor(float(codebook_size))
        ).item()
    bitrate_text = f", bitrate={bitrate:g} bps" if bitrate is not None else ""
    logging.info(
        f"Using {selected_n_codebooks}/{generator.nq} codebooks{bitrate_text}"
    )
    
    ## Find audios
    audio_paths = find_audio_files(input_dir=args.input_dir)
    
    ## Create output directory if not exists
    os.makedirs(args.output_dir, exist_ok=True)
    logging.info(f"Processing {len(audio_paths)} audio files, output will be saved to {args.output_dir}")

    with torch.no_grad():
        ## Process audios in batches
        batch_size = args.batch_size
        for i in range(0, len(audio_paths), batch_size):
            batch_paths = audio_paths[i:i + batch_size]
            logging.info(f"Processing batch {i // batch_size + 1}/{len(audio_paths) // batch_size + 1}, files: {batch_paths}")

            # Load audio files
            wav_list = [load_audio(path, target_sample_rate=generator.input_sample_rate).squeeze().to(device) for path in batch_paths]
            logging.info(f"Successfully loaded {len(wav_list)} audio files with lengths {[len(wav) for wav in wav_list]} samples")

            orig_lengths = [len(wav) for wav in wav_list]

            # Encode
            encode_result = generator.encode(
                wav_list, device=device, n_codebooks=selected_n_codebooks
            )
            codes_list = encode_result["codes_list"]  # B * (nq, T)
            logging.info(f"Encoding completed, code lengths: {[codes.shape[-1] for codes in codes_list]}")
            # logging.info(f"{codes_list = }")

            # Decode
            # 透传原始采样点数：frame stacking 向上取整会让最后一个 token 覆盖到
            # 零填充区域，不裁的话尾部会多出至多 1279 个采样点的噪声/静音。
            decode_result = generator.decode(
                codes_list,
                device=device,
                wav_lengths=encode_result["wav_lengths"],
                n_codebooks=selected_n_codebooks,
            )
            syn_wav_list = decode_result["syn_wav_list"]  # B * (T,)
            logging.info(f"Decoding completed, generated waveform lengths: {[len(wav) for wav in syn_wav_list]} samples")

            # Save generated audios（统一为 .wav，避免输入为 .flac 时 torchaudio 按扩展名写 FLAC 报错）
            for path, syn_wav, orig_len in zip(batch_paths, syn_wav_list, orig_lengths):
                stem = os.path.splitext(os.path.basename(path))[0]
                output_path = os.path.join(args.output_dir, stem + ".wav")
                syn_wav = syn_wav[:orig_len]
                save_audio(output_path, syn_wav.cpu().reshape(1, -1), sample_rate=generator.output_sample_rate)
                logging.info(
                    f"Saved generated audio to {output_path} "
                    f"({len(syn_wav)} / {orig_len} samples)"
                )

        
    logging.info("All audio processing completed")
