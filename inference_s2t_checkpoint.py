import argparse
from typing import Any, Dict

import torch
import torchaudio

from seamless_communication.inference import Translator


def _normalize_checkpoint_keys(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    prefixes = ("module.model.model.", "model.model.", "module.model.", "model.", "module.")
    normalized: Dict[str, Any] = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in prefixes:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
                break
        normalized[new_key] = value
    return normalized


def load_finetuned_checkpoint(translator: Translator, checkpoint_path: str, device: torch.device) -> None:
    raw_ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = raw_ckpt["model"] if isinstance(raw_ckpt, dict) and "model" in raw_ckpt else raw_ckpt
    if not isinstance(state_dict, dict):
        raise ValueError("Unsupported checkpoint format: expected dict or {'model': state_dict}.")

    state_dict = _normalize_checkpoint_keys(state_dict)
    missing, unexpected = translator.model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run T2ST inference with a local finetuned checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to finetuned checkpoint.pt")
    parser.add_argument("--text", type=str, required=True, help="Input source text")
    parser.add_argument("--src_lang", type=str, required=True, help="Source language (M4T code), e.g. vie")
    parser.add_argument("--tgt_lang", type=str, required=True, help="Target language (M4T code), e.g. vie")
    parser.add_argument("--output_wav", type=str, required=True, help="Output wav path")
    parser.add_argument("--model_name", type=str, default="seamlessM4T_medium", help="Base UnitY model")
    parser.add_argument("--vocoder_name", type=str, default="vocoder_36langs", help="Vocoder model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    translator = Translator(
        args.model_name,
        args.vocoder_name,
        device=device,
        dtype=dtype,
    )
    load_finetuned_checkpoint(translator, args.checkpoint, device=device)

    text_out, speech_out = translator.predict(
        input=args.text,
        task_str="T2ST",
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
    )

    if speech_out is None or not speech_out.audio_wavs:
        raise RuntimeError("No speech output generated.")

    torchaudio.save(
        args.output_wav,
        speech_out.audio_wavs[0][0].to(torch.float32).cpu(),
        speech_out.sample_rate,
    )
    print(f"Translated text: {text_out[0]}")
    print(f"Saved audio: {args.output_wav}")


if __name__ == "__main__":
    main()
