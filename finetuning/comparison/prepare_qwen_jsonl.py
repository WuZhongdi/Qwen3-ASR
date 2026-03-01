import os
import json
import yaml
import argparse
from pathlib import Path

import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm


# =========================
# Optional text normalizer
# =========================
def normalize_text(text: str) -> str:
    """
    Put your text normalization logic here.
    This is where you would:
        - lowercase
        - remove punctuation
        - expand numbers
        - apply language-specific normalization
        - etc.

    For now, it's identity.
    """

    # ---- EXAMPLE (uncomment if needed) ----
    # text = text.lower()
    # text = text.strip()
    # ---------------------------------------

    return text


# =========================
# Audio Processing
# =========================
def convert_and_resample_audio(
    input_path: Path,
    output_path: Path,
    target_sr: int = 16000,
):
    """
    Converts .webm (or any readable format) to .wav
    and resamples to target_sr (default 16kHz).
    """

    try:
        audio, sr = librosa.load(input_path, sr=None)  # load original sr

        # Resample if needed
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            sr = target_sr

        sf.write(output_path, audio, sr)
        return True

    except Exception as e:
        print(f"[ERROR] Failed audio processing: {input_path} | {e}")
        return False


# =========================
# Main processing
# =========================
def process_dataset(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    parquet_path = Path(config["parquet_path"])
    audio_root = Path(config["audio_root"])
    output_root = Path(config["output_root"])
    fold_column = config["fold_column"]
    val_fold_index = config["val_fold_index"]
    audio_column = config.get("audio_column", "audio")
    text_column = config.get("text_column", "text")
    target_sr = config.get("target_sample_rate", 16000)

    output_root.mkdir(parents=True, exist_ok=True)
    wav_output_dir = output_root / "wavs"
    wav_output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading parquet...")
    df = pd.read_parquet(parquet_path)

    print(f"Total samples before filtering: {len(df)}")

    # ----------------------------------
    # 1. Filter empty transcripts
    # ----------------------------------
    df = df[df[text_column].notna()]
    df = df[df[text_column].str.strip() != ""]

    print(f"After transcript filtering: {len(df)}")

    train_records = []
    val_records = []

    print("Processing audio files...")

    for _, row in tqdm(df.iterrows(), total=len(df)):

        audio_rel_path = row[audio_column]
        transcript = row[text_column]
        fold_value = row[fold_column]

        input_audio_path = audio_root / audio_rel_path

        if not input_audio_path.exists():
            print(f"[WARNING] Missing audio file: {input_audio_path}")
            continue  # drop instead of crash

        # Convert to wav filename
        wav_filename = Path(audio_rel_path).with_suffix(".wav").name
        output_audio_path = wav_output_dir / wav_filename

        success = convert_and_resample_audio(
            input_audio_path,
            output_audio_path,
            target_sr=target_sr,
        )

        if not success:
            continue

        # ----------------------------------
        # 2. TEXT NORMALIZATION HOOK
        # ----------------------------------
        transcript = normalize_text(transcript)

        # Qwen ASR format
        formatted_text = f"language English<asr_text>{transcript}"

        record = {
            "audio": str(output_audio_path),
            "text": formatted_text,
        }

        if fold_value == val_fold_index:
            val_records.append(record)
        else:
            train_records.append(record)

    # Write JSONL files
    train_path = output_root / "train.jsonl"
    val_path = output_root / "val.jsonl"

    print("Writing JSONL files...")

    with open(train_path, "w") as f:
        for r in train_records:
            f.write(json.dumps(r) + "\n")

    with open(val_path, "w") as f:
        for r in val_records:
            f.write(json.dumps(r) + "\n")

    print("Done.")
    print(f"Train samples: {len(train_records)}")
    print(f"Validation samples: {len(val_records)}")
    print(f"Audio resampled to: {target_sr} Hz")


# =========================
# CLI
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to data_config.yaml")
    args = parser.parse_args()

    process_dataset(args.config)