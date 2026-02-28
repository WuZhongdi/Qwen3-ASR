import os
import json
import yaml
import pandas as pd
from pathlib import Path
import subprocess
from tqdm import tqdm


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def format_text(language_tag: str, transcript: str) -> str:
    return f"language {language_tag}<asr_text>{transcript.strip()}"


def convert_webm_to_wav(input_path: Path, output_path: Path):
    """
    Uses ffmpeg to convert webm to wav.
    Assumes ffmpeg is installed on HPC.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        str(output_path)
    ]

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def process_split(df, config, split_name):
    output_dir = Path(config["output_dir"])
    output_file = output_dir / f"{split_name}.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_col = config["audio_id_column"]
    text_col = config["text_column"]
    language_tag = config.get("language_tag", "English")
    extension = config.get("audio_extension", ".webm")
    use_absolute = config.get("use_absolute_paths", True)

    convert_audio = config.get("convert_to_wav", False)
    converted_dir = Path(config.get("converted_audio_dir", ""))

    audio_dir = Path(config["audio_dir"])

    kept = 0
    skipped_empty = 0
    skipped_missing = 0

    with open(output_file, "w", encoding="utf-8") as fout:
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split_name}"):

            transcript = str(row[text_col]).strip()

            # 1️⃣ Filter empty transcripts
            if transcript == "" or transcript.lower() == "nan":
                skipped_empty += 1
                continue

            audio_id = str(row[audio_col])
            input_audio_path = audio_dir / f"{audio_id}{extension}"

            # 2️⃣ Drop missing audio files
            if not input_audio_path.exists():
                skipped_missing += 1
                continue

            # 3️⃣ Convert if needed
            if convert_audio:
                output_audio_path = converted_dir / f"{audio_id}.wav"

                if not output_audio_path.exists():
                    convert_webm_to_wav(input_audio_path, output_audio_path)

                final_audio_path = output_audio_path
            else:
                final_audio_path = input_audio_path

            final_audio_path = (
                final_audio_path.resolve() if use_absolute else final_audio_path
            )

            record = {
                "audio": str(final_audio_path),
                "text": format_text(language_tag, transcript)
            }

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

    print(f"\nSplit: {split_name}")
    print(f"Kept: {kept}")
    print(f"Skipped empty transcripts: {skipped_empty}")
    print(f"Skipped missing audio: {skipped_missing}")
    print(f"Saved to: {output_file}\n")


def main(config):

    df = pd.read_parquet(config["parquet_path"])

    audio_col = config["audio_id_column"]
    text_col = config["text_column"]
    fold_col = config["fold_column"]
    val_fold = config["validation_fold_index"]

    required_cols = [audio_col, text_col, fold_col]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in parquet.")

    # 4️⃣ Train/Validation split using fold
    train_df = df[df[fold_col] != val_fold].reset_index(drop=True)
    val_df = df[df[fold_col] == val_fold].reset_index(drop=True)

    print(f"Total samples: {len(df)}")
    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")

    process_split(train_df, config, "train")
    process_split(val_df, config, "validation")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare Qwen JSONL dataset.")
    parser.add_argument("--config", type=str, required=True)

    args = parser.parse_args()
    config = load_config(args.config)
    main(config)