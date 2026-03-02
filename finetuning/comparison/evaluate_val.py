import yaml
import json
import torch
from tqdm import tqdm
from torchmetrics.text import WordErrorRate, CharErrorRate
from qwen_asr import Qwen3ASRModel
from TextNormalizer import TextNormalizer
import argparse

class TextNormalize:
    def __init__(self):
        # Initialize any resources needed for normalization
        self.text_normalizer = TextNormalizer()

    def normalize(self, text: str) -> str:
        return self.text_normalizer.normalize(text)

def load_config(config_path: str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_model(cfg):
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }

    model = Qwen3ASRModel.from_pretrained(
        cfg["model"]["name"],
        cache_dir=cfg["model"]["cache_dir"],
        dtype=dtype_map[cfg["model"]["dtype"]],
        device_map=cfg["model"]["device"],
        max_inference_batch_size=cfg["model"]["max_inference_batch_size"],
        max_new_tokens=cfg["model"]["max_new_tokens"],
    )

    return model

def load_data(data_path: str):
    with open(data_path, "r") as f:
        return [json.loads(line) for line in f]

def run_inference(model, data, cfg):

    normalizer = TextNormalize()
    preds, targets = [], []

    use_context = cfg["evaluation"]["use_context"]
    context_field = cfg["evaluation"]["context_field"]

    for sample in tqdm(data):

        #  Context hook (leave for you to customize if needed)
        context_input = ""
        if use_context:
            context_input = sample.get(context_field, "")

        result = model.transcribe(
            audio=sample["audio"],
            context=context_input,
            language=cfg['evaluation']['language']
        )

        preds.append(normalizer.normalize(result[0].text))
        targets.append(sample["text"].split("<asr_text>")[-1])

    return preds, targets

def compute_metrics(preds, targets):
    wer_metric = WordErrorRate()
    cer_metric = CharErrorRate()

    wer = wer_metric(preds, targets).item()
    cer = cer_metric(preds, targets).item()

    return wer, cer

def main(config_path: str):

    cfg = load_config(config_path)

    model = load_model(cfg)

    data = load_data(cfg["data"]["val_file"])

    preds, targets = run_inference(model, data, cfg)

    wer, cer = compute_metrics(preds, targets)

    # Final Output ONLY
    print(f"WER: {wer:.4f}")
    print(f"CER: {cer:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASR Evaluation Script")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file"
    )

    args = parser.parse_args()

    main(args.config)