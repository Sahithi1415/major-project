import argparse
from pathlib import Path
import sys

import sacrebleu
import torch
from tokenizers import Tokenizer
from tqdm import tqdm

from .config import TrainConfig
from .data import load_parallel_data, split_items
from .inference import translate
from .model import HybridTranslator


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--beam_size", type=int, default=4)
    return p.parse_args()


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    artifacts = Path(args.checkpoint).parent
    cfg = TrainConfig.load(str(artifacts / "config.json"))
    tokenizer = Tokenizer.from_file(str(artifacts / "tokenizer.json"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HybridTranslator(
        vocab_size=tokenizer.get_vocab_size(),
        max_len=cfg.max_len,
        d_model=cfg.d_model,
        lstm_hidden=cfg.lstm_hidden,
        nhead=cfg.nhead,
        num_encoder_layers=cfg.num_encoder_layers,
        num_decoder_layers=cfg.num_decoder_layers,
        dim_ff=cfg.dim_ff,
        dropout=cfg.dropout,
        quantum_qubits=cfg.quantum_qubits,
        quantum_layers=cfg.quantum_layers,
        use_quantum=cfg.use_quantum,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    _, val_items = split_items(load_parallel_data(args.data_path), seed=cfg.seed)
    refs = []
    hyps = []
    for it in tqdm(val_items, desc="eval"):
        pred = translate(model, tokenizer, cfg, it.source, it.target_lang, device, beam_size=args.beam_size)
        refs.append(it.target)
        hyps.append(pred)

    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    print(f"BLEU: {bleu.score:.2f}")
    print("Sample predictions:")
    for i in range(min(5, len(hyps))):
        print(f"\nEN : {val_items[i].source}")
        print(f"REF: {refs[i]}")
        print(f"HYP: {hyps[i]}")


if __name__ == "__main__":
    main()
