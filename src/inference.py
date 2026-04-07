import argparse
from pathlib import Path
import sys

import torch
from tokenizers import Tokenizer

from .config import TrainConfig
from .model import HybridTranslator
from .utils import BOS, EOS, PAD, L_HI, L_TE, causal_mask


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--text", type=str, required=True)
    p.add_argument("--lang", type=str, required=True, choices=["hi", "te"])
    p.add_argument("--max_new_tokens", type=int, default=40)
    p.add_argument("--beam_size", type=int, default=4)
    return p.parse_args()


def pad_to_len(ids, max_len, pad_id):
    ids = ids[:max_len]
    return ids + [pad_id] * (max_len - len(ids))


@torch.no_grad()
def translate(
    model,
    tokenizer,
    cfg,
    text: str,
    lang: str,
    device,
    max_new_tokens: int = 40,
    beam_size: int = 4,
):
    vocab = tokenizer.get_vocab()
    pad_id = vocab[PAD]
    bos_id = vocab[BOS]
    eos_id = vocab[EOS]
    lang_id = vocab[L_HI] if lang == "hi" else vocab[L_TE]

    src_ids = tokenizer.encode(text).ids
    src = [bos_id, lang_id] + src_ids + [eos_id]
    src = torch.tensor([pad_to_len(src, cfg.max_len, pad_id)], dtype=torch.long, device=device)
    src_pad_mask = src.eq(pad_id)

    ban_ids = {pad_id, bos_id, vocab[L_HI], vocab[L_TE]}
    beams = [([bos_id, lang_id], 0.0)]
    max_steps = max(1, min(max_new_tokens, cfg.max_len - 2))
    for _ in range(max_steps):
        new_beams = []
        for seq, score in beams:
            if seq[-1] == eos_id:
                new_beams.append((seq, score))
                continue

            tgt = torch.tensor([seq], dtype=torch.long, device=device)
            tgt_mask = causal_mask(tgt.size(1), device=device)
            tgt_pad_mask = tgt.eq(pad_id)
            logits = model(
                src_ids=src,
                tgt_in_ids=tgt,
                tgt_mask=tgt_mask,
                src_key_padding_mask=src_pad_mask,
                tgt_key_padding_mask=tgt_pad_mask,
            )[0, -1]

            for bid in ban_ids:
                logits[bid] = -1e9

            for tok in set(seq):
                if tok not in ban_ids and tok < logits.size(0):
                    logits[tok] = logits[tok] - 0.5

            log_probs = torch.log_softmax(logits, dim=-1)
            topk = torch.topk(log_probs, k=beam_size)
            for lp, tid in zip(topk.values.tolist(), topk.indices.tolist()):
                cand = seq + [int(tid)]
                cand_score = score + float(lp)
                if len(cand) >= 4 and len(set(cand[-4:])) == 1:
                    cand_score -= 2.0
                new_beams.append((cand, cand_score))

        beams = sorted(
            new_beams,
            key=lambda x: x[1] / ((len(x[0]) ** 0.7) if len(x[0]) > 1 else 1.0),
            reverse=True,
        )[:beam_size]

        if all(seq[-1] == eos_id for seq, _ in beams):
            break

    best = beams[0][0]
    return tokenizer.decode(best, skip_special_tokens=True)


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

    out = translate(
        model,
        tokenizer,
        cfg,
        args.text,
        args.lang,
        device,
        max_new_tokens=args.max_new_tokens,
        beam_size=args.beam_size,
    )
    print(out)


if __name__ == "__main__":
    main()
