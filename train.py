import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import TrainConfig
from .data import load_parallel_data, split_items, train_or_load_tokenizer, TranslationDataset
from .model import HybridTranslator
from .utils import PAD, causal_mask, set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max_len", type=int, default=64)
    p.add_argument("--vocab_size", type=int, default=16000)
    p.add_argument("--rebuild_tokenizer", action="store_true")
    p.add_argument("--target_lang", type=str, default="both", choices=["both", "hi", "te"])
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--warmup_steps", type=int, default=2000)
    p.add_argument("--artifacts_dir", type=str, default="artifacts")
    return p.parse_args()


def train_one_epoch(model, loader, optimizer, scheduler, criterion, pad_id, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="train", leave=False):
        src = batch["src"].to(device)
        tgt_in = batch["tgt_in"].to(device)
        tgt_out = batch["tgt_out"].to(device)

        src_pad_mask = src.eq(pad_id)
        tgt_pad_mask = tgt_in.eq(pad_id)
        tgt_m = causal_mask(tgt_in.size(1), device=device)

        logits = model(
            src_ids=src,
            tgt_in_ids=tgt_in,
            tgt_mask=tgt_m,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
        )
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    return total_loss / max(1, len(loader))


@torch.no_grad()
def validate(model, loader, criterion, pad_id, device):
    model.eval()
    total_loss = 0.0
    for batch in loader:
        src = batch["src"].to(device)
        tgt_in = batch["tgt_in"].to(device)
        tgt_out = batch["tgt_out"].to(device)

        src_pad_mask = src.eq(pad_id)
        tgt_pad_mask = tgt_in.eq(pad_id)
        tgt_m = causal_mask(tgt_in.size(1), device=device)

        logits = model(
            src_ids=src,
            tgt_in_ids=tgt_in,
            tgt_mask=tgt_m,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
        )
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        total_loss += loss.item()
    return total_loss / max(1, len(loader))


def main():
    args = parse_args()
    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_len=args.max_len,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        warmup_steps=args.warmup_steps,
        use_quantum=False,
        artifacts_dir=args.artifacts_dir,
    )
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    items = load_parallel_data(args.data_path)
    if args.target_lang != "both":
        items = [x for x in items if x.target_lang == args.target_lang]
        print(f"Filtered to target_lang={args.target_lang} | rows={len(items)}")
    if len(items) < 100:
        raise ValueError("Not enough rows after filtering. Use a larger dataset or target_lang=both.")
    train_items, val_items = split_items(items, seed=cfg.seed)

    artifacts = Path(cfg.artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)
    tokenizer_path = artifacts / "tokenizer.json"
    tok = train_or_load_tokenizer(
        train_items + val_items,
        str(tokenizer_path),
        min_freq=cfg.min_freq,
        vocab_size=args.vocab_size,
        rebuild=args.rebuild_tokenizer,
    )
    vocab = tok.get_vocab()
    pad_id = vocab[PAD]

    train_ds = TranslationDataset(train_items, tok, cfg.max_len)
    val_ds = TranslationDataset(val_items, tok, cfg.max_len)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    model = HybridTranslator(
        vocab_size=tok.get_vocab_size(),
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = max(1, len(train_loader) * cfg.epochs)
    warmup_steps = min(cfg.warmup_steps, max(1, total_steps // 5))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        remain = total_steps - warmup_steps
        progress = (step - warmup_steps) / float(max(1, remain))
        return max(0.1, 1.0 - progress)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=cfg.label_smoothing)

    best_val = float("inf")
    for epoch in range(1, cfg.epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, pad_id, device)
        va = validate(model, val_loader, criterion, pad_id, device)
        print(f"Epoch {epoch}/{cfg.epochs} | train_loss={tr:.4f} | val_loss={va:.4f}")
        if va < best_val:
            best_val = va
            torch.save(model.state_dict(), artifacts / "model.pt")
            print("Saved best checkpoint.")

    cfg.save(str(artifacts / "config.json"))
    print(f"Artifacts saved to: {artifacts.resolve()}")


if __name__ == "__main__":
    main()
