from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from torch.utils.data import Dataset

from .utils import BOS, EOS, PAD, UNK, L_HI, L_TE, SPECIAL_TOKENS


@dataclass
class ParallelItem:
    source: str
    target: str
    target_lang: str


def load_parallel_data(path: str) -> List[ParallelItem]:
    df = pd.read_csv(path)
    required = {"source", "target", "target_lang"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")
    items = [
        ParallelItem(str(r.source).strip(), str(r.target).strip(), str(r.target_lang).strip())
        for r in df.itertuples(index=False)
    ]
    return [x for x in items if x.source and x.target and x.target_lang in {"hi", "te"}]


def _iter_for_vocab(items: List[ParallelItem]):
    for it in items:
        yield it.source
        yield it.target


def train_or_load_tokenizer(
    items: List[ParallelItem],
    tokenizer_path: str,
    min_freq: int = 2,
    vocab_size: int = 16000,
    rebuild: bool = False,
) -> Tokenizer:
    path = Path(tokenizer_path)
    if path.exists() and not rebuild:
        return Tokenizer.from_file(str(path))

    tokenizer = Tokenizer(BPE(unk_token=UNK))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        special_tokens=SPECIAL_TOKENS,
        min_frequency=min_freq,
        vocab_size=vocab_size,
    )
    tokenizer.train_from_iterator(_iter_for_vocab(items), trainer)
    path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(path))
    return tokenizer


class TranslationDataset(Dataset):
    def __init__(self, items: List[ParallelItem], tokenizer: Tokenizer, max_len: int):
        self.items = items
        self.tok = tokenizer
        self.max_len = max_len
        vocab = tokenizer.get_vocab()
        self.pad_id = vocab[PAD]
        self.bos_id = vocab[BOS]
        self.eos_id = vocab[EOS]
        self.hi_id = vocab[L_HI]
        self.te_id = vocab[L_TE]

    def __len__(self):
        return len(self.items)

    def _pad(self, ids: List[int]) -> List[int]:
        ids = ids[: self.max_len]
        return ids + [self.pad_id] * (self.max_len - len(ids))

    def __getitem__(self, idx: int):
        it = self.items[idx]
        lang_tok_id = self.hi_id if it.target_lang == "hi" else self.te_id

        src_ids = self.tok.encode(it.source).ids
        tgt_ids = self.tok.encode(it.target).ids

        enc = [self.bos_id, lang_tok_id] + src_ids + [self.eos_id]
        dec_in = [self.bos_id, lang_tok_id] + tgt_ids
        dec_out = tgt_ids + [self.eos_id]

        enc = self._pad(enc)
        dec_in = self._pad(dec_in)
        dec_out = self._pad(dec_out)

        return {
            "src": torch.tensor(enc, dtype=torch.long),
            "tgt_in": torch.tensor(dec_in, dtype=torch.long),
            "tgt_out": torch.tensor(dec_out, dtype=torch.long),
        }


def split_items(items: List[ParallelItem], test_size: float = 0.2, seed: int = 42) -> Tuple[List[ParallelItem], List[ParallelItem]]:
    stratify = [x.target_lang for x in items] if len(items) > 1 else None
    train, val = train_test_split(
        items,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    return train, val
