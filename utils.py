import random
import numpy as np
import torch


PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"
L_HI = "<2hi>"
L_TE = "<2te>"
SPECIAL_TOKENS = [PAD, BOS, EOS, UNK, L_HI, L_TE]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def causal_mask(size: int, device: torch.device) -> torch.Tensor:
    mask = torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()
    return mask
