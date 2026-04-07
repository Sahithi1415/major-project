from dataclasses import dataclass, asdict
import json
from pathlib import Path


@dataclass
class TrainConfig:
    max_len: int = 64
    d_model: int = 128
    lstm_hidden: int = 128
    nhead: int = 4
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    dim_ff: int = 256
    dropout: float = 0.1
    quantum_qubits: int = 4
    quantum_layers: int = 2
    use_quantum: bool = True
    batch_size: int = 8
    lr: float = 3e-4
    weight_decay: float = 0.01
    label_smoothing: float = 0.1
    warmup_steps: int = 2000
    epochs: int = 10
    min_freq: int = 1
    seed: int = 42
    artifacts_dir: str = "artifacts"

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @staticmethod
    def load(path: str) -> "TrainConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return TrainConfig(**data)
