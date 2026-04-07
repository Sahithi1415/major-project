import math
from typing import Optional

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class QuantumEmbedding(nn.Module):
    def __init__(self, d_model: int, n_qubits: int = 4, q_layers: int = 2, use_quantum: bool = True):
        super().__init__()
        self.d_model = d_model
        self.n_qubits = n_qubits
        self.q_layers = q_layers
        self.use_quantum = use_quantum

        self.in_proj = nn.Linear(d_model, n_qubits)
        self.out_proj = nn.Linear(n_qubits, d_model)
        self.classical_fallback = nn.Sequential(
            nn.Linear(n_qubits, n_qubits),
            nn.Tanh(),
            nn.Linear(n_qubits, n_qubits),
        )
        self.q_layer = None

        if use_quantum:
            try:
                import pennylane as qml

                dev = qml.device("default.qubit", wires=n_qubits)

                @qml.qnode(dev, interface="torch")
                def circuit(inputs, weights):
                    for i in range(n_qubits):
                        qml.RY(inputs[i], wires=i)
                    for l in range(q_layers):
                        for i in range(n_qubits):
                            qml.RZ(weights[l, i], wires=i)
                            qml.RX(weights[l, i], wires=i)
                        for i in range(n_qubits - 1):
                            qml.CNOT(wires=[i, i + 1])
                    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

                weight_shapes = {"weights": (q_layers, n_qubits)}
                self.q_layer = qml.qnn.TorchLayer(circuit, weight_shapes)
            except Exception:
                self.q_layer = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        z = self.in_proj(x).reshape(b * t, self.n_qubits)

        if self.q_layer is not None:
            q_out = torch.stack([self.q_layer(z_i) for z_i in z], dim=0)
        else:
            q_out = self.classical_fallback(z)

        q_out = q_out.reshape(b, t, self.n_qubits)
        return self.out_proj(q_out)


class HybridTranslator(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        d_model: int = 128,
        lstm_hidden: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_ff: int = 256,
        dropout: float = 0.1,
        quantum_qubits: int = 4,
        quantum_layers: int = 2,
        use_quantum: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, max_len=max_len)
        self.quantum = QuantumEmbedding(
            d_model=d_model,
            n_qubits=quantum_qubits,
            q_layers=quantum_layers,
            use_quantum=use_quantum,
        )
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm_proj = nn.Linear(2 * lstm_hidden, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
        )
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)
        self.out = nn.Linear(d_model, vocab_size)

    def encode(self, src_ids: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.embed(src_ids) * math.sqrt(self.d_model)
        x = self.pos(x)
        x = x + self.quantum(x)
        lstm_out, _ = self.lstm(x)
        enc_in = self.lstm_proj(lstm_out)
        memory = self.encoder(enc_in, src_key_padding_mask=src_key_padding_mask)
        return memory

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_in_ids: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        memory = self.encode(src_ids, src_key_padding_mask=src_key_padding_mask)
        tgt = self.embed(tgt_in_ids) * math.sqrt(self.d_model)
        tgt = self.pos(tgt)
        dec = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return self.out(dec)
