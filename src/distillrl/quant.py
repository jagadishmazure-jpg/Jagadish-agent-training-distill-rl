"""Local serving: a small torch MLP served in fp32 and in dynamic int8.

`torch.ao.quantization.quantize_dynamic` stores Linear weights as int8 and quantises
activations on the fly. We measure accuracy on the gold set, serialized size and single-request
CPU latency for both versions.
"""

from __future__ import annotations

import io

import numpy as np
import torch
from torch import nn

from .data import TOOLS, Example
from .student import teacher_labels, vectorizer
from .teacher import Teacher

IDX = {t: i for i, t in enumerate(TOOLS)}


class ToolMLP(nn.Module):
    def __init__(self, n_in: int, hidden: int = 512, n_out: int = len(TOOLS)) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_out)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def features(texts: list[str]) -> torch.Tensor:
    return torch.from_numpy(vectorizer().transform(texts).toarray().astype(np.float32))


def train_fp32(t: Teacher, pool: list[Example], epochs: int = 8, seed: int = 0) -> ToolMLP:
    if epochs < 0:
        raise ValueError("epochs must not be negative")
    torch.manual_seed(seed)
    texts, labels, _ = teacher_labels(t, pool)
    x, y = features(texts), torch.tensor([IDX[lab] for lab in labels])
    model = ToolMLP(x.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(len(x), generator=g)
        for i in range(0, len(x), 64):
            b = perm[i : i + 64]
            opt.zero_grad()
            nn.functional.cross_entropy(model(x[b]), y[b]).backward()
            opt.step()
    return model.eval()


def to_int8(model: ToolMLP) -> nn.Module:
    return torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)


def size_bytes(model: nn.Module) -> int:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes


@torch.no_grad()
def predict(model: nn.Module, x: torch.Tensor) -> list[str]:
    return [TOOLS[i] for i in model(x).argmax(dim=1).tolist()]


@torch.no_grad()
def max_logit_gap(a: nn.Module, b: nn.Module, x: torch.Tensor) -> float:
    return float((a(x) - b(x)).abs().max())
