"""Архитектуры из блокнота 03_experiments.ipynb.

E0–E2: BaselineSiamese — базовый двухвходовый энкодер/декодер.
E3–E4: SiameseUNet — двухвходовая модель со skip connections.

Дисклеймер: обе модели возвращают логиты (без Sigmoid). Так устроены
модели в экспериментах третьего этапа: BCE With LogitsLoss для обучения,
Sigmoid только при вычислении вероятностей. Имена/формы параметров
сохранены для загрузки файлов best.pt, созданных исходным блокнотом.
"""

import torch
from torch import nn


class BaselineSiamese(nn.Module):
    """Упрощённая сиамская модель из базового пайплайна, без Sigmoid."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat([self.encoder(a), self.encoder(b)], dim=1))


class ConvBlock(nn.Module):
    """Две свёртки 3×3 с BatchNorm и ReLU."""

    def __init__(self, inp: int, out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(inp, out, 3, padding=1, bias=False),
            nn.BatchNorm2d(out),
            nn.ReLU(True),
            nn.Conv2d(out, out, 3, padding=1, bias=False),
            nn.BatchNorm2d(out),
            nn.ReLU(True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SiameseUNet(nn.Module):
    """Siamese U-Net со skip connections и объединением concat/absdiff."""

    def __init__(self, fusion: str = "concat", base_channels: int = 16):
        super().__init__()
        if fusion not in ("concat", "absdiff"):
            raise ValueError("fusion должен быть 'concat' или 'absdiff'")
        self.fusion = fusion
        c = base_channels
        factor = 2 if fusion == "concat" else 1
        self.enc1 = ConvBlock(3, c)
        self.enc2 = ConvBlock(c, 2 * c)
        self.enc3 = ConvBlock(2 * c, 4 * c)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(factor * 4 * c, 2 * c, 2, stride=2)
        self.dec2 = ConvBlock(2 * c + factor * 2 * c, 2 * c)
        self.up1 = nn.ConvTranspose2d(2 * c, c, 2, stride=2)
        self.dec1 = ConvBlock(c + factor * c, c)
        self.head = nn.Conv2d(c, 1, 1)

    def encode(self, x: torch.Tensor):
        f1 = self.enc1(x)
        f2 = self.enc2(self.pool(f1))
        f3 = self.enc3(self.pool(f2))
        return f1, f2, f3

    def fuse(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.fusion == "concat":
            return torch.cat([x, y], dim=1)
        return torch.abs(x - y)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a1, a2, a3 = self.encode(a)
        b1, b2, b3 = self.encode(b)
        x = self.dec2(torch.cat([self.up2(self.fuse(a3, b3)), self.fuse(a2, b2)], dim=1))
        x = self.dec1(torch.cat([self.up1(x), self.fuse(a1, b1)], dim=1))
        return self.head(x)


def build_model(config: dict) -> nn.Module:
    """Создаёт модель по полям architecture/fusion из E0–E4 или best_model.json."""
    architecture = config["architecture"]
    if architecture == "baseline":
        return BaselineSiamese()
    if architecture == "unet":
        return SiameseUNet(
            fusion=config.get("fusion", "concat"),
            base_channels=int(config.get("base_channels", 16)),
        )
    raise ValueError(f"Неизвестная архитектура: {architecture}")
