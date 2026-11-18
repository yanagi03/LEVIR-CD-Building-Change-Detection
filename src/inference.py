"""Инференс для любой пары спутниковых RGB-снимков, без дообучения.

Из 04_external_evaluation.ipynb: разбиение на патчи 256×256, объединение
в полную маску, загрузка весов и порога из best_model.json.

Пример:
    python src/inference.py --image-a before.png --image-b after.png \
        --results-dir results/LEVIR_experiments_results --output prediction.png

Выход: бинарный PNG (0 — фон, 255 — изменения).
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

if __package__:
    from .models import build_model
else:
    from models import build_model


def load_trained_model(results_dir, device=None):
    """Загружает лучшие веса и метаданные, созданные train.py / блокнотом 03.

    results_dir содержит best_model.json и <E0..E4>/best.pt.
    Переобучения и подбора порога на тестовых данных здесь нет.
    """
    results_dir = Path(results_dir)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    path = results_dir / "best_model.json"
    if not path.is_file():
        raise FileNotFoundError(f"Нет {path}. Сначала обучите модели.")
    meta = json.loads(path.read_text(encoding="utf-8"))
    if meta.get("normalization") != "rgb_div_255":
        raise ValueError("Неподдерживаемая нормализация: ожидается rgb_div_255")
    patch_size = int(meta["input_size"])
    if patch_size <= 0 or patch_size % 4:
        raise ValueError("input_size должен быть положительным и кратным четырём")
    threshold = float(meta["threshold"])
    if not 0 <= threshold <= 1:
        raise ValueError("Порог должен быть в диапазоне [0, 1]")
    model = build_model(meta["experiment"]).to(device)
    checkpoint_path = (results_dir / meta["checkpoint"]).resolve()
    # Ссылка на checkpoint берётся из метаданных, созданных собственным обучением.
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Нет весов: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model, meta, device


@torch.inference_mode()
def predict_large(
    model,
    image_a: np.ndarray,
    image_b: np.ndarray,
    *,
    threshold: float,
    patch_size: int = 256,
    batch_size: int = 8,
    device=None,
) -> np.ndarray:
    """RGB uint8 A/B -> bool mask исходного размера без фильтрации фона.

    Неполные края дополняются отражением; потом размер восстанавливается.
    Точно как в блокноте проверки S2Looking, за исключением явных параметров.
    """
    if image_a.ndim != 3 or image_a.shape[2] != 3 or image_b.shape != image_a.shape:
        raise ValueError("A и B должны быть RGB-массивами H×W×3 одинакового размера")
    if patch_size <= 0 or patch_size % 4 or batch_size < 1:
        raise ValueError("patch_size должен быть кратен 4, batch_size >= 1")
    if image_a.dtype != np.uint8 or image_b.dtype != np.uint8:
        raise ValueError("Входные RGB-снимки должны иметь тип uint8 (0..255)")
    device = torch.device(device or next(model.parameters()).device)
    h, w = image_a.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("Пустое изображение")
    dh, dw = (-h) % patch_size, (-w) % patch_size
    # OpenCV допускает произвольную ширину отражения для изображений >1 пикселя.
    border_mode = cv2.BORDER_REFLECT_101 if min(h, w) > 1 else cv2.BORDER_REPLICATE
    a = cv2.copyMakeBorder(image_a, 0, dh, 0, dw, border_mode)
    b = cv2.copyMakeBorder(image_b, 0, dh, 0, dw, border_mode)
    hh, ww = a.shape[:2]
    coords = [(y, x) for y in range(0, hh, patch_size) for x in range(0, ww, patch_size)]
    result = np.zeros((hh, ww), dtype=np.uint8)
    model.eval()
    for start in range(0, len(coords), batch_size):
        batch_coords = coords[start:start + batch_size]
        aa = np.stack([a[y:y + patch_size, x:x + patch_size] for y, x in batch_coords])
        bb = np.stack([b[y:y + patch_size, x:x + patch_size] for y, x in batch_coords])
        ta = torch.from_numpy(np.ascontiguousarray(aa.transpose(0, 3, 1, 2))).float().to(device) / 255.0
        tb = torch.from_numpy(np.ascontiguousarray(bb.transpose(0, 3, 1, 2))).float().to(device) / 255.0
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            probabilities = torch.sigmoid(model(ta, tb).float())[:, 0]
        patches = (probabilities >= threshold).byte().cpu().numpy()
        for (y, x), patch in zip(batch_coords, patches):
            result[y:y + patch_size, x:x + patch_size] = patch
    return result[:h, :w].astype(bool)


def read_rgb(path) -> np.ndarray:
    """Читает изображение через OpenCV, возвращает RGB uint8."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Не удалось прочитать изображение: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def main():
    parser = argparse.ArgumentParser(description="Сегментация изменений между двумя снимками")
    parser.add_argument("--image-a", type=Path, required=True, help="Снимок до изменений")
    parser.add_argument("--image-b", type=Path, required=True, help="Снимок после изменений")
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="Папка с best_model.json и <E>/best.pt")
    parser.add_argument("--output", type=Path, required=True, help="Имя PNG для бинарной маски")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    model, meta, device = load_trained_model(args.results_dir)
    a, b = read_rgb(args.image_a), read_rgb(args.image_b)
    pred = predict_large(
        model, a, b,
        threshold=float(meta["threshold"]),
        patch_size=int(meta["input_size"]),
        batch_size=args.batch_size,
        device=device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), pred.astype(np.uint8) * 255):
        raise OSError(f"Не удалось записать {args.output}")
    print(f"Бинарная маска сохранена: {args.output} ({pred.shape[1]}×{pred.shape[0]})")


if __name__ == "__main__":
    main()
