"""Пиксельные метрики и внешняя оценка S2Looking без дообучения.

Использует функции и формат таблиц из 03_experiments.ipynb и
04_external_evaluation.ipynb. Тестовая разметка используется ТОЛЬКО
для расчёта метрик: веса и порог уже зафиксированы на LEVIR-CD.

Запуск:
    python src/evaluate.py --s2-root data/S2Looking --results-dir results/LEVIR_experiments_results
"""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

if __package__:
    from .inference import load_trained_model, predict_large, read_rgb
else:
    from inference import load_trained_model, predict_large, read_rgb

EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def pixel_counts(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Порядок: TP, FP, FN, TN."""
    pred, truth = np.asarray(pred, dtype=bool), np.asarray(truth, dtype=bool)
    if pred.shape != truth.shape:
        raise ValueError("Размеры предсказания и истинной маски не совпадают")
    return np.array([
        np.count_nonzero(pred & truth),
        np.count_nonzero(pred & ~truth),
        np.count_nonzero(~pred & truth),
        np.count_nonzero(~pred & ~truth),
    ], dtype=np.int64)


def scores(tp, fp, fn, tn) -> dict:
    """Micro-averaged по всем пикселям, как в блокноте 03."""
    tp, fp, fn, tn = [int(n) for n in (tp, fp, fn, tn)]
    def divide(x, y):
        return float(x / y) if y else 0.0
    return {
        "precision": divide(tp, tp + fp),
        "recall": divide(tp, tp + fn),
        "f1": divide(2 * tp, 2 * tp + fp + fn),
        "iou": divide(tp, tp + fp + fn),
        "accuracy": divide(tp + tn, tp + fp + fn + tn),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
    }


@torch.inference_mode()
def evaluate_loader(model, loader, *, device=None, threshold=0.5, per_patch=False):
    """Сегментация патчей LEVIR-CD, возвращает (общие метрики, DataFrame)."""
    device = torch.device(device or next(model.parameters()).device)
    model.eval()
    total = np.zeros(4, dtype=np.int64)
    rows = []
    for a, b, mask, names in loader:
        a, b, mask = a.to(device), b.to(device), mask.to(device)
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            probability = torch.sigmoid(model(a, b).float())
        pred = (probability >= threshold).cpu().numpy()
        truth = (mask >= 0.5).cpu().numpy()
        for index, name in enumerate(names):
            c = pixel_counts(pred[index], truth[index])
            total += c
            if per_patch:
                rows.append({"file": name, **scores(*c)})
    return scores(*total), pd.DataFrame(rows)


def resolve_s2looking_test(s2_root: Path) -> Path:
    s2_root = Path(s2_root)
    for base in (s2_root / "test", s2_root / "S2Looking" / "test", s2_root):
        if all((base / part).is_dir() for part in ("Image1", "Image2", "label")):
            return base
    raise FileNotFoundError(f"Не найдены папки test/Image1, Image2, label в {s2_root}")


def load_s2_pair(test_dir: Path, name: str):
    a = read_rgb(test_dir / "Image1" / name)
    b = read_rgb(test_dir / "Image2" / name)
    mask = cv2.imread(str(test_dir / "label" / name), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Не прочитана маска: {test_dir / 'label' / name}")
    if a.shape != b.shape or a.shape[:2] != mask.shape:
        raise ValueError(f"Не совпадают размеры A/B/label: {name}")
    truth = (mask > 0) if mask.max() <= 1 else (mask > 127)
    return a, b, truth


def error_map(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """TP зелёным, FP красным, FN синим, остальное чёрным."""
    out = np.zeros((*truth.shape, 3), dtype=np.uint8)
    out[truth & pred] = (30, 190, 30)
    out[~truth & pred] = (230, 50, 50)
    out[truth & ~pred] = (50, 100, 240)
    return out


def plot_external_examples(rows: pd.DataFrame, test_dir, model, meta, device, output, *, batch_size=8):
    positive = rows.loc[rows["changed_pixels"] > 0]
    if positive.empty:
        print("Нет изображений с изменениями — график примеров не создаётся")
        return
    selected = pd.concat((positive.nlargest(2, "f1"), positive.nsmallest(2, "f1")))
    selected = selected.drop_duplicates("file")
    fig, axes = plt.subplots(len(selected), 5, figsize=(16, 3.4 * len(selected)), squeeze=False)
    for row_idx, row in enumerate(selected.itertuples()):
        a, b, truth = load_s2_pair(test_dir, row.file)
        pred = predict_large(model, a, b, device=device,
                             threshold=float(meta["threshold"]),
                             patch_size=int(meta["input_size"]), batch_size=batch_size)
        for j, im in enumerate((a, b, truth, pred, error_map(pred, truth))):
            axes[row_idx, j].imshow(im, cmap="gray" if j in (2, 3) else None)
            axes[row_idx, j].axis("off")
            if row_idx == 0:
                axes[row_idx, j].set_title(("До", "После", "Разметка", "Прогноз", "TP/FP/FN")[j])
        axes[row_idx, 0].set_ylabel(f"F1={row.f1:.3f}")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def evaluate_s2looking(s2_root, results_dir, *, max_pairs=None, batch_size=8,
                       min_positive_pixels=15, make_examples=True):
    """Проверяет ВСЕ пиксели S2Looking Test и дополнительную выборку патчей >=15.

    Не удаляет пустые участки из основной оценки. Сохраняет исходные
    названия cross_dataset_comparison.csv и external_examples.png.
    """
    results_dir = Path(results_dir)
    test_dir = resolve_s2looking_test(Path(s2_root))
    model, meta, device = load_trained_model(results_dir)
    output_dir = results_dir / "S2Looking_external_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p.name for p in (test_dir / "Image1").iterdir() if p.suffix.lower() in EXTS)
    if not files:
        raise ValueError("Папка Image1 пуста")
    for name in files:
        if not (test_dir / "Image2" / name).is_file() or not (test_dir / "label" / name).is_file():
            raise FileNotFoundError(f"Нет пары/разметки для {name}")
    if max_pairs is not None:
        if not 1 <= max_pairs <= len(files):
            raise ValueError(f"max_pairs должен быть в диапазоне 1..{len(files)}")
        rng = np.random.default_rng(42)
        files = sorted(rng.choice(files, size=max_pairs, replace=False).tolist())
    total = np.zeros(4, dtype=np.int64)
    matched = np.zeros(4, dtype=np.int64)
    matched_patches = 0
    per_image = []
    patch = int(meta["input_size"])
    for name in tqdm(files, desc="S2Looking Test"):
        a, b, truth = load_s2_pair(test_dir, name)
        pred = predict_large(model, a, b, device=device,
                             threshold=float(meta["threshold"]), patch_size=patch,
                             batch_size=batch_size)
        counts = pixel_counts(pred, truth)
        total += counts
        per_image.append({"file": name, "changed_pixels": int(truth.sum()), **scores(*counts)})
        # Сопоставление с исходным LEVIR-CD (отфильтрованные положительные патчи).
        h, w = truth.shape
        for y in range(0, h, patch):
            for x in range(0, w, patch):
                tile_truth = truth[y:y + patch, x:x + patch]
                tile_pred = pred[y:y + patch, x:x + patch]
                if tile_truth.size == patch * patch and int(tile_truth.sum()) >= min_positive_pixels:
                    matched += pixel_counts(tile_pred, tile_truth)
                    matched_patches += 1
    external = scores(*total)
    external_matched = scores(*matched)
    rows = pd.DataFrame(per_image)
    rows.to_csv(output_dir / "per_image_metrics.csv", index=False)
    details = {
        "dataset": "S2Looking Test", "num_image_pairs": len(files),
        "checkpoint": str(results_dir / meta["checkpoint"]),
        "threshold_from_levir": float(meta["threshold"]),
        "all_pixels": external,
        "matched_positive_patches": {"patches": matched_patches, "metrics": external_matched},
    }
    (output_dir / "external_metrics.json").write_text(
        json.dumps(details, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    levir_path = results_dir / "levir_test_metrics.json"
    if not levir_path.is_file():
        raise FileNotFoundError(f"Нет {levir_path}; требуется итоговое тестирование LEVIR-CD")
    levir = json.loads(levir_path.read_text(encoding="utf-8"))["metrics"]
    columns = ("precision", "recall", "f1", "iou", "accuracy")
    comparison = pd.DataFrame([
        {"Датасет": "LEVIR-CD Test (отфильтрованные патчи)", **{k: levir[k] for k in columns}},
        {"Датасет": f"S2Looking Test (все {len(files)} пар)", **{k: external[k] for k in columns}},
        {"Датасет": f"S2Looking Test (патчи >={min_positive_pixels} пикс. изменений; "
                     f"{matched_patches} патчей)", **{k: external_matched[k] for k in columns}},
    ])
    comparison.to_csv(output_dir / "cross_dataset_comparison.csv", index=False)
    if make_examples:
        plot_external_examples(rows, test_dir, model, meta, device,
                               output_dir / "external_examples.png", batch_size=batch_size)
    print(comparison.round(4).to_string(index=False))
    print(f"Разница F1 (все S2Looking − отфильтрованный LEVIR): "
          f"{(external['f1'] - levir['f1']) * 100:+.2f} п.п.")
    print("Учтите: доля фона в этих тестовых датасетах различается.")
    return details, comparison


def main():
    parser = argparse.ArgumentParser(description="Оценка зафиксированной модели на S2Looking Test")
    parser.add_argument("--s2-root", type=Path, required=True,
                        help="Папка S2Looking, содержащая test/Image1, Image2, label")
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="Папка с best_model.json и <E>/best.pt")
    parser.add_argument("--max-pairs", type=int, default=None,
                        help="Только для пробного запуска; по умолчанию все")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--no-examples", action="store_true")
    args = parser.parse_args()
    evaluate_s2looking(args.s2_root, args.results_dir,
                       max_pairs=args.max_pairs, batch_size=args.batch_size,
                       make_examples=not args.no_examples)


if __name__ == "__main__":
    main()
