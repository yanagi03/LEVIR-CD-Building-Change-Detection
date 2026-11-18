"""Нарезка LEVIR-CD на 256×256 по блокноту 01_data_preparation.ipynb.

Воспроизводит имена "<имя>_r<строка>_c<столбец>.png" и фильтр
min_building_pixels=15. Использует готовые исходные train/val/test;
не перемешивает снимки и не переносит их между выборками.

Запуск:
    python src/prepare_data.py --input-dir data/LEVIR-CD --output-dir data/LEVIR_processed_5K
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm.auto import tqdm

EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def crop_and_filter_dataset(
    root_dir,
    output_dir,
    splits=("train", "val", "test"),
    patch_size: int = 256,
    min_building_pixels: int = 15,
):
    """Сохраняет полные RGB-патчи A/B и бинарные маски в том же split.

    Патчи с менее чем ``min_building_pixels`` пикселями изменений
    исключаются, как в исходном блокноте. Частичные краевые патчи
    не сохраняются: модель ожидает строго patch_size×patch_size.
    """
    root_dir, output_dir = Path(root_dir), Path(output_dir)
    if patch_size <= 0 or min_building_pixels < 0:
        raise ValueError("patch_size > 0 и min_building_pixels >= 0 обязательны")
    summary = {}
    for split in splits:
        image_a = root_dir / split / "A"
        image_b = root_dir / split / "B"
        labels = root_dir / split / "label"
        if not all(d.is_dir() for d in (image_a, image_b, labels)):
            raise FileNotFoundError(f"Ожидаются {split}/A, {split}/B, {split}/label в {root_dir}")
        output_paths = {name: output_dir / split / name for name in ("A", "B", "label")}
        for path in output_paths.values():
            path.mkdir(parents=True, exist_ok=True)
        filenames = sorted(p.name for p in image_a.iterdir() if p.suffix.lower() in EXTS)
        if not filenames:
            raise ValueError(f"Нет снимков: {image_a}")
        stats = {"source_pairs": len(filenames), "saved_patches": 0,
                 "skipped_low_change": 0, "skipped_partial": 0, "skipped_unreadable": 0}
        for fname in tqdm(filenames, desc=f"Нарезка {split}"):
            a = cv2.imread(str(image_a / fname), cv2.IMREAD_COLOR)
            b = cv2.imread(str(image_b / fname), cv2.IMREAD_COLOR)
            label = cv2.imread(str(labels / fname), cv2.IMREAD_GRAYSCALE)
            if a is None or b is None or label is None:
                stats["skipped_unreadable"] += 1
                print(f"Предупреждение: не прочитана пара или маска {split}/{fname}")
                continue
            if a.shape[:2] != b.shape[:2] or a.shape[:2] != label.shape:
                raise ValueError(f"Не совпадают размеры A/B/label: {split}/{fname}")
            h, w = label.shape
            for y in range(0, h, patch_size):
                for x in range(0, w, patch_size):
                    if y + patch_size > h or x + patch_size > w:
                        stats["skipped_partial"] += 1
                        continue
                    # Маски LEVIR-CD: >127 означает изменения; выход — строго 0/255.
                    mask = np.where(
                        label[y:y + patch_size, x:x + patch_size] > 127, 255, 0
                    ).astype(np.uint8)
                    if int(np.count_nonzero(mask)) < min_building_pixels:
                        stats["skipped_low_change"] += 1
                        continue
                    name = f"{Path(fname).stem}_r{y}_c{x}.png"
                    a_patch = a[y:y + patch_size, x:x + patch_size]
                    b_patch = b[y:y + patch_size, x:x + patch_size]
                    for folder, patch in (("A", a_patch), ("B", b_patch), ("label", mask)):
                        if not cv2.imwrite(str(output_paths[folder] / name), patch):
                            raise OSError(f"Не удалось сохранить: {output_paths[folder] / name}")
                    stats["saved_patches"] += 1
        summary[split] = stats
        print(f"{split}: {stats}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Нарезка LEVIR-CD на патчи с фильтрацией изменений")
    parser.add_argument("--input-dir", type=Path, required=True, help="Папка исходного LEVIR-CD с train/val/test")
    parser.add_argument("--output-dir", type=Path, required=True, help="Куда сохранить LEVIR_processed_5K")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--min-building-pixels", type=int, default=15,
                        help="Как в исходном ноутбуке; 0 — без фильтрации")
    args = parser.parse_args()
    if args.input_dir.resolve() == args.output_dir.resolve():
        parser.error("input-dir и output-dir должны различаться")
    crop_and_filter_dataset(args.input_dir, args.output_dir,
                            patch_size=args.patch_size,
                            min_building_pixels=args.min_building_pixels)


if __name__ == "__main__":
    main()
