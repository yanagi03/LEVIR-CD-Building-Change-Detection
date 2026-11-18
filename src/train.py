"""Обучение пяти конфигураций E0–E4 на LEVIR-CD.

Рефакторинг 03_experiments.ipynb. Архитектуры из models.py,
общие метрики из evaluate.py. Сохраняет те же имена файлов и формат
checkpoint: <E>/best.pt (ключ model_state), best_model.json и таблицы.

Запуск (из корня репозитория; Colab Drive уже должен быть подключён):
    python src/train.py --data-dir data/LEVIR_processed_5K --results-dir outputs/LEVIR_experiments_results

Можно передать --config-dir configs с файлами E0.json...E4.json;
без конфигов будут использованы настройки исходного блокнота.
"""

import argparse
import json
import os
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

if __package__:
    from .models import build_model
    from .evaluate import evaluate_loader, error_map
else:
    from models import build_model
    from evaluate import evaluate_loader, error_map

EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
EXPERIMENTS = [
    {"id": "E0", "architecture": "baseline", "fusion": "concat", "loss": "bce", "augment": False},
    {"id": "E1", "architecture": "baseline", "fusion": "concat", "loss": "bce_dice", "augment": False},
    {"id": "E2", "architecture": "baseline", "fusion": "concat", "loss": "bce_dice", "augment": True},
    {"id": "E3", "architecture": "unet", "fusion": "concat", "loss": "bce_dice", "augment": True},
    {"id": "E4", "architecture": "unet", "fusion": "absdiff", "loss": "bce_dice", "augment": True},
]


@dataclass
class Settings:
    seed: int = 42
    epochs: int = 30
    min_epochs: int = 12
    patience: int = 7
    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 1e-4
    workers: int = 2
    patch_size: int = 256


def read_experiment_configs(config_dir: Path | None):
    """Читает E0–E4.json, включая экспортированные notebook config.json."""
    if config_dir is None:
        return [dict(cfg) for cfg in EXPERIMENTS], Settings()
    configs = []
    common_names = ("seed", "epochs", "min_epochs", "patience", "batch_size", "lr", "weight_decay", "patch_size")
    reference = None
    for expected in ("E0", "E1", "E2", "E3", "E4"):
        path = Path(config_dir) / f"{expected}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Нет файла конфигурации: {path}")
        obj = json.loads(path.read_text(encoding="utf-8"))
        if obj.get("id") != expected:
            raise ValueError(f"Неверный id в {path}")
        if reference is None:
            reference = {key: obj[key] for key in common_names if key in obj}
        else:
            for key, value in reference.items():
                if key in obj and obj[key] != value:
                    raise ValueError(f"Параметр {key} должен быть одинаковым для всех экспериментов")
        configs.append({key: obj[key] for key in
                        ("id", "architecture", "fusion", "loss", "augment")})
    settings = Settings()
    for key, value in reference.items():
        setattr(settings, key, value)
    return configs, settings


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_splits(data_dir: Path):
    names_by_split = {}
    originals = {}
    for split in ("train", "val", "test"):
        base = data_dir / split
        for part in ("A", "B", "label"):
            if not (base / part).is_dir():
                raise FileNotFoundError(f"Нет {base / part}")
        names = sorted(p.name for p in (base / "A").iterdir() if p.suffix.lower() in EXTS)
        if not names:
            raise ValueError(f"Пустая выборка: {base}")
        for fname in names:
            if not (base / "B" / fname).is_file() or not (base / "label" / fname).is_file():
                raise FileNotFoundError(f"Нет B/label для {split}/{fname}")
        names_by_split[split] = names
        originals[split] = {re.sub(r"_r\d+_c\d+$", "", Path(name).stem) for name in names}
        print(f"{split}: {len(names)} пар")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if originals[a] & originals[b]:
            raise ValueError(f"Утечка данных: одни и те же исходные снимки в {a} и {b}")
    return names_by_split


class LEVIRDataset(Dataset):
    """RGB / 255, mask > 127, синхронные повороты/отражения для E2–E4."""

    def __init__(self, data_dir, split, names, *, augment=False, patch_size=256):
        self.root = Path(data_dir) / split
        self.names = list(names)
        self.augment = augment and split == "train"
        self.patch_size = patch_size

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        a = cv2.imread(str(self.root / "A" / name), cv2.IMREAD_COLOR)
        b = cv2.imread(str(self.root / "B" / name), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(self.root / "label" / name), cv2.IMREAD_GRAYSCALE)
        if a is None or b is None or mask is None:
            raise ValueError(f"Ошибка чтения {name}")
        if a.shape[:2] != b.shape[:2] or a.shape[:2] != mask.shape:
            raise ValueError(f"Не совпадают размеры {name}")
        if a.shape[:2] != (self.patch_size, self.patch_size):
            raise ValueError(f"Ожидался патч {self.patch_size}×{self.patch_size}: {name}")
        a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
        b = cv2.cvtColor(b, cv2.COLOR_BGR2RGB)
        mask = (mask > 127).astype(np.float32)
        if self.augment:
            k = random.randrange(4)
            a, b, mask = np.rot90(a, k), np.rot90(b, k), np.rot90(mask, k)
            if random.random() < 0.5:
                a, b, mask = np.fliplr(a), np.fliplr(b), np.fliplr(mask)
            if random.random() < 0.5:
                a, b, mask = np.flipud(a), np.flipud(b), np.flipud(mask)
        ta = torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1))).float() / 255.0
        tb = torch.from_numpy(np.ascontiguousarray(b.transpose(2, 0, 1))).float() / 255.0
        tm = torch.from_numpy(np.ascontiguousarray(mask[None])).float()
        return ta, tb, tm, name


def make_loader(data_dir, split, names, settings, *, augment=False, device=None):
    return DataLoader(
        LEVIRDataset(data_dir, split, names, augment=augment, patch_size=settings.patch_size),
        batch_size=settings.batch_size,
        shuffle=(split == "train"),
        num_workers=settings.workers,
        pin_memory=(torch.device(device).type == "cuda") if device else False,
        persistent_workers=settings.workers > 0,
    )


class BCEDiceLoss(nn.Module):
    """Та же сумма BCEWithLogits + (1 - Dice), что в блокноте 03."""

    def __init__(self, dice=False):
        super().__init__()
        self.dice = dice
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, target):
        loss = self.bce(logits, target)
        if self.dice:
            probability = torch.sigmoid(logits.float())
            dice = (2 * (probability * target).sum() + 1) / (
                probability.sum() + target.sum() + 1
            )
            loss = loss + 1 - dice
        return loss


def write_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def atomic_save(obj, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def run_experiment(cfg, data_dir, results_dir, names, settings, device, val_loader):
    folder = Path(results_dir) / cfg["id"]
    folder.mkdir(parents=True, exist_ok=True)
    spec = {
        **cfg,
        "epochs": settings.epochs,
        "min_epochs": settings.min_epochs,
        "patience": settings.patience,
        "batch_size": settings.batch_size,
        "lr": settings.lr,
        "weight_decay": settings.weight_decay,
        "seed": settings.seed,
    }
    spec_path = folder / "config.json"
    if spec_path.is_file() and json.loads(spec_path.read_text(encoding="utf-8")) != spec:
        raise ValueError(f"{cfg['id']}: изменились настройки. Используйте новую папку результатов.")
    write_json(spec, spec_path)
    summary_path = folder / "summary.json"
    if summary_path.is_file():
        if not (folder / "best.pt").is_file() or not (folder / "history.csv").is_file():
            raise FileNotFoundError(f"{cfg['id']}: результаты неполные (нет best.pt/history.csv)")
        print(f"{cfg['id']} уже завершён — используется сохранённый результат")
        return json.loads(summary_path.read_text(encoding="utf-8"))

    set_seed(settings.seed)
    model = build_model(cfg).to(device)
    criterion = BCEDiceLoss(dice=(cfg["loss"] == "bce_dice"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.lr,
                                  weight_decay=settings.weight_decay)
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    train_loader = make_loader(data_dir, "train", names["train"], settings,
                               augment=cfg["augment"], device=device)
    history, best, best_epoch, bad_epochs, first = [], -1.0, 0, 0, 1
    last_path = folder / "last.pt"
    if last_path.is_file():
        # Собственные checkpoint из train.py (содержат состояния NumPy/Python RNG).
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        history = state["history"]
        best, best_epoch, bad_epochs, first = (
            state["best"], state["best_epoch"], state["bad_epochs"], state["epoch"] + 1
        )
        if "random_state" in state:
            random.setstate(state["random_state"])
        if "numpy_state" in state:
            np.random.set_state(state["numpy_state"])
        if "torch_rng" in state:
            torch.set_rng_state(state["torch_rng"].cpu())
        if amp and "cuda_rng" in state:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        print(f"{cfg['id']}: возобновление с эпохи {first}")

    for epoch in range(first, settings.epochs + 1):
        if epoch - 1 >= settings.min_epochs and bad_epochs >= settings.patience:
            break
        model.train()
        loss_total = 0.0
        for a, b, mask, _ in tqdm(train_loader, desc=f"{cfg['id']} {epoch}/{settings.epochs}", leave=False):
            a, b, mask = a.to(device), b.to(device), mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                loss = criterion(model(a, b), mask)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_total += float(loss.item()) * len(a)
        val, _ = evaluate_loader(model, val_loader, device=device, threshold=0.5)
        epoch_loss = loss_total / len(train_loader.dataset)
        history.append({"epoch": epoch, "train_loss": epoch_loss,
                        **{f"val_{k}": val[k] for k in
                           ("precision", "recall", "f1", "iou", "accuracy")}})
        if val["f1"] > best + 1e-6:
            best, best_epoch, bad_epochs = val["f1"], epoch, 0
            atomic_save({"model_state": model.state_dict(),
                         "experiment": cfg, "best_epoch": best_epoch}, folder / "best.pt")
        else:
            bad_epochs += 1
        pd.DataFrame(history).to_csv(folder / "history.csv", index=False)
        atomic_save({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(), "history": history,
            "best": best, "best_epoch": best_epoch, "bad_epochs": bad_epochs,
            "epoch": epoch, "random_state": random.getstate(),
            "numpy_state": np.random.get_state(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if amp else [],
        }, last_path)
        print(f"{cfg['id']}, эпоха {epoch:02d}: loss={epoch_loss:.4f}, "
              f"val_F1={val['f1']:.4f}, val_IoU={val['iou']:.4f}")
        if epoch >= settings.min_epochs and bad_epochs >= settings.patience:
            print(f"Early Stopping, лучшая эпоха: {best_epoch}")
            break

    checkpoint = torch.load(folder / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    val, _ = evaluate_loader(model, val_loader, device=device, threshold=0.5)
    summary = {
        **cfg, "trained_epochs": len(history), "best_epoch": best_epoch,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        **{f"val_{k}": val[k] for k in ("precision", "recall", "f1", "iou", "accuracy")},
    }
    write_json(summary, summary_path)
    return summary


def plot_learning_curves(configs, results_dir):
    fig, axes = plt.subplots(2, 1, figsize=(10, 9))
    for cfg in configs:
        h = pd.read_csv(Path(results_dir) / cfg["id"] / "history.csv")
        axes[0].plot(h["epoch"], h["val_f1"], label=cfg["id"])
        axes[1].plot(h["epoch"], h["train_loss"], label=cfg["id"])
    axes[0].set(title="Validation F1", xlabel="Эпоха", ylabel="F1")
    axes[1].set(title="Train Loss (разные функции потерь имеют разные шкалы)",
                xlabel="Эпоха", ylabel="Loss")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(Path(results_dir) / "training_curves.png", dpi=150)
    plt.close(fig)


def plot_levir_examples(data_dir, per_patch, model, threshold, device, results_dir):
    selected = pd.concat((per_patch.nlargest(3, "f1"), per_patch.nsmallest(3, "f1")))
    selected = selected.drop_duplicates("file")
    if selected.empty:
        return
    fig, axes = plt.subplots(len(selected), 5, figsize=(15, len(selected) * 2.8), squeeze=False)
    test_dir = Path(data_dir) / "test"
    for i, row in enumerate(selected.itertuples()):
        a = cv2.imread(str(test_dir / "A" / row.file), cv2.IMREAD_COLOR)
        b = cv2.imread(str(test_dir / "B" / row.file), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(test_dir / "label" / row.file), cv2.IMREAD_GRAYSCALE)
        a, b = cv2.cvtColor(a, cv2.COLOR_BGR2RGB), cv2.cvtColor(b, cv2.COLOR_BGR2RGB)
        truth = mask > 127
        ta = torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1)))[None].float().to(device) / 255.0
        tb = torch.from_numpy(np.ascontiguousarray(b.transpose(2, 0, 1)))[None].float().to(device) / 255.0
        with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            pred = (torch.sigmoid(model(ta, tb).float())[0, 0].cpu().numpy() >= threshold)
        for j, image in enumerate((a, b, truth, pred, error_map(pred, truth))):
            axes[i, j].imshow(image, cmap="gray" if j in (2, 3) else None)
            axes[i, j].axis("off")
            if i == 0:
                axes[i, j].set_title(("До", "После", "Разметка", "Прогноз", "TP/FP/FN")[j])
        axes[i, 0].set_ylabel(f"F1={row.f1:.3f}")
    fig.tight_layout()
    fig.savefig(Path(results_dir) / "levir_examples.png", dpi=150)
    plt.close(fig)


def train_all(data_dir, results_dir, *, configs=None, settings=None,
              experiment_ids=None, cache_local=False, skip_examples=False):
    """Запускает эксперименты и сохраняет результаты третьего этапа."""
    settings = settings or Settings()
    configs = configs or [dict(cfg) for cfg in EXPERIMENTS]
    if experiment_ids:
        names = set(experiment_ids)
        if names - {cfg["id"] for cfg in configs}:
            raise ValueError(f"Неизвестные эксперименты: {names}")
        configs = [cfg for cfg in configs if cfg["id"] in names]
    if settings.patch_size % 4 or settings.patch_size <= 0:
        raise ValueError("Размер патча должен быть положительным и кратным четырём")
    if settings.batch_size < 1 or settings.epochs < 1 or settings.patience < 1 or settings.workers < 0:
        raise ValueError("Недопустимые настройки batch_size/epochs/patience/workers")
    data_dir = Path(data_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Нет данных: {data_dir}")
    if cache_local:
        destination = Path("/content/LEVIR_processed_5K")
        if data_dir.resolve() != destination.resolve() and not destination.is_dir():
            print(f"Копирование набора данных в {destination}")
            shutil.copytree(data_dir, destination)
        if destination.is_dir():
            data_dir = destination
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("Внимание: используется CPU; в Colab включите GPU")
    set_seed(settings.seed)
    names = validate_splits(data_dir)
    val_loader = make_loader(data_dir, "val", names["val"], settings, device=device)
    test_loader = make_loader(data_dir, "test", names["test"], settings, device=device)
    comparisons = []
    for cfg in configs:
        print(f"\n{'=' * 50}\nЭксперимент {cfg['id']}: {cfg}")
        comparisons.append(run_experiment(cfg, data_dir, results_dir, names, settings, device, val_loader))
    table = pd.DataFrame(comparisons).sort_values(["val_f1", "val_iou"], ascending=False)
    table.to_csv(results_dir / "experiment_comparison.csv", index=False)
    print("\nСравнение на Validation:\n", table[["id", "val_f1", "val_iou", "best_epoch"]].to_string(index=False))
    plot_learning_curves(configs, results_dir)
    winner = table.iloc[0]
    best_config = next(x for x in configs if x["id"] == winner["id"])
    model = build_model(best_config).to(device)
    checkpoint = torch.load(results_dir / best_config["id"] / "best.pt",
                            map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    rows = []
    for value in np.arange(0.30, 0.71, 0.05):
        value = round(float(value), 2)
        metric, _ = evaluate_loader(model, val_loader, device=device, threshold=value)
        rows.append({"threshold": value, **{k: metric[k]
                       for k in ("f1", "iou", "precision", "recall")}})
    thresholds = pd.DataFrame(rows)
    thresholds.to_csv(results_dir / "threshold_search.csv", index=False)
    thresholds["distance"] = (thresholds["threshold"] - 0.5).abs()
    threshold = float(thresholds.sort_values(["f1", "distance"], ascending=[False, True]).iloc[0]["threshold"])

    # Тестовая выборка используется один раз, после выбора модели/порога на Validation.
    test, per_patch = evaluate_loader(model, test_loader, device=device, threshold=threshold, per_patch=True)
    per_patch.to_csv(results_dir / "levir_test_per_patch.csv", index=False)
    write_json({"dataset": "LEVIR-CD", "experiment": best_config["id"],
                "threshold": threshold, "metrics": test}, results_dir / "levir_test_metrics.json")
    write_json({"experiment": best_config,
                "checkpoint": f"{best_config['id']}/best.pt",
                "threshold": threshold, "input_size": settings.patch_size,
                "normalization": "rgb_div_255"}, results_dir / "best_model.json")
    if not skip_examples:
        plot_levir_examples(data_dir, per_patch, model, threshold, device, results_dir)
    print("\nЛучшая конфигурация:", best_config["id"], "| порог:", threshold)
    print("LEVIR-CD Test:", {k: round(test[k], 4) for k in ("precision", "recall", "f1", "iou", "accuracy")})
    print("Сохранено в:", results_dir)
    return table, test


def main():
    parser = argparse.ArgumentParser(description="Обучение E0–E4 на LEVIR-CD")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Готовый LEVIR_processed_5K/train|val|test/A|B|label")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, default=None,
                        help="Необязательно: каталог с E0.json ... E4.json")
    parser.add_argument("--experiments", nargs="+", choices=[c["id"] for c in EXPERIMENTS],
                        help="Только выбранные эксперименты (для пробного запуска)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Переопределяет число эпох из конфигураций")
    parser.add_argument("--min-epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--cache-local", action="store_true",
                        help="Скопировать данные в /content/LEVIR_processed_5K для ускорения Colab")
    parser.add_argument("--skip-examples", action="store_true")
    args = parser.parse_args()
    configs, settings = read_experiment_configs(args.config_dir)
    for argument, attr in (("epochs", "epochs"), ("min_epochs", "min_epochs"),
                           ("patience", "patience"), ("batch_size", "batch_size"),
                           ("workers", "workers"), ("patch_size", "patch_size")):
        value = getattr(args, argument)
        if value is not None:
            setattr(settings, attr, value)
    train_all(args.data_dir, args.results_dir, configs=configs, settings=settings,
              experiment_ids=args.experiments, cache_local=args.cache_local,
              skip_examples=args.skip_examples)


if __name__ == "__main__":
    main()
