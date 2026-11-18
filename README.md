
# Building Change Detection — LEVIR-CD

## Описание проекта

Проект посвящён автоматическому обнаружению изменений
застройки на спутниковых снимках высокого разрешения
с использованием нейронной сети Siamese U-Net.

Модель получает два спутниковых снимка одной территории,
сделанных в разные даты, и формирует бинарную маску
изменений зданий.

## Используемые данные

### LEVIR-CD
Основной датасет для обучения и тестирования. Для задач детектирования изменений в зданиях (Building Change Detection) на основе методов глубокого обучения и дистанционного зондирования Земли (Remote Sensing).

Источник:
https://justchenhao.github.io/LEVIR/

Подготовленная выборка:
- Train: 3144 пары
- Validation: 433 пары
- Test: 931 пара
- Размер изображений: 256×256

### S2Looking
специализированный открытый датасет для задач обнаружения изменений в зданиях (Building Change Detection) с использованием спутниковых снимков.

Источник:
https://github.com/S2Looking/Dataset

## Метод

Используются две архитектуры: базовая сиамская CNN и **Siamese U-Net** с пропускающими связями. Проведены пять экспериментов:

| Эксперимент | Архитектура | Функция потерь | Аугментация | Объединение признаков |
|---|---|---|---|---|
| E0 | Базовая | BCE | Нет | Конкатенация |
| E1 | Базовая | BCE + Dice | Нет | Конкатенация |
| E2 | Базовая | BCE + Dice | Да | Конкатенация |
| E3 | Siamese U-Net | BCE + Dice | Да | Конкатенация |
| E4 | Siamese U-Net | BCE + Dice | Да | Абсолютная разность |

Лучшая модель выбирается по **Validation F1**. Оцениваются Precision, Recall, F1-score, IoU и Accuracy. Её веса и порог затем используются для оценки на S2Looking без изменения параметров.

## Запуск

Python 3.10+; для обучения рекомендуется **Google Colab с GPU**. Команды выполняются **из корня репозитория**. В Colab сначала подключите Google Диск:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Установите зависимости:

```bash
pip install -r requirements.txt
```

Задайте путь к вашим данным (в терминале Colab):

```bash
ROOT="/content/drive/MyDrive/Компьютерное зрение (Кейс LEVIR-CD)"
```

**1. Подготовка данных** — выполняется один раз; если папка `LEVIR_processed_5K` уже есть, пропустите. Путь `LEVIR-CD` должен указывать на распакованный исходный набор с папками `train/val/test`:

```bash
python src/prepare_data.py --input-dir "$ROOT/LEVIR-CD" --output-dir "$ROOT/LEVIR_processed_5K" --patch-size 256 --min-building-pixels 15
```

**2. Обучение E0–E4 и тест LEVIR-CD:**

```bash
python src/train.py --data-dir "$ROOT/LEVIR_processed_5K" --results-dir "$ROOT/LEVIR_experiments_results" --cache-local
```

При наличии `configs/E0.json`–`E4.json` можно добавить `--config-dir configs`.

**3. Независимая проверка на S2Looking** (после обучения):

```bash
python src/evaluate.py --s2-root "$ROOT/S2Looking" --results-dir "$ROOT/LEVIR_experiments_results"
```

**4. Предсказание на новой паре снимков** (одинакового размера):

```bash
python src/inference.py --image-a before.png --image-b after.png --results-dir "$ROOT/LEVIR_experiments_results" --output prediction.png
```

Команды рассчитаны на Bash-терминал (включая терминал Colab). В ячейке Colab используйте `%%bash` и поместите определение `ROOT` **в ту же ячейку**, что и запускаемую команду: переменные Bash не сохраняются между разными ячейками.

## Результаты

- `results/experiment_comparison.csv` — сравнение экспериментов E0–E4.
- `results/cross_dataset_comparison.csv` — LEVIR-CD Test и S2Looking Test.
- `results/training_curves.png`, `results/levir_examples.png`, `results/s2looking_examples.png` — графики и примеры сегментации.

После запуска программы сохраняют результаты в `LEVIR_experiments_results/`; для публикации скопируйте итоговые небольшие файлы в `results/`. Веса лучшей модели (`best.pt`) и конфигурация (`best_model.json`) нужны для инференса. Большие веса размещайте отдельно, со ссылкой в репозитории.

**Ограничение:** LEVIR-CD предварительно отфильтрован по наличию изменений (не менее 15 пикселей в патче), а S2Looking в основной внешней оценке не фильтруется. Разница состава выборок учитывается при сравнении метрик. Модель выделяет изменения, но не различает отдельно строительство и снос.
