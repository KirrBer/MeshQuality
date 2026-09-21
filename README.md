# MeshQuality

Модель для определения качества 3D-мешей и классификации артефактов по изображениям объекта и его геометрии. Обучалась на датасете с 8964 объектами.

## Постановка задачи

Для каждого объекта требуется предсказать:

- `quality`: пригоден ли меш для дальнейшего использования (`1` - хороший, `0` - плохой);
- наличие десяти типов дефектов: `abstract`, `artifacts`, `intersection`, `lowpoly`, `noisy`, `open`, `partial`, `scale`, `set`, `simple`.

У одного объекта может быть несколько дефектов. Объект считается хорошим, если ни один дефект не присутствует.

Итоговая метрика конкурса:

```text
final_score = 10 * F1(quality) + 10 * weighted_F1(defects)
```

## Подход

Решение объединяет два источника информации:

1. **Multi-view encoder** - предобученный `ResNet-50` обрабатывает шесть ракурсов из одного PNG. Для ракурсов используется attention pooling.
2. **PointNet encoder** - обрабатывает 1024 нормализованные точки из массива `vertices` в NPZ-файле. Используются T-Net, свертки `1x1`, max pooling и attention pooling.
3. **Gated fusion** - объединяет визуальные и геометрические признаки.
4. **Выходные головы** - десять бинарных классификаторов дефектов и отдельная голова качества, использующая также предсказания дефектов.

Во время обучения применяются Focal Loss, аугментации изображений и облаков точек, пятифолдовая кросс-валидация и фиксированные seed-значения. При инференсе используется ансамбль из пяти моделей и test-time augmentation (TTA).

## Структура проекта

```text
.
├── train.py                 # обучение пяти моделей по фолдам
├── test.py                  # инференс ансамбля и создание CSV
├── find_tresh.py            # подбор порогов на OOF-предсказаниях
├── train.csv                # разметка обучающей выборки
├── test.csv                 # список объектов тестовой выборки
├── train/                   # PNG и NPZ для обучения
├── test/                    # PNG и NPZ для тестирования
├── best_model_fold*.pth     # сохраненные веса моделей
└── predictions54.csv        # пример/результат предсказаний
```

## Формат данных

Для каждого `item_id` должны быть доступны:

```text
train/<item_id>.png
train/<item_id>.npz
test/<item_id>.png
test/<item_id>.npz
```

PNG содержит шесть видов 3D-объекта, расположенных сеткой `2 x 3`. NPZ должен содержать массив `vertices` формы `(N, 3)`; при отсутствии или ошибке чтения используются нулевые точки. Таблица `train.csv` содержит `item_id`, десять бинарных меток дефектов и `quality`. `test.csv` содержит как минимум `item_id`.

## Установка

Нужен Python 3.10+ и желательно CUDA-совместимая видеокарта. Создание окружения в Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch torchvision numpy pandas pillow scikit-learn
```

Версии `torch` и `torchvision` должны быть совместимы. Для GPU устанавливайте сборки PyTorch с официальной страницы: <https://pytorch.org/get-started/locally/>.

Предобученный `ResNet-50` может быть автоматически загружен torchvision при первом запуске, поэтому для первичного запуска требуется доступ к сети. Если веса уже находятся в кэше PyTorch, интернет не нужен.

## Обучение

Из корня проекта:

```powershell
python train.py
```

Скрипт автоматически:

- читает `train.csv` и данные из `train/`;
- создает пять воспроизводимых фолдов с seed-значениями `42`, `2024`, `418`, `122`, `1756`;
- обучает одну модель на каждом фолде;
- сохраняет лучшие веса в `best_model_fold0.pth` ... `best_model_fold4.pth`.

Параметры обучения заданы непосредственно в `train.py`: 1024 точки на объект, batch size `64`, 8 эпох с замороженным визуальным backbone и последующее дообучение последних слоев.

## Подбор порогов

Порог для бинаризации вероятностей можно подобрать на out-of-fold-предсказаниях:

```powershell
python find_tresh.py `
	--train_csv train.csv `
	--train_dir train/ `
	--mesh_dir train/ `
	--model_pattern "best_model_fold{}.pth" `
	--output thresholds.json
```

Скрипт сохранит пороги и OOF-метрики в `thresholds.json`, а также напечатает готовые значения для констант `test.py`. Текущая версия `test.py` не читает `thresholds.json` автоматически: значения после подбора нужно перенести в `QUALITY_THRESHOLD` и `DEFAULT_DEFECT_THRESHOLDS`.

## Инференс

После обучения запустите ансамбль из пяти моделей:

```powershell
python test.py `
	--test_csv test.csv `
	--test_dir test/ `
	--mesh_dir test/ `
	--output submission.csv `
	--save_probs
```

По умолчанию включены TTA и пять файлов `best_model_fold*.pth`. Результат `submission.csv` содержит колонки:

```text
item_id,abstract,artifacts,intersection,lowpoly,noisy,open,partial,scale,set,simple,quality
```

Для отключения TTA используйте `--no_tta`. Для другого набора моделей передайте пути через запятую:

```powershell
python test.py --test_csv test.csv --test_dir test/ --mesh_dir test/ `
	--model_paths "best_model_fold0.pth,best_model_fold1.pth" `
	--output submission.csv
```

Опция `--save_probs` дополнительно создает файл `submission_with_probs.csv` с вероятностями до бинаризации.

## Воспроизводимость

В коде используется `SEED = 42`, а для фолдов заданы отдельные фиксированные seed-значения. Для максимально близкого воспроизведения результата используйте ту же версию Python/PyTorch, те же веса ImageNet и те же параметры batch size, числа воркеров и TTA. При отсутствии CUDA вычисления будут выполняться на CPU, но обучение существенно замедлится.

## Ссылки на исходную задачу

- [Описание задачи](task.md)