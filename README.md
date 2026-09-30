# RF-DETR Train

Dataset preparation, evaluation and fine-tuning experiments with [Roboflow RF-DETR](https://github.com/roboflow/rf-detr).

The repository currently supports downloading COCO 2017 and Open Images V7 through FiftyOne,
converting downloaded data to native COCO or Roboflow COCO layouts, evaluating pretrained
RF-DETR checkpoints through `model.evaluate()`, and fine-tuning through `model.train()`.
Available model variants are `nano`, `small`, `medium` and `large`.

The Open Images workflow prepares a single `person` class for a before/after fine-tuning experiment.
The workflow includes dataset conversion, evaluation of the original COCO checkpoint, fine-tuning,
and loading the fine-tuned checkpoint for evaluation with the Roboflow loader.

## Installation

### Python dependencies

Python 3.13 or newer is required.

This project uses [uv](https://docs.astral.sh/uv/) to manage dependencies and run scripts.

To install the dependencies, run the following command:

```bash
uv sync
```

If you want to install without development dependencies, run:

```bash
uv sync --no-dev
```

If you want to install the package with development dependencies, run:

```bash
uv sync --group dev
```

Or in general for any specific group of dependencies defined in `pyproject.toml`:

```bash
uv sync --group <GROUP_NAME>
```

To add a package to the project, use:

```bash
uv add <PACKAGE_NAME>
```

If you want to add a package with development dependencies, use:

```bash
uv add --group dev <PACKAGE_NAME>
```

If you want bump project version, use:

```bash
uv version <VERSION>
```

Or specify which part of the version to bump:

```bash
uv version --bump <SEMVER_PART> # e.g. major, minor, patch
```

## Running scripts and project directories

Run the commands below from the repository root, after installing dependencies. All script examples
use `uv run --no-sync` to use the existing environment without installing or updating dependencies.
Dataset download scripts still download dataset files, and evaluation or training can download pretrained weights.

Directory defaults are defined in `src/rf_detr_train/consts.py`:

| Constant | Default directory | Contents |
| --- | --- | --- |
| `FIFTYONE_DATASETS_DIRPATH` | `datasets/fiftyone/` | Downloaded images and original annotations |
| `NATIVE_COCO_DATASETS_DIRPATH` | `datasets/native_coco/` | Datasets for the native COCO loader |
| `ROBOFLOW_DATASETS_DIRPATH` | `datasets/roboflow/` | Datasets for the Roboflow COCO loader |
| `MODELS_DIRPATH` | `models/` | Pretrained weights; the model builder sets `RF_HOME` to this directory |
| `RUNS_DIRPATH` | `runs/` | Run configurations, evaluation metrics, training logs and checkpoints |

## Available datasets and download options

| Dataset | Download script | Current defaults |
| --- | --- | --- |
| COCO 2017 | `scripts/datasets/download_coco_dataset.py` | `validation`, all classes, no sample limit |
| Open Images V7 | `scripts/datasets/download_open_images_dataset.py` | `train`, `Person Man Woman Girl Boy`, up to 5000 images |

Both downloaders accept:

| Argument | Meaning |
| --- | --- |
| `--dataset-dirpath PATH` | FiftyOne dataset root; the downloader adds `coco-2017/` or `open-images-v7/` below it |
| `--split train/validation/test` | Source split to download |
| `--classes CLASS [CLASS ...]` | Select images containing requested classes; names are dataset-specific |
| `--max-samples N` | Maximum number of images requested for the selected split, not per class |
| `--all-samples` | Remove the sample limit while retaining the class filter; mutually exclusive with `--max-samples` |
| `--help` | Show supported arguments and their current defaults |

COCO class names include `person` and `car`; Open Images uses names such as `Person` and `Car`.
Selecting classes during download selects images; it does not automatically make them a single-class
training dataset. Conversion decides which annotations and category IDs are exported.

FiftyOne reuses downloaded files. A smaller subsequent `--max-samples` request does not clean an
existing download directory. Use a separate `--dataset-dirpath` for independent subsets and pass
the resulting dataset root as `--source-dirpath` when converting. The Open Images converter considers
all images present in the selected split's `data/` directory.

For Open Images, images containing all requested classes are prioritized before images containing
only some of them. This is not balanced sampling per class. Download limits apply to images; metadata
and annotation CSV files can still cover the complete source split.

## Native COCO and Roboflow COCO layouts

Both layouts use COCO JSON annotations. They differ in directory structure and in how RF-DETR's
selected loader maps category IDs to model outputs.

| Layout | RF-DETR argument | Image and annotation paths | Use in this repository |
| --- | --- | --- | --- |
| Native COCO | `dataset_file="coco"` | `train2017/`, `val2017/`, `annotations/instances_<split>2017.json` | Evaluate original COCO checkpoints while preserving their category IDs |
| Roboflow COCO | `dataset_file="roboflow"` | `train/`, `valid/`, `test/`, each with `_annotations.coco.json` | Prepare custom-class data for fine-tuning and subsequent evaluation |

The original COCO checkpoint predicts sparse COCO IDs, for example `person=1` and `car=3`.
The native loader preserves these IDs. The Roboflow loader remaps dataset categories to contiguous
model indices starting at zero. For our one-class Open Images dataset, the exported JSON contains
`person=1`, while the Roboflow loader uses model index `0` internally and maps it back for COCO metrics.

Therefore, evaluating an unchanged COCO checkpoint directly with the one-class Roboflow loader
would compare incompatible label indices. We prepare a native COCO view for the baseline and a
Roboflow view for fine-tuning. The corresponding test views contain identical images and annotations.
This is a local directory layout; no upload to Roboflow is required.

All converters default to relative symlinks. Keep the downloaded source images available, or use
`--image-mode copy` for an independent copy. Conversion preserves the image files without resizing
or re-encoding them. Each invocation converts one split and refuses to overwrite an existing output
split; other splits can be added to the same dataset root.

## COCO 2017: download, convert and evaluate

### Download

Download the full labelled validation split:

```bash
uv run --no-sync scripts/datasets/download_coco_dataset.py \
    --split validation --all-samples
```

The result is stored under:

```text
datasets/fiftyone/coco-2017/validation/
├── data/
└── labels.json
```

For a smaller experiment, use `--max-samples 100` instead of `--all-samples`.
Add `--classes person car` to request images containing those classes. A subset evaluation does
not represent the full COCO benchmark.

### Convert to native COCO

```bash
uv run --no-sync scripts/datasets/convert_fiftyone_coco_to_native_coco.py \
    --split validation
```

Output:

```text
datasets/native_coco/coco-2017/
├── val2017/
└── annotations/
    └── instances_val2017.json
```

The COCO converters copy `labels.json` unchanged, preserving category IDs, image IDs and annotations.
They rearrange the files; they do not merge classes or filter annotations.

### Evaluate the original checkpoint

```bash
uv run --no-sync scripts/evaluation/evaluate_coco.py \
    --model-name small \
    --dataset-dirpath datasets/native_coco/coco-2017 \
    --run-name coco_small_baseline
```

Omitting `--checkpoint-filepath` loads the selected variant's official pretrained COCO weights.
The default device is `cuda`; use `--device cpu` when needed.

### Optional: prepare COCO in the Roboflow layout

If preparing training data, first download the training split, then convert both labelled splits:

```bash
uv run --no-sync scripts/datasets/download_coco_dataset.py --split train --all-samples
uv run --no-sync scripts/datasets/convert_fiftyone_coco_to_roboflow.py --split train
uv run --no-sync scripts/datasets/convert_fiftyone_coco_to_roboflow.py --split validation
```

This produces `datasets/roboflow/coco-2017/train/` and `valid/`, each containing images and
`_annotations.coco.json`. These exports are for the corresponding Roboflow training/evaluation
loader; use the native export for the unchanged COCO checkpoint baseline.

Both COCO conversion scripts accept:

| Argument | Default / behavior |
| --- | --- |
| `--source-dirpath PATH` | `datasets/fiftyone/coco-2017` |
| `--output-dirpath PATH` | `datasets/native_coco/coco-2017` or `datasets/roboflow/coco-2017`, depending on the script |
| `--split train/validation` | `validation`; native maps these to `train2017/val2017`, Roboflow to `train/valid` |
| `--image-mode symlink/copy` | `symlink` |

COCO test images have no public detection ground truth. The download script exposes `test`, but
the COCO converters only accept `train` and `validation` for training or local evaluation.

## Evaluation arguments and saved results

`scripts/evaluation/evaluate_coco.py` calls `model.evaluate()` with the loader selected by
`--dataset-file coco/roboflow`. The default `coco` accepts native-layout datasets, including our
converted Open Images test subset. Use `roboflow` for checkpoints fine-tuned on our Roboflow
exports. The script does not select a loader from the directory name, and changing the loader
does not change the default dataset path; supply `--dataset-dirpath` for your export.

Defaults live in `EvaluationConfig` in `src/rf_detr_train/evaluation/schemas.py`.
Argparse passes only explicitly supplied overrides, so there is one source of defaults.

| Argument | Default | Meaning |
| --- | --- | --- |
| `--model-name` | `small` | `nano`, `small`, `medium` or `large` |
| `--checkpoint-filepath PATH` | Official COCO weights | Custom checkpoint matching the selected variant and the loader's class mapping |
| `--dataset-file coco/roboflow` | `coco` | Select the native COCO or Roboflow COCO loader |
| `--dataset-dirpath PATH` | `datasets/native_coco/coco-2017` | Dataset root matching the loader: native `val2017/` and `annotations/`, or Roboflow `valid/` and/or `test/` with `_annotations.coco.json` |
| `--device` | `cuda` | Device passed to RF-DETR |
| `--split val/test` | `val` | Roboflow: `val` reads `valid/`, `test` reads `test/`; native COCO reads `val2017/` in both modes |
| `--batch-size` | `8` | Images per batch; reduce it if GPU memory is insufficient |
| `--num-workers` | `4` | Data loading worker processes; `0` loads in the main process |
| `--eval-max-dets` | `100` | Maximum detections per image and category used by COCO metrics |
| `--log-per-class-metrics` / `--no-log-per-class-metrics` | Enabled | Control per-class metric reporting |
| `--run-name NAME` | `rf_detr_<model-name>` | Experiment name used to generate the result directory |
| `--output-dirpath PATH` | `runs/<run-name>/<date_and_time>` | Exact output directory; overrides the automatic path without appending a timestamp |

For example, `--run-name coco_small_baseline` writes:

```text
runs/coco_small_baseline/<YYYY-MM-DD_HH-MM-SS>/results.json
```

Without a custom run name, the Small model writes to `runs/rf_detr_small/<date_and_time>/`.
The dataset name does not automatically determine this directory. The directory is created by
the script. Use a new directory if repeating an explicit `--output-dirpath`, as `results.json`
at that exact path will be overwritten.

`results.json` contains `config` (including the model, checkpoint path, dataset loader/path, run name and evaluation options)
and `metrics` returned by RF-DETR. Paths are serialized as JSON strings. Only returned metric values
are saved; the console's per-class table may contain additional values that the package does not
return. The script does not modify RF-DETR callbacks to collect them.

## Open Images V7: prepare a person fine-tuning experiment

### Download the three splits

The following commands request 5000 training images, 500 validation images and 500 test images:

```bash
uv run --no-sync scripts/datasets/download_open_images_dataset.py \
    --split train --classes Person Man Woman Girl Boy --max-samples 5000

uv run --no-sync scripts/datasets/download_open_images_dataset.py \
    --split validation --classes Person Man Woman Girl Boy --max-samples 500

uv run --no-sync scripts/datasets/download_open_images_dataset.py \
    --split test --classes Person Man Woman Girl Boy --max-samples 500
```

Keep the official split assignments. Training uses `train`, checkpoint selection uses `validation`,
and the final before/after comparison uses the same fixed `test` subset.

FiftyOne stores each split in this form:

```text
datasets/fiftyone/open-images-v7/<train|validation|test>/
├── data/
├── labels/
│   └── detections.csv
└── metadata/
    └── classes.csv
```

The downloader supports other Open Images classes through `--classes`. The current Open Images
converter specifically prepares the five person categories above; it is not a general exporter
for arbitrary class selections.

### Conversion rules

`scripts/datasets/convert_fiftyone_open_images_to_coco.py` reads local CSV annotations and image
dimensions directly. It creates COCO JSON with pixel-space `[x, y, width, height]` boxes and:

- Merges `Person`, `Man`, `Woman`, `Girl` and `Boy` into `person`, with JSON category ID `1`.
- Excludes an entire image if any of these five categories has `IsGroupOf=1` (group) or
  `IsGroupOf=-1` (unknown). Only confirmed `IsGroupOf=0` person annotations are retained.
- Excludes images without person boxes. Group flags of other classes do not exclude an image.
- Omits `Human body` and other classes from the output annotations.
- Deduplicates identical person boxes while retaining distinct overlapping boxes and
  occluded/truncated individuals.

The exported counts can be lower than the downloaded counts. The converter logs selected/excluded
image counts and stores these counts in the COCO JSON `info` field. Unknown-status exclusions are
counted separately from images already excluded for a confirmed group.

This is a filtered person detection task evaluated with COCO metrics. It does not reproduce the
official Open Images evaluation protocol, which handles class hierarchies and partial annotations
differently.

### Prepare the Roboflow layout for fine-tuning

```bash
uv run --no-sync scripts/datasets/convert_fiftyone_open_images_to_coco.py --split train --layout roboflow
uv run --no-sync scripts/datasets/convert_fiftyone_open_images_to_coco.py --split validation --layout roboflow
uv run --no-sync scripts/datasets/convert_fiftyone_open_images_to_coco.py --split test --layout roboflow
```

Output:

```text
datasets/roboflow/open-images-v7-person/
├── train/
│   ├── <images>
│   └── _annotations.coco.json
├── valid/
│   ├── <images>
│   └── _annotations.coco.json
└── test/
    ├── <images>
    └── _annotations.coco.json
```

### Prepare the native COCO test view for the original checkpoint

```bash
uv run --no-sync scripts/datasets/convert_fiftyone_open_images_to_coco.py \
    --split test --layout native
```

Output:

```text
datasets/native_coco/open-images-v7-person-test/
├── val2017/
└── annotations/
    └── instances_val2017.json
```

The directory is named `val2017` to satisfy RF-DETR's native COCO loader. These remain **Open Images
test images**; the conversion does not move them into the experiment's validation split. Both layouts
apply the same filtering and retain `person=1` in JSON. Keep the downloaded source subset unchanged
between exports so the baseline and fine-tuned evaluation use the same test samples.

Open Images conversion arguments:

| Argument | Default / behavior |
| --- | --- |
| `--source-dirpath PATH` | `datasets/fiftyone/open-images-v7` |
| `--output-dirpath PATH` | Roboflow: `datasets/roboflow/open-images-v7-person`; native train/validation: `datasets/native_coco/open-images-v7-person`; native test: `datasets/native_coco/open-images-v7-person-test` |
| `--split train/validation/test` | `validation` |
| `--layout roboflow/native` | `roboflow` |
| `--image-mode symlink/copy` | `symlink` |

For native output, `train` maps to `train2017`; both `validation` and `test` map to `val2017`.
When overriding `--output-dirpath`, use separate roots for native validation and test to keep
them distinct. The default paths already separate them.

### Evaluate before fine-tuning

```bash
uv run --no-sync scripts/evaluation/evaluate_coco.py \
    --model-name small \
    --dataset-file coco \
    --dataset-dirpath datasets/native_coco/open-images-v7-person-test \
    --split test \
    --run-name open_images_small_before_finetuning
```

The original COCO checkpoint is evaluated on the prepared `person` test annotations. Results go to:

```text
runs/open_images_small_before_finetuning/<date_and_time>/results.json
```

### Fine-tune on the Roboflow dataset

```bash
uv run --no-sync scripts/training/train_rf_detr.py \
    --run-name open_images_small_finetuning
```

This starts from the official RF-DETR Small COCO weights, trains on `train/`, and validates on
`valid/` to select the best checkpoint. RF-DETR adjusts the detection head to the dataset's
classes automatically. The default dataset is `datasets/roboflow/open-images-v7-person`.
Each split must contain images and `_annotations.coco.json` as shown above.

Training arguments (defaults are defined only in `TrainingConfig`):

| Argument | Default / behavior |
| --- | --- |
| `--model-name nano/small/medium/large` | `small` |
| `--dataset-dirpath PATH` | `datasets/roboflow/open-images-v7-person` |
| `--checkpoint-filepath PATH` | Omitted: official pretrained COCO weights; supplied: initial weights matching the selected variant |
| `--device DEVICE` | `cuda` |
| `--epochs N` | `20` |
| `--batch-size N` | `4` images per device |
| `--grad-accum-steps N` | `4`; with the default batch size, the effective batch size on one GPU is `16` |
| `--lr FLOAT` | `0.0001` |
| `--num-workers N` | `4`; `0` loads data in the main process |
| `--run-name NAME` | Optional; defaults to `rf_detr_<model_name>` |
| `--output-dirpath PATH` | Optional exact output directory; overrides the automatic path and timestamp |
| `--help` | Show supported arguments and their current defaults |

For example, to reduce the per-device batch size while keeping the same effective batch size:

```bash
uv run --no-sync scripts/training/train_rf_detr.py \
    --batch-size 2 --grad-accum-steps 8 \
    --run-name open_images_small_finetuning
```

The script saves `config.json` before training. RF-DETR writes its training logs and checkpoints
to the same directory, including `checkpoint_best_ema.pth` for the EMA model with the best
validation score. EMA weights are an exponential moving average of the model weights during training:

```text
runs/open_images_small_finetuning/<date_and_time>/
```

Validation uses `eval_max_dets=100` and `log_per_class_metrics=True`, matching the default
evaluation settings. TensorBoard is disabled. `run_test=False` keeps the test split out of this
training run; `test/` is not required to start training. `--checkpoint-filepath` initializes
model weights; it does not resume a previous run's optimizer or scheduler state.

### Load and evaluate after fine-tuning

After training finishes, load `checkpoint_best_ema.pth` from that training run and evaluate
the Roboflow test export. Replace `<date_and_time>` below with the actual training directory name.
Keep `--model-name` identical to the variant used for training.

```bash
uv run --no-sync scripts/evaluation/evaluate_coco.py \
    --model-name small \
    --checkpoint-filepath "runs/open_images_small_finetuning/<date_and_time>/checkpoint_best_ema.pth" \
    --dataset-file roboflow \
    --dataset-dirpath datasets/roboflow/open-images-v7-person \
    --split test \
    --run-name open_images_small_after_finetuning
```

The script loads the checkpoint through the existing model builder, evaluates `test/`, and saves
the configuration and returned metrics to:

```text
runs/open_images_small_after_finetuning/<date_and_time>/results.json
```

This evaluation creates a new timestamped run. To evaluate the validation split instead, pass
`--split val`; the Roboflow loader then reads `valid/`.

The one-class model predicts person at index `0`, matching the mapping used by the Roboflow
loader during training and evaluation. The original COCO checkpoint predicts person at sparse
COCO ID `1`, which is why the before-training evaluation uses the native export.

Compare `metrics` in the before/after `results.json` files for the same test images. Both commands
use `split=test`, `eval_max_dets=100`, and per-class reporting. The native and Roboflow test exports
must come from the same downloaded subset with the same conversion filters, as described above.

## Pre-commit

This project uses pre-commit to ensure that the code is formatted correctly.

To install pre-commit, run the following command:

```bash
pre-commit install
```

or

```bash
uv run --no-sync pre-commit install
```

To run pre-commit on all files, run the following command:

```bash
pre-commit run --all-files
```

or

```bash
uv run --no-sync pre-commit run --all-files
```
