# DL Final Project

Multimodal science multiple-choice QA experiments built around SmolVLM (Hugging Face), with zero-shot baselines, frozen-head baselines, and LoRA / DoRA / RSLoRA fine-tuning studies. Training and evaluation are driven from Jupyter notebooks; shared logic lives under `src/`.

## Data layout

- `data/train.csv`, `data/val.csv`, `data/test.csv` — question text, choices, labels, hints/lecture fields, and image references.
- `data/images/` — image assets referenced by the CSVs (paths may be split-relative or numeric IDs; see `input_ablation_utils.resolve_split_image_path`).
- `data/sample_submission.csv` — example submission format.

## Notebooks

### `notebooks/01_eda.ipynb`

EDA — “Pixels to Predictions”: validates schema and split integrity, summarizes labels and text/image/metadata behavior, analyzes missingness and context availability, builds difficulty cohorts, and writes CSV artifacts to `outputs/eda_outputs/` for downstream modeling.

### `notebooks/zero_shot_study/gen_zero_shot_ablations.ipynb`

Phase-1 generative zero-shot ablations for frozen SmolVLM: prompt and output-format ablations; the model generates an answer string and the notebook parses it to a choice index. Produces validation tables and optional test submissions.

### `notebooks/zero_shot_study/non_gen_zero_shot_ablations.ipynb`

Phase-1 non-generative zero-shot ablations: scores each answer candidate under the prompt (highest score wins). Same frozen-backbone assumption; outputs validation metrics and optional submissions.

### `notebooks/zero_shot_study/zero_shot_best_candidates_full_run.ipynb`

Runs both generative and non-generative zero-shot ablation suites in one place (broader candidate screening than the single-mode notebooks).

### `notebooks/LoRA_study/01_lora_sft_vs_option_scoring.ipynb`

Compares two training objectives (option-scoring vs next-token SFT) using standard LoRA on attention projection modules (`q_proj`, `k_proj`, `v_proj`, `o_proj`). Uses a sanity-sized train slice by default and saves under `outputs/01_lora_sft_vs_option_scoring/`.

### `notebooks/LoRA_study/02_rslora.ipynb`

Trains RSLoRA (rank 16) with option-scoring on attention projections; writes to `outputs/02_lora_dora_rslora_scoring/` and can emit submissions.

### `notebooks/LoRA_study/03_dora.ipynb`

Runs the shared generative LoRA training pipeline with ablation key `dora`, validation-focused defaults (`sanity_check=True`), and outputs under `outputs/03_dora_scoring/`. Adapter flags are defined in the notebook cells; verify `use_dora` / `use_rslora` match the experiment you intend.

### `notebooks/LoRA_study/dora_extended_run.ipynb`

Same overall pipeline as `03_dora.ipynb` but with `sanity_check=False` (full training split) and longer-run artifacts under `outputs/03_dora_scoring_long_run/` (e.g., submission and validation predictions).

### `notebooks/LoRA_study/rslora_attn_mlp_full_run.ipynb`

Full run with RSLoRA adapters on attention and MLP layers (`q/k/v/o_proj` plus `up_proj`, `down_proj`, `gate_proj`), option-scoring objective, metadata in the prompt, and outputs under `outputs/lora_mlp_attention_option_scoring/`. Despite the “dora” wording in the ablation name, the cell config uses the RSLoRA path; confirm flags in the notebook before interpreting results.

## Source code (`src/`)

| Module | Role |
|--------|------|
| `eda_utils.py` | Helpers for EDA notebooks: dataset cards, text cleaning, choice parsing, text features, context flags, simple stats, CSV saves. |
| `input_ablation_utils.py` | Data-side ablations: resolving image paths per split, light augmentations (resize, brightness jitter, rotation), and prompt/metadata construction hooks shared across zero-shot and LoRA flows. |
| `frozen_baseline_utils.py` | Frozen SmolVLM baseline: loading splits, prompt building, datasets/loaders, linear/MLP/choice-wise heads, training/eval loops, submission helpers. |
| `lora_baseline_utils.py` | Earlier LoRA-on-backbone utilities (PEFT `LoraConfig`, param counts, training helpers) building on the frozen baseline stack. |
| `zero_shot_utils.py` | Lightweight helpers for the basic zero-shot notebook (seeds, splits, prompts, image paths). |
| `zero_shot_ablation_utils.py` | Phase-1 zero-shot ablation drivers: richer prompting, image handling via `input_ablation_utils`, and evaluation flows for generative vs scoring setups. |
| `gen_lora_study_utils.py` | Core generative LoRA study pipeline: model loading, LoRA/DoRA/RSLoRA wiring, option-scoring and SFT training, validation metrics, submissions, Colab/Drive helpers. |
| `lora_tuning_block_utils.py` | Block sweeps and orchestration: Cartesian product of hyperparameters, `run_single_lora_ablation`, test submission from saved adapters, trainable-parameter counting. |
| `lora_best_candidates_full_run_utils.py` | Expands named ablation dicts (one or more variants per name) into flat configs and runs `run_lora_candidate_on_val` for full-run notebooks. |

## Other paths

- **`outputs/`** — Run artifacts (CSVs for validation predictions, submissions, ablation result tables). Paths mirror notebook output directories.
- **`report/main.tex`** — LaTeX source for the project write-up.
- **`notebooks_pdf/`** — Exported notebook PDFs (if present), for offline reading.

## Environment

There is no checked-in `requirements.txt`. Notebooks expect a Python environment with PyTorch, transformers, PEFT, Pillow, pandas, tqdm, and (for some flows) Google Colab and Drive mounting. Install versions compatible with your CUDA setup and the pinned `MODEL_ID` in each notebook (commonly `HuggingFaceTB/SmolVLM-500M-Instruct`).

## Running

1. Place or symlink the competition `data/` tree (CSVs and images) as expected above.
2. Open the relevant notebook; set `PROJECT_ROOT` / Colab paths if not using the default layout.
3. For LoRA notebooks, outputs and adapter checkpoints are written under the `outputs/<run_name>/` folder configured in the first cells.
