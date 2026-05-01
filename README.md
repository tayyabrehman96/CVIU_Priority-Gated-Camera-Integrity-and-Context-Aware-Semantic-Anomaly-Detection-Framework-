# Priority-Gated Camera Integrity and Context-Aware Semantic Anomaly Detection

Research code release for **hybrid risk simulation**, **frame-level anomaly benchmarks**, and **figure reproduction** accompanying work submitted to *Computer Vision and Image Understanding* (CVIU). This repository is a **slim, reproducibility-focused** snapshot: the full offline demo GUI, VLM service layer, and live-engine orchestration are **not** included here.

**Public repository:** [github.com/tayyabrehman96/CVIU_Priority-Gated-Camera-Integrity-and-Context-Aware-Semantic-Anomaly-Detection-Framework-](https://github.com/tayyabrehman96/CVIU_Priority-Gated-Camera-Integrity-and-Context-Aware-Semantic-Anomaly-Detection-Framework-)

## Contribution (summary)

The framework combines **priority-gated** hard guards (camera health, occlusion, blur) with **context-aware** semantic cues (object-centric models and schedule-driven risk). This codebase focuses on:

- **Reproducible scenario packs** under `simulations/` (generated, not committed), mixing dataset-native `TEST_*.mp4` clips with **synthetic risk injections** driven by [`risk_simulation_protocol.json`](risk_simulation_protocol.json).
- **Benchmarking** via [`test_pipeline.py`](test_pipeline.py), which evaluates aligned labels against a defined temporal schedule and reports accuracy, macro-F1, and risk-quality metrics used in the experiment matrix.

## Methodology overview

1. **Hybrid risk simulation** ([`hybrid_risk_simulation.py`](hybrid_risk_simulation.py)) builds a run directory with a `manifest.json` and a `videos/` subfolder. Modes include `dataset_only`, `synthetic_only`, and `hybrid`. Severity (`low` / `high`) and RNG `seed` are controlled from the CLI; temporal structure and scales come from the protocol JSON.
2. **Experiment matrix** ([`run_experiment_matrix.py`](run_experiment_matrix.py)) runs the simulation grid (modes × severities × seeds), invokes `test_pipeline.run_benchmark` on each pack, and writes timestamped CSV summaries under `reports/`. [`aggregate_experiment_matrix.py`](aggregate_experiment_matrix.py) aggregates multiple runs; [`run_experiment_cell.py`](run_experiment_cell.py) runs a single cell.
3. **Detection stack in this repo** Minimal [`core/guards.py`](core/guards.py) implements CV-based fire/smoke, occlusion/cover, blur/block heuristics, and related helpers; tunables load from a local **`config.py`** you create by copying [`config.example.py`](config.example.py). Optional **Ultralytics YOLOv8** paths are used inside [`test_pipeline.py`](test_pipeline.py) for full-model evaluation.

## Data: how to obtain inputs

Large media files are **not** committed (see [`.gitignore`](.gitignore)).

| Source | Role |
|--------|------|
| [`dataset_preparation.ipynb`](dataset_preparation.ipynb) | Curated **public** clip URLs, normalization to 720p H.264, optional frame extraction; extend `VIDEO_SOURCES` as documented in the notebook. |
| [`download_test_videos.ipynb`](download_test_videos.ipynb) | Additional helpers for obtaining baseline surveillance-style footage. |
| [`generate_test_videos.py`](generate_test_videos.py) | Builds **synthetic multi-scene** `TEST_*.mp4` clips under `videos/` from at least one **normal** base file (see `find_source_video()` for expected filenames). |
| Public benchmarks (e.g. UCF-Crime, ShanghaiTech) | Cite and use per your manuscript’s experimental section; this repo does not redistribute those datasets. |

Place normalized or generated clips in [`videos/`](videos/README.md) before running simulations. For **`--metadata-only`** simulation runs, existing `TEST_*.mp4` files are copied into each scenario pack without full re-encoding.

## Models

- **YOLOv8 (general):** download with Ultralytics (see [`models/README.md`](models/README.md)); default checkpoint name `yolov8s.pt` at repo root.
- **Weapon-specialized YOLO:** optional; path from `WEAPON_MODEL_PATH` in your local `config.py` (see [`config.example.py`](config.example.py)). Omit weights or pass `--no-models` where supported for faster, guard-heavy runs.

## Reproduction

**1. Environment**

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux/macOS:
# source .venv/bin/activate

# Create local config (required once per clone)
# Windows PowerShell: Copy-Item config.example.py config.py
# bash: cp config.example.py config.py

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

**2. Weights** — follow [`models/README.md`](models/README.md).

**3. Videos** — populate `videos/` (see **Data** above).

**4. Single simulation pack (example, metadata-only)**

```bash
python hybrid_risk_simulation.py --mode hybrid --severity high --seed 37 --metadata-only
```

**5. Full experiment matrix**

```bash
python run_experiment_matrix.py --sample-frames 5 --out-dir reports
```

Committed **CSV** exports under [`reports/`](reports/) (`risk_matrix_*.csv`, `detections_*.csv`, `frame_log_*.csv`, `video_summary_*.csv`, etc.) record dataset-derived experiment outputs. Regenerating runs still writes new files alongside them; large scratch `.txt` logs remain untracked.

Use `--no-models` on `run_experiment_matrix.py` / `test_pipeline.py` if you want to skip YOLO loading.

**6. Optional analysis / figures**

```bash
python eval_resample_imbalance.py --help
python generate_sota_figures.py
python generate_methodology_diagram.py
```

## Repository layout

```
├── config.example.py              # Template → copy to config.py (gitignored)
├── core/
│   └── guards.py                  # CV guards & fire/smoke heuristics (minimal core)
├── hybrid_risk_simulation.py      # Scenario pack builder
├── risk_simulation_protocol.json  # Severities and temporal schedule
├── run_experiment_matrix.py      # Full grid + benchmark
├── run_experiment_cell.py        # Single grid cell
├── aggregate_experiment_matrix.py
├── test_pipeline.py               # Benchmark driver
├── eval_resample_imbalance.py
├── generate_*.py                  # Test videos & publication figures
├── dataset_preparation.ipynb
├── download_test_videos.ipynb
├── Images/                        # Placeholder only — PNGs gitignored; run generate_*.py or add files locally
├── reports/                       # *.csv experiment outputs (tracked)
├── models/README.md
├── videos/README.md
├── requirements.txt
└── LICENSE
```

## Citation

If you use this code, please cite the **CVIU manuscript** once it is available, and the **code repository**:

```bibtex
@software{rehman2026cviu_priority_gated_code,
  author       = {Rehman, Tayyab},
  title        = {{Priority-Gated Camera Integrity and Context-Aware Semantic Anomaly Detection}: research code},
  year         = {2026},
  url          = {https://github.com/tayyabrehman96/CVIU_Priority-Gated-Camera-Integrity-and-Context-Aware-Semantic-Anomaly-Detection-Framework-},
  note         = {Supplemental implementation; add the journal \texttt{@article} with DOI when published.}
}
```

Related survey and VAD literature for related work is cited in the CVIU manuscript (bib entries were maintained in a local `cas-refs.bib` — not distributed with this repository).

## License

Released under the [MIT License](LICENSE).
