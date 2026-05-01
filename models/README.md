# Model weights (not committed)

This release does **not** store weight files in Git.

1. **YOLOv8 (general objects)** — required for full `test_pipeline` runs (person / vehicle cues, etc.):

   ```bash
   python -c "from ultralytics import YOLO; YOLO('yolov8s.pt')"
   ```

   Place `yolov8s.pt` in the repository root (or adjust code to your path). The file is gitignored.

2. **Weapon-specialized YOLO** — optional; your local `config.py` (from [`config.example.py`](../config.example.py)) sets `WEAPON_MODEL_PATH` (default `models/weapon/weights/best.pt`). For guard-only / fast runs, use `--no-models` on the benchmark scripts where supported.
