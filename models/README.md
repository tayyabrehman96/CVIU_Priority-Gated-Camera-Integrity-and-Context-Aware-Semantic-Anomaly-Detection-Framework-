# Model weights (not committed)

This release does **not** store weight files in Git.

1. **YOLOv8 (general objects)** — required for full `test_pipeline` runs (person / vehicle cues, etc.):

   ```bash
   python -c "from ultralytics import YOLO; YOLO('yolov8s.pt')"
   ```

   Place `yolov8s.pt` in the repository root (or adjust code to your path). The file is gitignored.

2. **Weapon-specialized YOLO** — optional; [`config.py`](../config.py) uses `WEAPON_MODEL_PATH` (default `models/weapon/weights/best.pt`). Train or obtain weights separately and place them under that path. For guard-only / fast runs, use `--no-models` on the benchmark scripts where supported.
