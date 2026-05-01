# Video inputs

This directory is intentionally **empty in Git**. Populate it before running experiments:

1. **Synthetic suite:** from the repo root, after at least one normal base clip is present (see `generate_test_videos.py` → `find_source_video()`), run:

   ```bash
   python generate_test_videos.py
   ```

2. **Curated clips:** use [`dataset_preparation.ipynb`](../dataset_preparation.ipynb) or [`download_test_videos.ipynb`](../download_test_videos.ipynb) for public sources and normalization steps.

Video files (`*.mp4`, etc.) are listed in `.gitignore` so they are not committed.
