# OpenCV Augmentation Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `notebooks/02_1_opencv_augmentation_training.ipynb` with a WHY/HOW/RESULT structure for train-only OpenCV data augmentation and comparison against the baseline training notebook.

**Architecture:** The notebook reuses the split CSV from `01`, augments only train images offline with OpenCV, builds an augmented train DataFrame, trains the same SimpleCNN with EarlyStopping/checkpointing, and saves comparison-ready outputs.

**Tech Stack:** Python, pandas, Pillow, matplotlib, OpenCV (`cv2`), PyTorch, torchvision transforms.

---

### Task 1: Add Dependency

**Files:**
- Modify: `requirements.txt`

- [ ] Add `opencv-python` so the notebook can import `cv2`.

- [ ] Verify import:

```powershell
.\.venv\Scripts\python.exe -c "import cv2; print(cv2.__version__)"
```

### Task 2: Create Notebook

**Files:**
- Create: `notebooks/02_1_opencv_augmentation_training.ipynb`

- [ ] Create numbered markdown cells from Step 0 through Step 15.

- [ ] Include WHY/HOW/RESULT language in the markdown cells.

- [ ] Add code cells for paths, split CSV loading, train-only augmentation, sample display, augmented CSV creation, Dataset/DataLoader, SimpleCNN training, loss curve, and comparison summary.

### Task 3: Verify Notebook

**Files:**
- Verify: `notebooks/02_1_opencv_augmentation_training.ipynb`

- [ ] Validate JSON:

```powershell
Get-Content -Raw -Encoding UTF8 notebooks\02_1_opencv_augmentation_training.ipynb | ConvertFrom-Json | Out-Null
```

- [ ] Compile all code cells:

```powershell
.\.venv\Scripts\python.exe -c "import json, pathlib; nb=json.loads(pathlib.Path('notebooks/02_1_opencv_augmentation_training.ipynb').read_text(encoding='utf-8')); [compile(''.join(c.get('source', [])), f'cell{i}', 'exec') for i, c in enumerate(nb['cells']) if c.get('cell_type') == 'code']; print('code syntax ok')"
```

### Task 4: Commit

**Files:**
- Modify: `requirements.txt`
- Create: `notebooks/02_1_opencv_augmentation_training.ipynb`
- Create: `docs/superpowers/specs/2026-07-03-opencv-augmentation-training-design.md`
- Create: `docs/superpowers/plans/2026-07-03-opencv-augmentation-training-plan.md`

- [ ] Commit:

```powershell
git add requirements.txt notebooks/02_1_opencv_augmentation_training.ipynb docs/superpowers/specs/2026-07-03-opencv-augmentation-training-design.md docs/superpowers/plans/2026-07-03-opencv-augmentation-training-plan.md
git commit -m "Add OpenCV augmentation training notebook"
```
