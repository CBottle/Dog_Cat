# Data Split Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create the first Jupyter notebook that shows Jun's step-by-step reasoning while preparing the Oxford-IIIT Pet dog/cat split.

**Architecture:** Use a notebook-first workflow for the learning and presentation trail. The notebook contains markdown decision notes plus executable code cells that download/load the dataset, convert labels, create train/validation/test splits, save CSV outputs, and save a class distribution figure.

**Tech Stack:** Python 3.12, Jupyter Notebook, ipykernel, torchvision, pandas, scikit-learn, matplotlib.

---

## File Structure

- Modify: `requirements.txt` to add `ipykernel`.
- Create: `notebooks/01_data_split.ipynb`.
- Create outputs when the notebook is run: `data/processed/pet_binary_splits.csv`, `data/processed/class_distribution.csv`, `outputs/figures/class_distribution.png`.

### Task 1: Add Notebook Dependency

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add `ipykernel`**

Add `ipykernel` to `requirements.txt` so VS Code can use the project `.venv` as a notebook kernel.

- [ ] **Step 2: Install the new dependency**

Run:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Expected: `ipykernel` installs successfully and already-installed packages remain satisfied.

### Task 2: Create Data Split Notebook

**Files:**
- Create: `notebooks/01_data_split.ipynb`

- [ ] **Step 1: Add markdown reasoning cells**

The notebook must include:

- Goal
- Why Oxford-IIIT Pet was chosen
- What train/validation/test mean
- Why the split ratio is `70/15/15`
- How to judge whether the split looks balanced
- Presentation-ready summary

- [ ] **Step 2: Add executable code cells**

The notebook must include code cells for:

- imports
- dataset loading
- binary dog/cat label conversion
- stratified split creation
- CSV saving
- distribution table and figure saving

### Task 3: Verify Notebook

**Files:**
- Verify: `notebooks/01_data_split.ipynb`

- [ ] **Step 1: Validate notebook JSON**

Run:

```powershell
.\.venv\Scripts\python.exe -c "import json; json.load(open('notebooks/01_data_split.ipynb', encoding='utf-8')); print('notebook_json_ok')"
```

Expected: `notebook_json_ok`

- [ ] **Step 2: Verify imports**

Run:

```powershell
.\.venv\Scripts\python.exe -c "import ipykernel, torch, torchvision, pandas, sklearn, matplotlib; print('imports_ok')"
```

Expected: `imports_ok`

### Task 4: Commit

**Files:**
- Commit: `requirements.txt`, `docs/superpowers/plans/2026-07-01-data-split-notebook-plan.md`, `notebooks/01_data_split.ipynb`

- [ ] **Step 1: Commit notebook setup**

Run:

```bash
git add requirements.txt docs/superpowers/plans/2026-07-01-data-split-notebook-plan.md notebooks/01_data_split.ipynb
git commit -m "Add data split notebook"
```

Expected: commit succeeds.
