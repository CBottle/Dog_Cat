# Evaluation Analysis Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `notebooks/03_evaluation_analysis.ipynb` to compare the final two regularized CNN candidates with accuracy, confusion matrix, precision, recall, and misclassified images.

**Architecture:** The notebook first summarizes all saved training results, explains why `Regularized Original` is the final primary model and `Regularized OpenCV` is the comparison model, then evaluates both checkpoints on the same held-out test set.

**Tech Stack:** Python, pandas, matplotlib, scikit-learn, Pillow, PyTorch, torchvision transforms.

---

### Task 1: Create Evaluation Notebook

**Files:**
- Create: `notebooks/03_evaluation_analysis.ipynb`

- [ ] Add numbered markdown cells from Step 0 through Step 11.
- [ ] Load training summaries from `outputs/metrics`.
- [ ] Build a comparison table and bar charts for best validation loss, validation accuracy, and training time.
- [ ] Explain why `Regularized Original` is selected as the primary final model and why `Regularized OpenCV` remains a comparison candidate.
- [ ] Define the same `RegularizedCNN` architecture used in `02_2_regularized_cnn_training.ipynb`.
- [ ] Load both checkpoints and evaluate them on the same test split.
- [ ] Save test metrics, confusion matrices, predictions, and misclassified image summaries.

### Task 2: Verify Notebook Structure

**Files:**
- Verify: `notebooks/03_evaluation_analysis.ipynb`

- [ ] Confirm the notebook JSON is valid.
- [ ] Confirm the required checkpoint paths are referenced:
  - `outputs/checkpoints/best_03_regularized_original_cnn.pt`
  - `outputs/checkpoints/best_06_final_opencv_regularized_cnn.pt`
- [ ] Confirm the notebook includes beginner-friendly Korean explanations for accuracy, confusion matrix, precision, and recall.
