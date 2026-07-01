# Development Environment

## Python

Use Python 3.12 for this project.

## Package Roles

### torch

`torch` is the core deep learning library.
It is used for tensors, model layers, loss calculation, backpropagation, and optimizer updates.

In this assignment, this is the package that makes the direct PyTorch training loop possible:

```text
optimizer.zero_grad -> forward -> loss -> backward -> step
```

### torchvision

`torchvision` supports image-focused deep learning work.
It provides image transforms, common vision datasets, and pretrained image models.

In this assignment, it will be used to load the Oxford-IIIT Pet Dataset and prepare images for training.

### scikit-learn

`scikit-learn` provides common machine learning utilities.

In this assignment, it will be used for:

- train/validation/test splitting
- confusion matrix
- precision
- recall
- accuracy checks

### pandas

`pandas` is used for table-shaped data.

In this assignment, it will be used to save and inspect:

- image paths
- dog/cat labels
- split names
- metric summaries

### matplotlib

`matplotlib` is used for plots.

In this assignment, it will be used to create:

- train/validation loss curves
- class distribution charts
- confusion matrix figures

### pillow

`pillow` is the Python image library.

In this assignment, it helps load and inspect image files.
It is also used internally by many image datasets and transforms.

### tqdm

`tqdm` shows progress bars.

In this assignment, it makes long-running training and evaluation loops easier to monitor.

## Installation

Create a virtual environment and install packages:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

This project starts with the CPU build of PyTorch because it is the simplest and most stable option for a first pass.
CUDA/GPU support can be added later if training speed becomes a problem.
