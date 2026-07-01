# Dog/Cat Image Classification Project Agent Design

## Goal

Build a beginner-friendly but credible dog/cat image classification project for a Day 2 presentation.
The project should help Jun explain the full machine learning workflow in their own words:

- How train, validation, and test sets are created and used
- How a PyTorch training loop works
- How train/validation loss curves reveal overfitting
- How accuracy, confusion matrix, precision, and recall differ
- How misclassified images can reveal failure patterns

The final deliverables are a result summary document and a presentation deck.

## Project Direction

Use a balanced approach.

The project should stay simple enough for a beginner to explain during an interview-style presentation, while still including enough practical modeling choices to produce meaningful results. The main goal is not to chase the highest possible accuracy. The main goal is to show a thoughtful process.

## Dataset Decision

Use the Oxford-IIIT Pet Dataset.

Reason:

- It does not require a Kaggle login.
- It gives Jun a real chance to create train, validation, and test splits directly.
- It supports a stronger explanation of why each split exists.
- It can be converted from 37 pet breeds into a binary dog/cat classification task.

The presentation should explicitly mention that this dataset was chosen because creating the split directly helps demonstrate understanding of dataset roles.

## Agent Structure

Use a role-separated agent structure.

Root file:

- `AGENTS.md`

Role files:

- `.agents/project-mentor.md`
- `.agents/data-agent.md`
- `.agents/training-agent.md`
- `.agents/evaluation-agent.md`
- `.agents/error-analysis-agent.md`
- `.agents/report-agent.md`
- `.agents/presentation-agent.md`

This structure keeps the project understandable. Each agent owns one area of the assignment and has a clear output.

## Agent Responsibilities

### Project Mentor

Owns the overall learning path and decision flow.

Responsibilities:

- Explain choices in beginner-friendly language
- Keep the work aligned with the assignment requirements
- Ask Jun for decisions when there are meaningful trade-offs
- Connect code results to presentation talking points

### Data Agent

Owns dataset preparation.

Responsibilities:

- Load the Oxford-IIIT Pet Dataset
- Convert breed labels into binary dog/cat labels
- Create train, validation, and test splits
- Check class distribution in each split
- Explain the role of each split in plain language

### Training Agent

Owns model training.

Responsibilities:

- Write the PyTorch training loop directly
- Make the `optimizer.zero_grad -> forward -> loss -> backward -> step` flow visible
- Start with a simple, explainable baseline
- Optionally compare with transfer learning if time allows
- Save training history for later analysis

### Evaluation Agent

Owns metrics and interpretation.

Responsibilities:

- Calculate accuracy
- Build a confusion matrix
- Calculate precision and recall
- Explain when precision matters more and when recall matters more
- Connect metric results to the dog/cat task

### Error Analysis Agent

Owns misclassification review.

Responsibilities:

- Collect incorrectly classified images
- Group failure cases by visible pattern
- Suggest possible causes
- Suggest practical improvement ideas
- Summarize what changed after any improvement attempt

### Report Agent

Owns the written result document.

Responsibilities:

- Turn experiments into a clear report
- Include dataset split explanation
- Include loss curve interpretation
- Include metric interpretation
- Include misclassification analysis
- Keep the writing aligned with the assignment requirements

### Presentation Agent

Owns the Day 2 presentation material.

Responsibilities:

- Create a slide-by-slide story
- Keep slides concise and presentation-friendly
- Include visuals such as split distribution, loss curves, confusion matrix, and misclassified examples
- Prepare speaker notes that Jun can explain naturally

## Proposed Project Folders

```text
Dog_Cat/
  AGENTS.md
  .agents/
    project-mentor.md
    data-agent.md
    training-agent.md
    evaluation-agent.md
    error-analysis-agent.md
    report-agent.md
    presentation-agent.md
  docs/
    planning/
    reports/
    superpowers/
      specs/
  slides/
  src/
  notebooks/
  data/
    raw/
    processed/
  outputs/
    figures/
    metrics/
    misclassified/
```

## Data Flow

1. Download or load the Oxford-IIIT Pet Dataset.
2. Convert original breed labels into binary labels:
   - cat
   - dog
3. Split data into train, validation, and test sets.
4. Train the model on the train set.
5. Tune decisions using validation results.
6. Evaluate final performance on the test set.
7. Save metrics, plots, and misclassified examples.
8. Use saved outputs for the report and presentation.

## Error Handling And Practical Notes

- If dataset download fails, document the issue and use a manually downloaded copy.
- If training is slow, reduce image size, batch size, or epoch count first.
- If results are unstable, fix random seeds and preserve split files.
- If the model overfits, compare train and validation loss and then try simple improvements such as augmentation or transfer learning.

## Verification Plan

The project is considered ready for delivery when:

- Train, validation, and test split counts are saved and explainable
- A PyTorch training loop runs end to end
- Train/validation loss curves are generated
- Accuracy, confusion matrix, precision, and recall are generated
- Misclassified images are saved and analyzed
- A written result document exists
- A Day 2 presentation deck or slide outline exists

## Next Step

After Jun reviews this design, create the implementation plan and then generate the initial project files:

- `AGENTS.md`
- role-specific `.agents/*.md` files
- project folders
- initial planning documents
