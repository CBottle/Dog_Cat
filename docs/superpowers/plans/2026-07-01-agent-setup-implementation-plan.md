# Agent Setup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create the initial agent guidance files and project folder structure for the Dog/Cat image classification assignment.

**Architecture:** The root `AGENTS.md` explains the shared project rules and points to focused role files under `.agents/`. Each role file owns one assignment area so future work can stay easy to reason about.

**Tech Stack:** Markdown documentation, Git, PyTorch project planning.

---

## File Structure

- Create: `AGENTS.md` as the root collaboration guide.
- Create: `.agents/project-mentor.md` for overall learning and decision flow.
- Create: `.agents/data-agent.md` for Oxford-IIIT Pet dataset preparation.
- Create: `.agents/training-agent.md` for PyTorch training guidance.
- Create: `.agents/evaluation-agent.md` for metric calculation and interpretation.
- Create: `.agents/error-analysis-agent.md` for misclassified image review.
- Create: `.agents/report-agent.md` for the result document.
- Create: `.agents/presentation-agent.md` for the Day 2 slide deck.
- Create directories: `docs/planning`, `docs/reports`, `slides`, `src`, `notebooks`, `data/raw`, `data/processed`, `outputs/figures`, `outputs/metrics`, `outputs/misclassified`.

### Task 1: Add Root Agent Guide

**Files:**
- Create: `AGENTS.md`

- [ ] **Step 1: Write the root guide**

Create `AGENTS.md` with the project mission, confirmed decisions, agent map, operating rules, and next workflow.

- [ ] **Step 2: Verify the file exists**

Run: `Test-Path AGENTS.md`
Expected: `True`

### Task 2: Add Role Agent Files

**Files:**
- Create: `.agents/project-mentor.md`
- Create: `.agents/data-agent.md`
- Create: `.agents/training-agent.md`
- Create: `.agents/evaluation-agent.md`
- Create: `.agents/error-analysis-agent.md`
- Create: `.agents/report-agent.md`
- Create: `.agents/presentation-agent.md`

- [ ] **Step 1: Write each role file**

Each file must include:

- Purpose
- Responsibilities
- Inputs
- Outputs
- Beginner explanation rule
- Done criteria

- [ ] **Step 2: Verify all role files exist**

Run: `Get-ChildItem .agents -File`
Expected: the seven role files are listed.

### Task 3: Create Project Folders

**Files:**
- Create directories listed in the file structure section.

- [ ] **Step 1: Create directories**

Run: `New-Item -ItemType Directory -Force docs/planning, docs/reports, slides, src, notebooks, data/raw, data/processed, outputs/figures, outputs/metrics, outputs/misclassified`
Expected: directories exist.

- [ ] **Step 2: Preserve empty directories**

Create `.gitkeep` files in directories that are otherwise empty so Git can track the project structure.

### Task 4: Verify And Commit

**Files:**
- Modify Git index and create commit.

- [ ] **Step 1: Check status**

Run: `git status --short`
Expected: new documentation and folder marker files appear.

- [ ] **Step 2: Commit**

Run:

```bash
git add AGENTS.md .agents docs slides src notebooks data outputs
git commit -m "Add project agent structure"
```

Expected: commit succeeds on `agent-setup`.
