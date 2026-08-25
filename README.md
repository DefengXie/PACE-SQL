# PACE-SQL

**Complete or Abstain? Aligning SQL Completion with User Preferences**

> 🚧 **Code and datasets are coming soon. Stay tuned!** 🚧

---

## 📖 Overview

**PACE-SQL** is a **P**reference-**A**ligned SQL **C**ode compl**E**tion framework that improves both completion accuracy and alignment with real user preferences. Unlike existing SQL code completion solutions that always attempt to produce a suggestion, PACE-SQL learns **when to complete** and **when to abstain** — significantly boosting real-world acceptance rate.

### ✨ Key Highlights

- 🎯 **Preference-Aligned Completion**: Learns to abstain when no suggestion is needed, avoiding unnecessary interruptions.
- 🗄️ **SQL-Specific Training**: Continual pretraining + supervised fine-tuning tailored for SQL semantics and FIM (fill-in-the-middle) scenarios.
- 🏆 **Reinforcement Learning with a Custom Reward**: A specially designed reward function to align model behavior with authentic developer preferences.
- 📊 **Two New Datasets**:
  - **WeSQL-1.5M** — 1.5M samples from real-world SQL editing interaction logs, capturing genuine user preferences (both completion and abstention).
  - **OpenSQL-FIM** — an open-source SQL FIM dataset curated from public corpora.

---

## 🚀 Results

On the **WeSQL-1.5M** benchmark, PACE-SQL outperforms the strongest baseline by:

| Metric | Improvement |
| :--- | :--- |
| Exact Match (EM) | **+9.67%** |
| Edit Similarity (ES) | **+7.52%** |

Deployed online, PACE-SQL raises the **user acceptance rate from 5% to 29%**.

---

## 🧩 Framework

PACE-SQL is built on three key stages:

1. **Continual Pretraining** on large-scale SQL corpora to inject SQL-specific knowledge.
2. **Supervised Fine-Tuning** on completion + abstention samples for FIM-style SQL editing.
3. **Preference Alignment via RL** using a reward function designed to encourage completion when helpful and abstention when not.

---

## 📦 Release Plan

| Item | Status |
| :--- | :--- |
| 📄 Paper | Coming soon |
| 💾 OpenSQL-FIM dataset | Coming soon |
| 🛠️ evaluation code | Coming soon |

We are actively preparing the release. Please **⭐ Star** and **👀 Watch** this repository to get notified when materials are available.

---

## 📬 Contact

For questions, collaborations, or early access requests, please open an issue in this repository.

---

## 📄 Citation

A BibTeX citation will be provided upon paper release.

```
@article{pace-sql,
  title   = {Complete or Abstain? Aligning SQL Completion with User Preferences},
  author  = {Anonymous},
  year    = {2026},
  note    = {Preprint. Code and datasets coming soon.}
}
```
