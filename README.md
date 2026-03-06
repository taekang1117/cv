# Bolt Classifier — Computer Vision Pipeline

A multi-classifier computer vision pipeline that detects and classifies **bolts** from images using an ensemble of machine learning models.

---

## Table of Contents

- [Overview](#overview)
- [Pipeline](#pipeline)
- [Classifiers](#classifiers)
- [Getting Started](#getting-started)
- [Usage](#usage)

---

## Overview

This project compares three classification algorithms — **Random Forest**, **KNN**, and **SVM** — on object detection data, then averages their predictions to produce a final classification result.

---

## Pipeline

```
Object Detected
      │
      ├──► Random Forest ──► 80% Bolt ──┐
      │                                  │
      ├──► KNN           ──► 60% Bolt ──┼──► Avg 65% Bolt ──► BOLT
      │                                  │
      └──► SVM           ──► 55% Bolt ──┘
```

The pipeline feeds each detected object through three independent classifiers. Their confidence scores are averaged to produce the final prediction.

---

## Classifiers

| Model         | Confidence | Notes                              |
|---------------|------------|------------------------------------|
| Random Forest | 80%        | Highest individual accuracy        |
| KNN           | 60%        | Distance-based classification      |
| SVM           | 55%        | Margin-based classification        |
| **Ensemble**  | **65%**    | Average of all three classifiers   |

---

## Getting Started

### Prerequisites

Make sure you have the following installed:

- Python 3.8+
- `scikit-learn`
- `numpy`
- `opencv-python`

Install dependencies:

```bash
pip install -r requirements.txt
```

### Clone the Repository

**HTTPS:**
```bash
git clone https://github.com/taekang1117/cv.git
cd cv
```

**SSH** (recommended if you have SSH keys set up):
```bash
git clone git@github.com:taekang1117/cv.git
cd cv
```

---

## Usage

```bash
python main.py --input <path-to-image>
```

Example:
```bash
python main.py --input images/sample_bolt.jpg
```

---

## 📁 Project Structure

```
cv/
├── classifiers/
│   ├── random_forest.py
│   ├── knn.py
│   └── svm.py
├── pipeline.py
├── main.py
├── requirements.txt
└── README.md
```

---

## License

This project is open source and available under the [MIT License](LICENSE).
