# Multimodal Fall Detection System Using Wi-Fi CSI and Vision-Based Skeleton Estimation

Multimodal fall detection using Wi-Fi Channel State Information (CSI) and OpenPose-based human skeleton estimation, with unimodal benchmarking, multimodal fusion, and robustness analysis.

## Overview

This project (master's thesis) detects falls by combining two sources:

- **Vision**: human skeleton keypoints extracted from RGB video with OpenPose.
- **Wi-Fi CSI**: channel state information that captures body movement without a camera.

Both modalities are first evaluated separately, then fused, and finally tested under degraded conditions.

## Repository structure

├── CSI/       # Wi-Fi CSI preprocessing and models
├── Vision/    # Skeleton estimation and vision models
├── Fusion/    # Multimodal fusion experiments
├── docs/      # Thesis and figures
└── README.md

## Dataset

FallDeWideo (CSI + RGB): [lien Kaggle]

The dataset is not included in this repository because of its size. Download it and update the paths in the notebooks.

## Methods

- Vision: OpenPose skeleton keypoints + [tes modèles]
- CSI: [prétraitement] + [tes modèles]
- Fusion: [early / late / feature-level fusion]
- Robustness analysis: [bruit, données manquantes, etc.]

## Results

| Model | Modality | Accuracy | F1-score |
|-------|----------|----------|----------|
| [ ]   | CSI      |          |          |
| [ ]   | Vision   |          |          |
| [ ]   | Fusion   |          |          |

## How to run

1. Clone the repository
2. Install the requirements: `pip install -r requirements.txt`
3. Download the dataset and set the paths
4. Run the notebooks in the order: `Vision/` → `CSI/` → `Fusion/`

## Author

Imane [nom] — [université], [année]
