# Multimodal Fall Detection System Using Wi-Fi CSI and Vision-Based Skeleton Estimation

Multimodal fall detection using Wi-Fi Channel State Information (CSI) and OpenPose-based human skeleton estimation, with unimodal benchmarking, multimodal fusion, and robustness analysis.

> Master thesis, Master in Artificial Intelligence and Emerging Technologies, Faculty of Applied Sciences of Nador, University Mohammed Premier, Oujda (2025/2026). Host company: Caplogy Data.

## Highlights

- **Best CSI model:** ResNet-18 on time-frequency representations of CSI amplitude and phase, 96.0–96.8% test accuracy (depending on the evaluation pipeline, see the results tables).
- **Best vision model:** MobileNetV4 (Conv-Small) with CBAM on OpenPose heatmaps, **98.4% accuracy** (F1-macro 0.984) with only 2.9M parameters.
- **Best fusion:** fixed-weight late fusion with a single grid-searched coefficient (α = 0.45), **98.9% accuracy, F1-macro 0.989**, 98.7% sensitivity and 0.27% false-positive rate.
- **Robustness:** decision-level fusion tolerates strong CSI noise; feature-level fusion tolerates total OpenPose loss. Neither family is safe in both cases.
- **Confidence score:** flags 3.7% of test predictions as low-confidence; accuracy is 99.79% on the rest versus 74.77% on the flagged ones.

## Problem

Single-sensor fall detection has known blind spots. Vision systems fail under occlusion or poor lighting, and Wi-Fi sensing alone is sensitive to noise and interference. This project combines both:

- **Vision:** human skeleton estimation (OpenPose), a privacy-preserving representation because no raw RGB frames are used.
- **Wi-Fi CSI:** channel state information, which captures body motion through its effect on radio multipath propagation, without a camera.

## Dataset: FallDeWideo

Synchronized Wi-Fi CSI and vision dataset (CSI, OpenPose, and Mask R-CNN modalities).

- 6 volunteers, 4 camera angles, 2 environments (a bare room and an office).
- CSI: Intel 5300 NIC, 3 receivers, 30 OFDM subcarriers, 3×3 antennas, sampled at 1000 Hz.
- Video: 20 fps. Each video frame is paired with 50 consecutive CSI packets, which defines one sample.
- OpenPose outputs: 19-channel joint heatmap (18 joints + background) and 38-channel part affinity field, at 36×64 resolution. No raw RGB frames are released.
- Subset used here: 402 five-second recordings, i.e. **40,200 samples**.

| Class | Events | CSI samples | OpenPose sequences |
|-------|--------|-------------|--------------------|
| `fall` | forward, leftward, rightward, backward falls; fall while sitting | 10,900 (27.1%) | 441 (26.6%) |
| `nfall1` | walking, jumping, squatting, bending over, sitting down | 15,000 (37.3%) | 636 (38.3%) |
| `nfall2` | standing up, lying down / getting up (face-down and face-up) | 14,300 (35.6%) | 582 (35.1%) |

The dataset is **not included** in this repository. Download it from Kaggle: `[add FallDeWideo_CSI_RGB link here]`.

## Pipeline

```
FallDeWideo recordings
   ├── Wi-Fi CSI  → amplitude + sanitized phase → time-frequency tensors → CSI classifier (ResNet-18)  ─┐
   │                                                                                                   ├→ fusion → fall / nfall1 / nfall2
   └── OpenPose   → joint heatmaps + PAF (57 channels) → CNN classifier (MobileNetV4 + CBAM)          ─┘
```

**CSI preprocessing:** amplitude and phase extraction from the complex tensors, phase unwrapping and linear detrending along the subcarriers (to remove carrier and sampling frequency offsets), normalization, and construction of the tensors used by each architecture. Data augmentation: Gaussian noise and time/frequency masking.

**OpenPose preprocessing:** 57-channel heatmap tensors (joint heatmaps + part affinity fields), MixUp augmentation for the CNNs, and 28-D keypoint sequences for the recurrent models.

**Training:** class imbalance is handled with a weighted random sampler and label-smoothed cross-entropy (class-weighted cross-entropy for the recurrent models). Macro-F1 is the primary comparison metric.

## Models

| Modality | Models |
|----------|--------|
| CSI, classical ML | Logistic Regression, Random Forest, XGBoost, SVM (RBF) on 44 statistical features |
| CSI, deep learning | CNN + BiGRU, CNN + LSTM, THAR (two-stream Conformer-type transformer), CSI-BERT2, ResNet-18 |
| OpenPose, heatmap CNNs | MobileNetV2 + CBAM, MobileNetV3-Large + CBAM, MobileNetV4 Conv-Small + CBAM |
| OpenPose, sequences | LSTM, GRU on 28-D keypoint sequences |
| Fusion | Late fusion (learned), late fusion (fixed weight), gated average fusion, feature concatenation, cross-attention fusion |

## Results

### Wi-Fi CSI (unimodal, held-out test set of 6,030 samples)

| Model | Test accuracy | Test macro-F1 |
|-------|---------------|---------------|
| Logistic Regression | 0.5663 | 0.5625 |
| SVM (RBF) | 0.6871 | 0.6858 |
| XGBoost | 0.8342 | 0.8350 |
| Random Forest | 0.8471 | 0.8482 |
| CNN + LSTM | 0.9199 | 0.9228 |
| CNN + BiGRU | 0.9280 | 0.9304 |
| THAR | 0.9328 | 0.9354 |
| CSI-BERT2 | 0.9478 | 0.9497 |
| **ResNet-18** | **0.9602** | **0.9619** |

ResNet-18 reaches 0.9675 accuracy / 0.9684 macro-F1 in the fusion pipeline, which uses its own train/validation/test split.

### OpenPose (unimodal)

| Model | Input | Parameters | Test accuracy | Test macro-F1 |
|-------|-------|-----------|---------------|---------------|
| LSTM | 28-D keypoint sequences | 1,111,555 | 0.8554 | 0.8556 |
| GRU | 28-D keypoint sequences | 834,051 | 0.8554 | 0.8580 |
| MobileNetV2 + CBAM | 57-channel heatmap | 2,608,677 | 0.9788 | 0.9787 |
| MobileNetV3 + CBAM | 57-channel heatmap | 4,579,061 | 0.9788 | 0.9784 |
| **MobileNetV4 + CBAM** | 57-channel heatmap | 2,877,829 | **0.9818** | **0.9817** |

Treating OpenPose heatmaps as an image classification problem is clearly better than modeling keypoint trajectories with recurrent networks on this dataset. In the fusion pipeline, MobileNetV4 reaches 0.9841 accuracy / 0.9840 macro-F1.

### Multimodal fusion

| Model | Trainable params | Test accuracy | Test macro-F1 |
|-------|------------------|---------------|---------------|
| ResNet-18 (CSI only) | – | 0.9675 | 0.9684 |
| MobileNetV4 (OpenPose only) | – | 0.9841 | 0.9840 |
| Late fusion (learned) | 163 | 0.9821 | 0.9828 |
| **Late fusion (fixed weight, α = 0.45)** | **1** | **0.9887** | **0.9888** |
| Gated average fusion | 657,155 | 0.9706 | 0.9715 |
| Feature concatenation | 229,891 | 0.9716 | 0.9725 |
| Cross-attention fusion | 1,052,675 | 0.9698 | 0.9707 |

Only the single-parameter fixed-weight late fusion improves on the best unimodal model (+0.47 F1 points). The three learned feature-level fusions do not, which points to a ceiling effect: the OpenPose branch is already near-saturated, so a more complex fusion head has little to gain and can add optimization noise.

The fused prediction is `p = α · p_CSI + (1 − α) · p_OpenPose`, so each modality's contribution is exactly known (glass-box, no post-hoc attribution needed).

### Clinical metrics (fall vs non-fall)

| Model | Sensitivity | Specificity | FPR | Precision | F1 (fall) |
|-------|-------------|-------------|-----|-----------|-----------|
| ResNet-18 (CSI only) | 0.9725 | 0.9943 | 0.0057 | 0.9845 | 0.9785 |
| MobileNetV4 (OpenPose only) | 0.9804 | 0.9954 | 0.0046 | 0.9877 | 0.9840 |
| Late fusion (learned) | **0.9969** | 0.9948 | 0.0052 | 0.9861 | **0.9915** |
| Late fusion (fixed weight) | 0.9872 | **0.9973** | **0.0027** | **0.9926** | 0.9899 |
| Gated average fusion | 0.9761 | 0.9948 | 0.0052 | 0.9858 | 0.9809 |
| Feature concatenation | 0.9774 | 0.9950 | 0.0050 | 0.9864 | 0.9819 |
| Cross-attention fusion | 0.9743 | 0.9948 | 0.0052 | 0.9858 | 0.9800 |

Two useful operating points: the learned late fusion misses the fewest falls (highest sensitivity, suited to hospital or high-risk settings), while the fixed-weight late fusion raises the fewest false alarms (suited to home monitoring where alarm fatigue matters). For the fixed-weight model, 1,614 of 1,635 falls are detected, and 12 of 4,395 non-falls are wrongly flagged as falls.

## Robustness analysis

Each sensor was degraded one at a time (CSI: additive Gaussian noise of standard deviation σ; OpenPose: random channel occlusion of fraction f). Values are F1-macro.

**CSI noise (OpenPose kept clean)**

| σ | CSI only | OpenPose only | Late (learned) | Late (fixed) | Gated | Concat | Cross-attn |
|---|----------|---------------|----------------|--------------|-------|--------|------------|
| 0.0 | 0.9684 | 0.9840 | 0.9828 | 0.9888 | 0.9715 | 0.9725 | 0.9707 |
| 1.0 | 0.6315 | 0.9840 | 0.8296 | 0.9830 | 0.6859 | 0.7035 | 0.6653 |
| 4.0 | 0.1817 | 0.9840 | 0.5605 | 0.9758 | 0.1861 | 0.1904 | 0.1829 |

**OpenPose occlusion (CSI kept clean)**

| f | CSI only | OpenPose only | Late (learned) | Late (fixed) | Gated | Concat | Cross-attn |
|---|----------|---------------|----------------|--------------|-------|--------|------------|
| 0.0 | 0.9684 | 0.9840 | 0.9828 | 0.9888 | 0.9715 | 0.9725 | 0.9707 |
| 0.6 | 0.9684 | 0.7787 | 0.8755 | 0.9127 | 0.9696 | 0.9699 | 0.9691 |
| 1.0 | 0.9684 | 0.1422 | 0.1422 | 0.1422 | 0.9684 | 0.9683 | 0.9684 |

Main finding, an asymmetric fault-tolerance profile:

- **Decision-level (late) fusion** is robust to CSI noise (the fixed-weight variant still reaches F1 = 0.976 at σ = 4) but collapses when OpenPose is totally lost.
- **Feature-level fusion** (gated, concatenation, cross-attention) is the mirror image: it falls back to CSI-only performance under total OpenPose loss but collapses under strong CSI noise.

The choice of fusion method therefore depends on which sensor is more likely to fail in the target deployment.

### Confidence score

The mean maximum softmax probability of the fused output is 0.900 on clean data. With a 0.6 threshold, 222 of 6,030 predictions (3.7%) are flagged as low-confidence: accuracy is **99.79%** on the remaining predictions versus **74.77%** on the flagged ones, so errors concentrate where the score signals low reliability. The score also drops as sensors degrade (0.900 clean, 0.665 under the strongest CSI noise, 0.609 under total OpenPose occlusion), which can trigger human review or a conservative fallback.

## Limitations

- The paired samples used for fusion carry no subject identifier, so subject-independent generalization at the fusion stage could not be verified. Fusion results measure generalization within the provided split.
- Fusion heads were trained on frozen backbones (no joint fine-tuning), which caps what feature-level fusion can reach.
- Each model was trained with a single random seed, so small differences (for example among the MobileNet variants) are not statistically established. No CBAM on/off ablation was run.
- Robustness tests degrade one modality at a time with synthetic noise and occlusion, not measured real-world degradation, and joint degradation was not tested.
- Only three coarse classes are used, and the data comes from a small number of volunteers.

## Future work

- Jointly fine-tune both backbones with the best fusion head.
- Evaluate fusion with leave-one-subject-out validation.
- Test simultaneous degradation of both sensors, ideally on real degraded recordings.
- Fall localization in room coordinates (2D/3D) using the Mask R-CNN modality and camera calibration.

## Repository structure

```
├── CSI/        # Wi-Fi CSI preprocessing and models (e.g. csi-all-models.ipynb)
├── Vision/     # OpenPose / skeleton models (e.g. falldewideo-dataset.ipynb)
├── Fusion/     # Multimodal fusion and robustness experiments
├── docs/       # Thesis (PDF) and figures
├── requirements.txt
└── README.md
```

## How to run

All experiments were run on Kaggle notebooks with a single NVIDIA Tesla T4 GPU (16 GB), Python 3.12, PyTorch 2.6+, and timm 1.0.26.

1. Clone the repository or upload the notebooks to Kaggle.
2. Install the requirements: `pip install -r requirements.txt`
3. Download the FallDeWideo dataset and update the data paths at the top of each notebook.
4. Run the notebooks in this order: `Vision/` → `CSI/` → `Fusion/`.

## Tech stack

PyTorch, torchvision, timm, scikit-learn, XGBoost, NumPy, pandas, Matplotlib, OpenCV.

## Thesis

Full text: `[add a link to the thesis PDF in docs/ here]`

## Author

**Imane Mokhtari**, Master in Artificial Intelligence and Emerging Technologies, FSA Nador, University Mohammed Premier, Oujda, 2025/2026.
Supervisor: Pr. Siham Essahraui.

**Keywords:** fall detection, Wi-Fi CSI, OpenPose, multimodal fusion, deep learning
