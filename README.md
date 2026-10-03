# Rehab Exercise Assessment

This project builds a computer-vision system that analyses images or video frames of a person performing selected exercises and classifies their posture as correct or potentially incorrect, giving simple form-feedback cues (e.g. “align your back”, “knee going too far forward”) to support safer home workouts.

**Dataset:** [UcoPhyRehab++](https://zenodo.org/records/17935737) — 27 subjects, 16 rehabilitation exercises, 5 camera angles, scored 1–5 per rep by physiotherapists.

---

## Setup
Before running the code, you need to download the dataset:
1. Go to the [UcoPhyRehab++ repository](https://zenodo.org/records/17935737).
2. Download the `semantic_segmentation` folder.
3. Place the `semantic_segmentation` folder directly in the main directory of this project (alongside `run_pipeline.sh`).

---

## How to Run

```bash
bash run_pipeline.sh
# OR separately:
python3 feature_extraction_opencv.py   # Step 1
python3 scoring_model.py               # Step 2
```

---

## Project Structure

```
dip/
├── feature_extraction_opencv.py   # Step 1: Segment video into per-rep clips + signals
├── scoring_model.py               # Step 2: Train & evaluate ML scoring models
├── run_pipeline.sh                # Runs Step 1 → Step 2 end-to-end
├── semantic_segmentation/         # Input: pre-generated semantic segmentation videos
│   └── <subject>/<exercise>/cam0.mp4
└── ucophyrehab2_data.jsonl        # Ground truth: per-rep scores + frame boundaries
```

---

## Step 1 — `feature_extraction_opencv.py`

**Purpose:** Unsupervised temporal segmentation — automatically slices a full exercise video into individual repetitions.

**Input:** `semantic_segmentation/<subject>/<exercise>/cam0.mp4` (pre-segmented video with coloured body regions)

**Output per exercise:**
- `rep_N.mp4` — trimmed video clip of rep N
- `rep_N_features.npy` — 1D smoothed motion signal for rep N (used as ML input)
- `temporal_segments.json` — list of `{rep_idx, start_frame, end_frame, signal_type}`

**How it works:**
1. Extracts dominant semantic colour from the first frame
2. Builds **4 parallel motion signals** from that colour region:
   - `cy` — Y-centroid (captures vertical motion: leg raises, squats)
   - `cx` — X-centroid (captures horizontal motion: arm swings, pendulum)
   - `area` — mask pixel area (captures body expansion: squats, knee bends)
   - `optical_flow` — mean Farneback flow magnitude (captures any motion)
3. Picks the signal with highest variance as the primary segmentation signal
4. Applies Butterworth low-pass filter (cutoff=1.5Hz, fs=30fps)
5. Detects **valleys** = rep boundaries via `scipy.signal.find_peaks`
6. Saves clips and `.npy` signals between consecutive valleys

> **Note:** No ground truth is used in this step — segmentation is fully unsupervised.

---

## Step 2 — `scoring_model.py`

**Purpose:** Train ML models to predict physiotherapist score (1–5) for each rep from its 1D motion signal.

**Input:**
- `reps/*/temporal_segments.json` + `.npy` feature files (from Step 1)
- `ucophyrehab2_data.jsonl` — ground truth scores and frame boundaries

**How matching works:**
- Each extracted rep is matched to the closest GT rep using **IoU > 30%** on frame indices
- Matched reps' signals are resampled to 50 features and z-score normalised

**Models trained:** Random Forest · SVM (RBF) · LSTM (optional, requires TensorFlow)

**Evaluation — 5 Metrics:**

| # | Metric | What it measures |
|---|--------|-----------------|
| 1 | **MAE** | Average score error (lower = better) |
| 2 | **Accuracy ±1** | % of preds within 1 point of GT |
| 3 | **Pearson r** | Linear correlation of scores |
| 4 | **Spearman r** | Rank-order correlation (which reps are better/worse) |
| 5 | **Segmentation IoU** | Quality of Step 1 rep boundaries vs GT boundaries |

Confusion matrix (score 1–5) is also printed per model.

**Output:** `pipeline_results.json` with all metrics for each model.

---

## Limitations / Known Issues

- Folders with no detected reps produce an empty `temporal_segments.json` (segmentation failed — usually due to low motion variance or exercise type mismatch)
- Segmentation currently uses `cam0` only; adding more camera angles could improve robustness
- Scoring model uses only 1D time-series signal; body-part-specific angle features (from `dataset_3d_with_angles.json`) are not yet incorporated

---

## Planned: LLM-based Body Part Feedback

Ongoing