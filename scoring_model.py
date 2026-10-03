import os
import json
import numpy as np
import pandas as pd
import glob
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, confusion_matrix
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    TF_AVAILABLE = True
except ImportError:
    print("TensorFlow not found. LSTM model will not be trained.")
    TF_AVAILABLE = False

def calculate_iou(start1, end1, start2, end2):
    """Compute Intersection-over-Union between two frame intervals."""
    intersection = max(0, min(end1, end2) - max(start1, start2))
    union = (end1 - start1) + (end2 - start2) - intersection
    return intersection / union if union > 0 else 0

def load_ground_truth(jsonl_path):
    """
    Load per-rep ground-truth scores and frame boundaries from the JSONL file.
    Returns a dict keyed by (subject_id, exercise_id) → list of rep dicts.
    """
    gt_dict = {}
    if not os.path.exists(jsonl_path):
        return gt_dict
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            sub = data['subject_id']
            ex = data['exercise_id']
            if (sub, ex) not in gt_dict:
                gt_dict[(sub, ex)] = []
            for rep in data['scores']:
                gt_dict[(sub, ex)].append(rep)
    return gt_dict

def prepare_real_dataset(gt_dict, num_features=50):
    """
    Match extracted reps (from reps/ directory) to ground-truth reps via IoU > 30%.
    Resizes each rep's 1D signal to `num_features` and normalises it.
    Returns X (feature matrix) and y (score labels).
    Also returns iou_scores list for Segmentation IoU metric.
    """
    print("Loading extracted features from reps/ directory...")
    X, y, iou_scores = [], [], []
    json_files = glob.glob("reps/*/*/temporal_segments.json")
    
    match_count = 0
    for jpath in json_files:
        parts = jpath.split(os.sep)
        sub, ex = parts[-3], parts[-2]
        
        if (sub, ex) not in gt_dict:
            continue
            
        with open(jpath, 'r') as f:
            segments = json.load(f)
            
        out_dir = os.path.dirname(jpath)
        for seg in segments:
            sf, ef = seg['start_frame'], seg['end_frame']
            
            # Match to ground truth rep via IoU
            best_iou = 0
            best_score = None
            for gt_rep in gt_dict[(sub, ex)]:
                iou = calculate_iou(sf, ef, gt_rep['init_frame'], gt_rep['final_frame'])
                if iou > best_iou:
                    best_iou = iou
                    best_score = gt_rep['score']
            
            if best_iou > 0.3 and best_score is not None:
                feat_path = os.path.join(out_dir, seg['feature_file'])
                if os.path.exists(feat_path):
                    cy = np.load(feat_path)
                    if len(cy) > 0:
                        x_old = np.linspace(0, 1, len(cy))
                        x_new = np.linspace(0, 1, num_features)
                        cy_resized = np.interp(x_new, x_old, cy)
                        cy_resized = (cy_resized - np.mean(cy_resized)) / (np.std(cy_resized) + 1e-8)
                        
                        X.append(cy_resized)
                        y.append(best_score)
                        iou_scores.append(best_iou)
                        match_count += 1
                        
    print(f"Matched {match_count} extracted reps with ground-truth labels.")
    return np.array(X), np.array(y), iou_scores

def compute_all_metrics(y_true, y_pred_raw, iou_scores=None):
    """
    Compute 5 evaluation metrics for a scoring model:

    1. MAE (Mean Absolute Error):
       Average absolute difference between predicted and true scores.
       Lower is better.

    2. Accuracy ±1 (%):
       Percentage of predictions within 1 point of the true score.
       Clinically, ±1 error is usually acceptable.

    3. Pearson Correlation:
       Linear correlation between predicted and true scores.
       Measures if model correctly captures score magnitude trends.

    4. Spearman Correlation:
       Rank-order correlation (non-parametric).
       Measures if model correctly ranks reps from worst to best.

    5. Segmentation IoU (mean):
       Average IoU between extracted rep boundaries and ground-truth boundaries.
       Measures quality of the temporal segmentation step (not the scorer).
       Only available when iou_scores are passed in.
    """
    y_pred = np.round(y_pred_raw)

    mae = mean_absolute_error(y_true, y_pred)
    acc_1 = np.mean(np.abs(y_true - y_pred) <= 1) * 100
    pearson_r, pearson_p = pearsonr(y_true, y_pred_raw)
    spearman_r, spearman_p = spearmanr(y_true, y_pred_raw)
    seg_iou = np.mean(iou_scores) * 100 if iou_scores else None

    return {
        "MAE": mae,
        "Accuracy_±1 (%)": acc_1,
        "Pearson_r": pearson_r,
        "Pearson_p": pearson_p,
        "Spearman_r": spearman_r,
        "Spearman_p": spearman_p,
        "Segmentation_IoU (%)": seg_iou,
    }

def print_metrics(model_name, metrics, y_true, y_pred_raw):
    """Pretty-print all 5 metrics + confusion matrix for a model."""
    y_pred = np.round(y_pred_raw).astype(int)
    y_true_int = np.round(y_true).astype(int)

    print(f"\n{'='*50}")
    print(f"  {model_name} — Evaluation Metrics")
    print(f"{'='*50}")
    print(f"  1. MAE (↓ better)          : {metrics['MAE']:.3f}")
    print(f"  2. Accuracy ±1  (↑ better) : {metrics['Accuracy_±1 (%)']:.1f}%")
    print(f"  3. Pearson r    (↑ better) : {metrics['Pearson_r']:.3f}  (p={metrics['Pearson_p']:.3f})")
    print(f"  4. Spearman r   (↑ better) : {metrics['Spearman_r']:.3f}  (p={metrics['Spearman_p']:.3f})")
    if metrics['Segmentation_IoU (%)'] is not None:
        print(f"  5. Segmentation IoU        : {metrics['Segmentation_IoU (%)']:.1f}%")
    
    print(f"\n  Confusion Matrix (rows=true, cols=pred):")
    all_labels = sorted(set(y_true_int.tolist() + y_pred.tolist()))
    cm = confusion_matrix(y_true_int, y_pred, labels=all_labels)
    header = "       " + "  ".join(f"P{l}" for l in all_labels)
    print(f"  {header}")
    for label, row in zip(all_labels, cm):
        row_str = "  ".join(f"{v:3d}" for v in row)
        print(f"  T{label} |  {row_str}")

def train_evaluate_rf(X_train, X_test, y_train, y_test, iou_scores_test):
    """Train Random Forest regressor and evaluate with all 5 metrics."""
    print("\n--- Training Random Forest ---")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    metrics = compute_all_metrics(y_test, preds, iou_scores_test)
    print_metrics("Random Forest", metrics, y_test, preds)
    return metrics

def train_evaluate_svm(X_train, X_test, y_train, y_test, iou_scores_test):
    """Train SVM regressor and evaluate with all 5 metrics."""
    print("\n--- Training SVM ---")
    svm = SVR(kernel='rbf', C=10)
    svm.fit(X_train, y_train)
    preds = svm.predict(X_test)
    metrics = compute_all_metrics(y_test, preds, iou_scores_test)
    print_metrics("SVM (RBF)", metrics, y_test, preds)
    return metrics

def train_evaluate_lstm(X_train, X_test, y_train, y_test, iou_scores_test):
    """Train LSTM regressor and evaluate with all 5 metrics. Requires TensorFlow."""
    if not TF_AVAILABLE:
        return None
        
    print("\n--- Training LSTM ---")
    X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_lstm  = X_test.reshape((X_test.shape[0],  X_test.shape[1],  1))
    
    model = Sequential([
        LSTM(64, input_shape=(X_train_lstm.shape[1], 1), return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.fit(X_train_lstm, y_train, epochs=20, batch_size=16,
              validation_data=(X_test_lstm, y_test), verbose=0)
    
    preds = model.predict(X_test_lstm, verbose=0).flatten()
    metrics = compute_all_metrics(y_test, preds, iou_scores_test)
    print_metrics("LSTM", metrics, y_test, preds)
    return metrics

if __name__ == "__main__":
    gt_dict = load_ground_truth("ucophyrehab2_data.jsonl")
    X, y, iou_scores = prepare_real_dataset(gt_dict, num_features=50)
    
    if len(X) < 10:
        print("Not enough data to train. Please run feature extraction on all videos first.")
        exit(1)
        
    # Split indices so we can track iou_scores for test set
    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    iou_test = [iou_scores[i] for i in test_idx]

    print(f"\nTraining on {len(X_train)} samples, testing on {len(X_test)} samples.")
    print(f"Mean Segmentation IoU (all matched reps): {np.mean(iou_scores)*100:.1f}%")
    
    rf_metrics  = train_evaluate_rf(X_train, X_test, y_train, y_test, iou_test)
    svm_metrics = train_evaluate_svm(X_train, X_test, y_train, y_test, iou_test)
    lstm_metrics = train_evaluate_lstm(X_train, X_test, y_train, y_test, iou_test)
    
    # Save consolidated results
    results_dict = {
        "Random_Forest": {k: (round(float(v), 4) if v is not None else None) for k, v in rf_metrics.items()},
        "SVM":           {k: (round(float(v), 4) if v is not None else None) for k, v in svm_metrics.items()},
        "LSTM":          {k: (round(float(v), 4) if v is not None else None) for k, v in lstm_metrics.items()} if lstm_metrics else "N/A",
    }
    with open("pipeline_results.json", "w") as f:
        json.dump(results_dict, f, indent=4)
    print("\n\nResults successfully saved to pipeline_results.json")
