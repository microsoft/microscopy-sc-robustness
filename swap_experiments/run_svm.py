import os
import pickle
import numpy as np
import csv
import random
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed

# --- Config ---
SEED = 42
VERBOSE = True # Set to False to diasable printing
np.random.seed(SEED)
random.seed(SEED)

LOC_KEYS = [
    "actin", "bud neck", "cell periphery", "cytoplasm", "endosome", "ER", "Golgi", "mitochondrion",
    "nucleus", "nuclear periphery", "nucleolus", "spindle pole", "peroxisome", "vacuolar membrane", "vacuole"
]

# --- Evaluation ---
def evaluate_model(train_idx, test_idx, features, labels):
    X_train, X_test = features[train_idx], features[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]
    model = SVC(kernel='linear', max_iter=100000, random_state=SEED)
    model.fit(X_train, y_train)
    return accuracy_score(y_test, model.predict(X_test))

def get_nn_score_svc(pkl1, pkl2, len1=1000, len2=1000):
    if not os.path.exists(pkl1) or not os.path.exists(pkl2):
        if VERBOSE:
            print(f"Missing file: {pkl1 if not os.path.exists(pkl1) else pkl2}")
        return None, None

    with open(pkl1, 'rb') as f:
        features1 = np.vstack(pickle.load(f, encoding='latin1'))[:len1]
    with open(pkl2, 'rb') as f:
        features2 = np.vstack(pickle.load(f, encoding='latin1'))[:len2]

    features = np.vstack([features1, features2])
    labels = np.array([0] * len1 + [1] * len2)

    skf = StratifiedKFold(n_splits=5)

    accuracies = []
    for train, test in skf.split(features, labels):
        acc = evaluate_model(train, test, features, labels)
        accuracies.append(acc)

    return np.mean(accuracies), np.std(accuracies)

# --- CSV Utilities ---
def row_exists(file_path, *values):
    if not os.path.exists(file_path):
        return False
    with open(file_path, "r") as f:
        return any(row[:len(values)] == list(values) for row in csv.reader(f))

def write_header_if_needed(file_path, header):
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        with open(file_path, "w", newline="") as f:
            csv.writer(f).writerow(header)

def append_result(file_path, row):
    with open(file_path, "a", newline="") as f:
        csv.writer(f).writerow(row)

# --- Evaluators ---
def eval_triplet(base_dir, layer, subfolder, output_csv, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, output_csv)
    write_header_if_needed(path, ["Location1", "Location2", "Background", "Accuracy", "StdDev"])

    for i, loc1 in enumerate(LOC_KEYS):
        for j, loc2 in enumerate(LOC_KEYS):
            if j <= i:
                continue
            for loc3 in LOC_KEYS:
                if row_exists(path, loc1, loc2, loc3):
                    continue
                pkl1 = os.path.join(base_dir, subfolder, layer, f"{loc1}_{loc3}.pkl")
                pkl2 = os.path.join(base_dir, subfolder, layer, f"{loc2}_{loc3}.pkl")
                acc, std = get_nn_score_svc(pkl1, pkl2)
                if acc is not None:
                    append_result(path, [loc1, loc2, loc3, acc, std])
                    if VERBOSE:
                        print(f"{loc1} vs {loc2} | bg: {loc3} => {acc:.4f}")

def eval_pairwise(base_dir, layer, subfolder, output_csv, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, output_csv)
    write_header_if_needed(path, ["Location1", "Location2", "Accuracy", "StdDev"])

    for i, loc1 in enumerate(LOC_KEYS):
        for j, loc2 in enumerate(LOC_KEYS):
            if j <= i:
                continue
            if row_exists(path, loc1, loc2):
                continue
            pkl1 = os.path.join(base_dir, subfolder, layer, f"{loc1}.pkl")
            pkl2 = os.path.join(base_dir, subfolder, layer, f"{loc2}.pkl")
            acc, std = get_nn_score_svc(pkl1, pkl2)
            if acc is not None:
                append_result(path, [loc1, loc2, acc, std])
                if VERBOSE:
                    print(f"{loc1} vs {loc2} => {acc:.4f}")

def eval_same_localization_swap(base_dir, layer, subfolder, output_csv, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, output_csv)
    write_header_if_needed(path, ["Location1", "Location2", "Accuracy", "StdDev"])

    for i, loc1 in enumerate(LOC_KEYS):
        for j, loc2 in enumerate(LOC_KEYS):
            if j <= i:
                continue
            if row_exists(path, loc1, loc2):
                continue
            pkl1 = os.path.join(base_dir, subfolder, layer, f"{loc1}_{loc1}.pkl")
            pkl2 = os.path.join(base_dir, subfolder, layer, f"{loc2}_{loc2}.pkl")
            acc, std = get_nn_score_svc(pkl1, pkl2)
            if acc is not None:
                append_result(path, [loc1, loc2, acc, std])
                if VERBOSE:
                    print(f"{loc1} vs {loc2} (same-loc) => {acc:.4f}")

# --- Main Driver ---
def get_results_svm(base_dir, conv_layer, output_dir):
    if VERBOSE:
        print(f"\n>>> Running evaluation for: {base_dir} (Layer: {conv_layer})")

    eval_triplet(base_dir, conv_layer, subfolder="logs", output_csv="results_diff.csv", output_dir=output_dir)
    eval_triplet(base_dir, conv_layer, subfolder="logs_batch", output_csv="results_batch.csv", output_dir=output_dir)
    eval_same_localization_swap(base_dir, conv_layer, subfolder="logs", output_csv="results_same.csv", output_dir=output_dir)
    eval_pairwise(base_dir, conv_layer, subfolder="logs_base", output_csv="results_base.csv", output_dir=output_dir)
    eval_pairwise(base_dir, conv_layer, subfolder="logs_mask", output_csv="results_mask.csv", output_dir=output_dir)

if __name__ == "__main__":
    get_results_svm("features/pifia", "conv8", "results/pifia")
    get_results_svm("features/pci", "conv4_1", "results/pci")
    get_results_svm("features/deeploc", "conv5", "results/deeploc")
