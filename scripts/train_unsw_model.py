import os
import json
import numpy as np
import pandas as pd

np.random.seed(42)

TRAIN_PATH = "data/UNSW_NB15_training-set.csv"
TEST_PATH = "data/UNSW_NB15_testing-set.csv"

# Six numerical features selected for the 6-input ZK circuit
FEATURES = [
    "sttl",
    "dttl",
    "ct_state_ttl",
    "ct_dst_src_ltm",
    "ct_srv_dst",
    "rate",
]

os.makedirs("model", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("results", exist_ok=True)

print("Loading UNSW-NB15 dataset...")
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print("Training set:", train_df.shape)
print("Testing set:", test_df.shape)

# -------------------------------------------------------
# Feature scaling
# We cap each feature using 99th percentile from training set
# then convert values to integers in range 0-100.
# -------------------------------------------------------

caps = {}
for f in FEATURES:
    cap = float(np.percentile(train_df[f].fillna(0).values, 99))
    if cap <= 0:
        cap = 1.0
    caps[f] = cap

def scale_features(df):
    X = df[FEATURES].fillna(0).astype(float).values
    cap_values = np.array([caps[f] for f in FEATURES], dtype=float)

    X[X < 0] = 0
    X = np.minimum(X, cap_values)
    X = np.rint((X / cap_values) * 100).astype(np.int64)

    return X

# -------------------------------------------------------
# Balanced training subset
# -------------------------------------------------------

normal_train = train_df[train_df["label"] == 0]
attack_train = train_df[train_df["label"] == 1]

n_each = min(20000, len(normal_train), len(attack_train))

normal_sample = normal_train.sample(n=n_each, random_state=42)
attack_sample = attack_train.sample(n=n_each, random_state=43)

train_sample = pd.concat([normal_sample, attack_sample]).sample(frac=1, random_state=44)

X_train = scale_features(train_sample)
y_train = train_sample["label"].astype(int).values

# Validation subset
normal_val = normal_train.drop(normal_sample.index, errors="ignore").sample(n=2000, random_state=45)
attack_val = attack_train.drop(attack_sample.index, errors="ignore").sample(n=2000, random_state=46)

val_sample = pd.concat([normal_val, attack_val]).sample(frac=1, random_state=47)

X_val = scale_features(val_sample)
y_val = val_sample["label"].astype(int).values

# -------------------------------------------------------
# ZK-compatible small neural model:
# 6 input -> 4 hidden square neurons -> 2 output scores
#
# hidden[j] = (W1[j] dot x + b1[j])^2
# scores = W2 dot hidden + b2
#
# For lightweight training, we search integer W1/b1 and train
# output layer W2/b2 using ridge regression.
# -------------------------------------------------------

def train_output_layer(X, y, W1, b1):
    H = (X @ W1.T + b1) ** 2
    Z = np.concatenate([H, np.ones((H.shape[0], 1))], axis=1).astype(float)

    # Target: label 0 -> normal class, label 1 -> attack class
    T = np.zeros((len(y), 2), dtype=float)
    T[np.arange(len(y)), y] = 100.0

    ridge = 1e-2
    A = Z.T @ Z + ridge * np.eye(Z.shape[1])
    B = Z.T @ T

    beta = np.linalg.solve(A, B)
    return beta

def evaluate_float_model(X, y, W1, b1, beta):
    H = (X @ W1.T + b1) ** 2
    Z = np.concatenate([H, np.ones((H.shape[0], 1))], axis=1).astype(float)
    scores = Z @ beta
    pred = np.argmax(scores, axis=1)
    acc = float(np.mean(pred == y))
    return acc

best = None
rng = np.random.default_rng(42)

print("Training/searching small ZK-compatible model...")

for trial in range(500):
    W1 = rng.integers(-3, 4, size=(4, 6), dtype=np.int64)
    b1 = rng.integers(-5, 6, size=(4,), dtype=np.int64)

    # avoid all-zero hidden neurons
    for j in range(4):
        if not np.any(W1[j]):
            W1[j, rng.integers(0, 6)] = 1

    beta = train_output_layer(X_train, y_train, W1, b1)
    val_acc = evaluate_float_model(X_val, y_val, W1, b1, beta)

    if best is None or val_acc > best["val_acc"]:
        best = {
            "val_acc": val_acc,
            "W1": W1,
            "b1": b1,
            "beta": beta,
            "trial": trial,
        }

print("Best validation accuracy:", round(best["val_acc"] * 100, 2), "%")
print("Best trial:", best["trial"])

W1 = best["W1"]
b1 = best["b1"]
beta = best["beta"]

# -------------------------------------------------------
# Quantize output layer into integer weights for Circom
# -------------------------------------------------------

best_quant = None

for scale in [1000, 10000, 100000, 1000000]:
    W2 = np.rint(beta[:4, :].T * scale).astype(np.int64)
    b2 = np.rint(beta[4, :] * scale).astype(np.int64)

    H_val = (X_val @ W1.T + b1) ** 2
    scores_val = H_val @ W2.T + b2
    pred_val = np.argmax(scores_val, axis=1)
    acc_val = float(np.mean(pred_val == y_val))

    if best_quant is None or acc_val > best_quant["acc"]:
        best_quant = {
            "scale": scale,
            "acc": acc_val,
            "W2": W2,
            "b2": b2,
        }

W2 = best_quant["W2"]
b2 = best_quant["b2"]

print("Best quantization scale:", best_quant["scale"])
print("Quantized validation accuracy:", round(best_quant["acc"] * 100, 2), "%")

# -------------------------------------------------------
# Select 50 Normal + 50 Attack rows from testing set
# -------------------------------------------------------

normal_test = test_df[test_df["label"] == 0].sample(n=50, random_state=100)
attack_test = test_df[test_df["label"] == 1].sample(n=50, random_state=101)

selected_test = pd.concat([normal_test, attack_test]).sample(frac=1, random_state=102)
X_selected = scale_features(selected_test)
y_selected = selected_test["label"].astype(int).values

# Shift output scores positive for presentation and input simplicity.
H_selected = (X_selected @ W1.T + b1) ** 2
scores_selected = H_selected @ W2.T + b2
min_score = int(scores_selected.min())

if min_score < 0:
    shift = abs(min_score) + 1000
    b2 = b2 + shift
else:
    shift = 0

scores_selected = H_selected @ W2.T + b2
pred_selected = np.argmax(scores_selected, axis=1)
selected_acc = float(np.mean(pred_selected == y_selected))

print("Selected 100-row accuracy:", round(selected_acc * 100, 2), "%")
print("Output score shift added:", shift)

# -------------------------------------------------------
# Save trained model
# -------------------------------------------------------

def to_list(x):
    return np.array(x).astype(object).tolist()

model = {
    "model_name": "UNSW_NB15_ZK_Compatible_Square_NN",
    "features": FEATURES,
    "feature_caps": caps,
    "architecture": "6 input -> 4 square hidden neurons -> 2 output scores",
    "W1": to_list(W1),
    "b1": to_list(b1),
    "W2": to_list(W2),
    "b2": to_list(b2),
    "training_note": "Small ZK-compatible neural model. Hidden layer uses square activation. Output layer trained using ridge regression after integer hidden-feature search.",
    "validation_accuracy_percent": round(best_quant["acc"] * 100, 2),
    "selected_100_accuracy_percent": round(selected_acc * 100, 2),
}

with open("model/trained_model.json", "w", encoding="utf-8") as f:
    json.dump(model, f, indent=2)

# -------------------------------------------------------
# Save selected 100 rows metadata
# -------------------------------------------------------

selected_records = []

for i, (_, row) in enumerate(selected_test.iterrows()):
    x = X_selected[i].astype(int).tolist()

    normal_score = int(scores_selected[i][0])
    attack_score = int(scores_selected[i][1])

    prediction = "Attack" if attack_score > normal_score else "Normal"
    dataset_label = "Attack" if int(row["label"]) == 1 else "Normal"

    selected_records.append({
        "sample_id": i,
        "original_test_row_index": int(row.name),
        "attack_cat": str(row["attack_cat"]),
        "dataset_label": dataset_label,
        "raw_features": {f: float(row[f]) for f in FEATURES},
        "x": x,
        "normalScore": str(normal_score),
        "attackScore": str(attack_score),
        "prediction": prediction,
        "correct": prediction == dataset_label,
    })

with open("data/selected_100_rows.json", "w", encoding="utf-8") as f:
    json.dump(selected_records, f, indent=2)

summary = {
    "training_set_rows": int(len(train_df)),
    "testing_set_rows": int(len(test_df)),
    "features_used": FEATURES,
    "selected_samples": 100,
    "normal_samples": 50,
    "attack_samples": 50,
    "validation_accuracy_percent": round(best_quant["acc"] * 100, 2),
    "selected_100_accuracy_percent": round(selected_acc * 100, 2),
}

with open("results/model_training_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

pd.DataFrame(selected_records).to_csv("results/selected_100_rows_summary.csv", index=False)

print("\nSaved files:")
print("model/trained_model.json")
print("data/selected_100_rows.json")
print("results/model_training_summary.json")
print("results/selected_100_rows_summary.csv")