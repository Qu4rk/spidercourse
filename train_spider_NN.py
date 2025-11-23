import numpy as np
from pathlib import Path
import pathlib
import sys
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
from NN import Network
from activations import *
import numpy as np
import matplotlib.pyplot as plt

def load_individual_from_txt(path: Path) -> np.ndarray:
    """Load a single best-individual CSV produced by the GA.

    Accepts files with or without a leading frame column.
    Expected joint format per row: [frame, a1, a2, ...] or [a1, a2, ...].
    Returns array shaped (num_frames, num_joints) with only joint angles.
    """
    try:
        arr = np.loadtxt(path, delimiter=',')
    except Exception:
        arr = np.genfromtxt(path, delimiter=',')
    if arr.ndim == 1:
        # single row: make it (1, n)
        arr = arr.reshape(1, -1)

    # If there's a leading frame column, drop it.
    # Heuristic: if there are more columns than expected (>=25) and first column looks integer-like
    if arr.shape[1] > 24:
        first_col = arr[:, 0]
        # check if first column is integer-like (frame indices)
        if np.allclose(first_col, np.round(first_col), atol=1e-8):
            arr = arr[:, 1:]

    return arr.astype(np.float64)


def main():
    # 1. Load the GA data
    # default location: project_root/NN_Logic/best_individual.txt
    ga_output_file = Path("best_individual.txt")
    if not ga_output_file.exists():
        raise FileNotFoundError(f"{ga_output_file} not found")

    # ga_data is shape (num_frames, 24) or may include leading frame column
    ga_data = load_individual_from_txt(ga_output_file)
    num_frames, num_joints = ga_data.shape

    # If ga_data still contains a leading frame index (25 columns), drop it.
    if num_joints == 25:
        ga_data = ga_data[:, 1:]
        num_frames, num_joints = ga_data.shape

    # Final check: we must have 24 joint columns so train_data becomes 25 columns (time + 24)
    assert num_joints == 24, f"expected 24 joint columns (got {num_joints}); check {ga_output_file}"

    print(f"Loaded GA data with shape {ga_data.shape}")

    # 2. Create Training Data
    X_train = np.linspace(0, 1, num_frames)[:, np.newaxis]

    # Concatenate time (input) and GA joint angles (targets) into one array:
    train_data = np.hstack([X_train, ga_data])  # shape (num_frames, 1 + num_joints)
    assert train_data.shape[1] == 25, f"train_data must have 25 columns (time + 24 joints); got {train_data.shape[1]}"

    # 3. Build and train network
    layers = [1, 64, 64, 24]
    activations = [ReLU, ReLU, Linear]  # last layer Linear for regression

    network = Network(seed=42, layers=layers, activations=activations)

    lr = 0.001
    epochs = 1000
    losses = network.train(train_data, lr, epochs)

    # 4. Visualize Training Loss
    plt.figure(figsize=(8, 4))
    plt.plot(losses)
    plt.title("NN Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Avg MSE")
    plt.tight_layout()
    plt.show()

    # --- ADDED: predict all frames and save predictions (no frame label) ---
    X_all = np.linspace(0, 1, num_frames)[:, np.newaxis]
    preds = network.predict(X_all)  # shape (num_frames, 24)

    out_path = Path("predicted_individual.txt")
    np.savetxt(out_path, preds, delimiter=',', fmt='%.6f')
    print(f"Saved predictions to {out_path} (shape {preds.shape})")


if __name__ == "__main__":
    main()