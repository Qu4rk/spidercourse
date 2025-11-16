import numpy as np
from pathlib import Path
from NN import Network
import matplotlib.pyplot as plt


def load_individual_from_txt(path: Path) -> np.ndarray:
    """Load a single best-individual text file produced by GA.save_population_txt.
    (Copied from show_best.py)
    """
    try:
        arr = np.loadtxt(path, delimiter=',')
    except Exception as e:
        raise RuntimeError(f"Failed to load '{path}': {e}")
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    return arr


def main():
    # 1. Load the GA data
    ga_output_file = Path("best_individual.txt")
    if not ga_output_file.exists():
        print("best_individual.txt not found. Run GA.py (Phase 1) first.")
        return

    # ga_data is shape (num_frames, 24)
    ga_data = load_individual_from_txt(ga_output_file)
    num_frames, num_joints = ga_data.shape

    print(f"Loaded GA data with shape {ga_data.shape}")

    # 2. Create Training Data
    # Input X: Time, normalized from 0 to 1
    # We add a new axis to make it shape (num_frames, 1)
    X_train = np.linspace(0, 1, num_frames)[:, np.newaxis]

    # Output Y: The 24 joint angles
    Y_train = ga_data

    # 3. Configure and Train the NN
    # Input layer: 1 (for time)
    # Hidden layers: e.g., 64 nodes, 64 nodes
    # Output layer: 24 (for the joint angles)
    layers = [1, 64, 64, num_joints]

    # Activations: 'relu' for hidden, 'linear' for output
    activations = ['relu', 'relu', 'linear']

    network = Network(seed=42, layers=layers, activations=activations)

    # Train the network
    lr = 0.001
    epochs = 1000
    losses = network.train(X_train, Y_train, lr, epochs)

    # 4. Visualize Training Loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title("NN Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.show()

    # You can call network.forward([0.5]) to predict the pose at the halfway point.


if __name__ == "__main__":
    main()