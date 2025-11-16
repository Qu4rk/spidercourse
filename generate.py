from NN import Network
import numpy as np
import numpy.typing as npt
import activations as atv

# --- Copied from your gait_generator.py ---

def generate_oscillating_gait(
    frames: int, 
    angles: int, 
    base_angle: float, 
    amplitude: float
) -> npt.NDArray[np.float64]:
    """
    Generates a 2D NumPy array (frames, angles) representing a smooth,
    oscillating gait using vectorized operations.
    """
    time_steps = np.linspace(0, 2 * np.pi, num=frames)
    angle_phases = np.linspace(0, 2 * np.pi, num=angles, endpoint=False)
    phase_grid = time_steps[:, None] + angle_phases[None, :]
    oscillation = np.sin(phase_grid) * amplitude
    gait = base_angle + oscillation
    return gait

# --- New main function inspired by your testNN.py ---

def main():
    # 1. Generate the raw gait data
    gait_data = generate_oscillating_gait(
        frames=300, 
        angles=24, 
        base_angle=0.0, 
        amplitude=0.8
    )
    
    # 2. Create the dataset: [Angle 1, Angle 0]
    #    Input (X) will be Angle 0
    #    Label (y) will be Angle 1
    #    This is a (300, 2) array, just like the house price data
    data = np.stack([gait_data[:, 1], gait_data[:, 0]], axis=1)

    # 3. Normalize the data (good practice, though it's already small)
    data_mean = np.mean(data, axis=0, keepdims=True)
    data_std = np.std(data, axis=0, keepdims=True)
    data_std[data_std == 0] = 1 
    normalized_data = (data - data_mean) / data_std
    
    # ---------------------------------------------------

    # 4. Define network
    #    Input: 1 feature (Angle 0)
    #    Hidden: 10 neurons (to learn the sine wave relationship)
    #    Output: 1 feature (Angle 1)
    layers = [1, 10, 1]
    
    # Use ReLU for the hidden layer, Linear for the regression output
    activations = [atv.ReLU, atv.Linear] 

    network = Network(42, layers, activations)
    
    # 4. Train on the NORMALIZED data
    #    A small LR is good for this.
    network.train(normalized_data, 0.001, 200, 0.2) # 200 epochs


if __name__ == '__main__':
    main()