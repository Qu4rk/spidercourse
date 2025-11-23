import numpy as np
from config import *

def calculateFitness(population) -> np.ndarray[np.float64]:
    if population is None or len(population) == 0:
        return np.array([], dtype=np.float64)

    # ==========================================
    # 1. GENERATE WAVEFORMS (Vectorized)
    # ==========================================
    
    N_frames = population.shape[1]
    t = np.linspace(0, 1, N_frames, endpoint=False)
    
    # --- Equation 1: Horizontal Swing (Continuous Sine) ---
    # y = C + A * sin(wt + phi)
    
    target_coxa_A = COXA_MIDPOINT + COXA_AMPLITUDE * np.sin(ANGULAR_FREQUENCY * t)
    target_coxa_B = COXA_MIDPOINT + COXA_AMPLITUDE * np.sin(ANGULAR_FREQUENCY * t + PHASE_TRIPOD)

    # --- Equation 2: Vertical Lift (Rectified Sine) ---
    # y = C + A * max(0, sin(wt + phi))
    
    # Tripod A Phase: Needs both Ellipse Lag and Timing Offset
    phi_lift_A = PHASE_ELLIPSE_LAG + PHASE_LIFT_TIMING
    
    # Tripod B Phase: Needs only Ellipse Lag (it swings when A is planted)
    phi_lift_B = PHASE_ELLIPSE_LAG

    # Generate Raw Waves
    raw_wave_A = np.sin(ANGULAR_FREQUENCY * t + phi_lift_A)
    raw_wave_B = np.sin(ANGULAR_FREQUENCY * t + phi_lift_B)

    # Apply Rectifier (The Floor Constraint)
    # Negative values (Stance) -> 0.0
    # Positive values (Swing)  -> Curve
    lift_mask_A = np.maximum(0, raw_wave_A)
    lift_mask_B = np.maximum(0, raw_wave_B)

    # Calculate Final Vertical Angles
    target_femur_A = FEMUR_MIDPOINT + FEMUR_AMPLITUDE * lift_mask_A
    target_femur_B = FEMUR_MIDPOINT + FEMUR_AMPLITUDE * lift_mask_B
    
    target_tibia_A = TIBIA_MIDPOINT + TIBIA_AMPLITUDE * lift_mask_A
    target_tibia_B = TIBIA_MIDPOINT + TIBIA_AMPLITUDE * lift_mask_B

    # ==========================================
    # 2. CALCULATE FITNESS ERROR
    # ==========================================
    
    # Indices for Tripod Groups (Set A vs Set B)
    # Set A: Legs 2, 4, 6 (Indices 1, 3, 5) + 7 (Dummy/Padding if used)
    # Set B: Legs 1, 3, 5 (Indices 0, 2, 4) + 6
    IDX_A = np.array([1, 3, 5, 7]) 
    IDX_B = np.array([0, 2, 4, 6])

    # Extract population joints (Shape: Pop x Frames x Legs)
    pop_coxa  = population[:, :, 0::3]
    pop_femur = population[:, :, 1::3]
    pop_tibia = population[:, :, 2::3]

    # Create Target Matrices (Broadcast targets across all legs/individuals)
    tgt_coxa = np.zeros_like(pop_coxa)
    tgt_coxa[:, :, IDX_A] = target_coxa_A[None, :, None]
    tgt_coxa[:, :, IDX_B] = target_coxa_B[None, :, None]

    tgt_femur = np.zeros_like(pop_femur)
    tgt_femur[:, :, IDX_A] = target_femur_A[None, :, None]
    tgt_femur[:, :, IDX_B] = target_femur_B[None, :, None]

    tgt_tibia = np.zeros_like(pop_tibia)
    tgt_tibia[:, :, IDX_A] = target_tibia_A[None, :, None]
    tgt_tibia[:, :, IDX_B] = target_tibia_B[None, :, None]

    # Calculate Mean Absolute Error (MAE)
    err_coxa  = np.mean(np.abs(pop_coxa - tgt_coxa), axis=(1, 2))
    err_femur = np.mean(np.abs(pop_femur - tgt_femur), axis=(1, 2))
    err_tibia = np.mean(np.abs(pop_tibia - tgt_tibia), axis=(1, 2))

    # Final Combined Score
    error_total = (err_coxa + err_femur + err_tibia) / 3.0
    fitness_scores = 100.0 / (1.0 + (W_CONSTRAINT * error_total))

    # ==========================================
    # 3. LOGGING
    # ==========================================
    best_idx = np.argmax(fitness_scores)
    print(f"\nOptimization Check - Best: {fitness_scores[best_idx]:.4f}/100.0")
    print(f"  Coxa Error: {err_coxa[best_idx]:.4f}")
    print(f"  Femur Error: {err_femur[best_idx]:.4f}")
    print(f"  Tibia Error: {err_tibia[best_idx]:.4f}")
    print(f"--------------------------------")

    return fitness_scores