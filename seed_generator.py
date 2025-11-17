import numpy as np

def generate_tripod_gait_seed(num_frames=100):
    """
    Generates a proper tripod gait pattern seed for the spider.
    This creates a smooth, coordinated walking motion that the GA can refine.
    
    Tripod groups:
    - Group A: R1, R3, L2, L4 (indices 7, 5, 1, 3 in foot array)
    - Group B: R2, R4, L1, L3 (indices 6, 4, 0, 2 in foot array)
    
    Leg order in angles: [L1, L2, L3, L4, R4, R3, R2, R1] × 3 joints each
    """
    
    # Initialize array: (frames, 24 joints)
    gait = np.zeros((num_frames, 24))
    
    # Define leg groups (index in the 24-element array)
    # Each leg has 3 joints: coxa (0), femur (1), tibia (2)
    GROUP_A = [21, 22, 23,  # R1
               15, 16, 17,  # R3
               3, 4, 5,     # L2
               9, 10, 11]   # L4
    
    GROUP_B = [18, 19, 20,  # R2
               12, 13, 14,  # R4
               0, 1, 2,     # L1
               6, 7, 8]     # L3
    
    # Phase offset: Group B is 180° out of phase with Group A
    phase_shift = np.pi
    
    for frame in range(num_frames):
        # Progress through the gait cycle (0 to 2π)
        t = (frame / num_frames) * 2 * np.pi
        
        # Process each leg
        for leg_idx in range(8):
            base_idx = leg_idx * 3
            coxa_idx = base_idx + 0
            femur_idx = base_idx + 1
            tibia_idx = base_idx + 2
            
            # Determine if this leg is in Group A or Group B
            is_group_a = coxa_idx in GROUP_A
            
            # Phase for this leg
            phase = t if is_group_a else (t + phase_shift)
            
            # Normalize phase to [0, 2π]
            phase = phase % (2 * np.pi)
            
            # Determine if leg is in stance (on ground) or swing (in air)
            # Stance: 60% of cycle, Swing: 40% of cycle
            stance_duration = 0.6 * 2 * np.pi
            is_stance = phase < stance_duration
            
            # === COXA (left-right swing) ===
            # Oscillates between -0.3 and +0.3 radians
            # Left legs: positive values, Right legs: negative values
            is_left = leg_idx < 4
            coxa_amplitude = 0.25
            coxa_base = coxa_amplitude if is_left else -coxa_amplitude
            
            if is_stance:
                # During stance: sweep backward (propulsion)
                stance_progress = phase / stance_duration
                coxa_angle = coxa_base * (1 - 2 * stance_progress)  # +max to -max
            else:
                # During swing: return forward
                swing_progress = (phase - stance_duration) / (2 * np.pi - stance_duration)
                coxa_angle = coxa_base * (-1 + 2 * swing_progress)  # -max to +max
            
            gait[frame, coxa_idx] = coxa_angle
            
            # === FEMUR (up-down motion) ===
            # Stance: -0.8 (leg extended down)
            # Swing: -0.5 (leg lifted slightly)
            if is_stance:
                femur_angle = -0.9  # Extended for stance
            else:
                # Smooth transition during swing
                swing_progress = (phase - stance_duration) / (2 * np.pi - stance_duration)
                # Lift and lower smoothly
                lift = 0.3 * np.sin(swing_progress * np.pi)
                femur_angle = -0.9 + lift
            
            gait[frame, femur_idx] = femur_angle
            
            # === TIBIA (coordinated with femur) ===
            # Must be more negative than femur to point foot down
            # Stance: -1.0 (leg bent for support)
            # Swing: -0.6 (slightly less bent during swing)
            if is_stance:
                tibia_angle = -1.1  # More bent during stance
            else:
                swing_progress = (phase - stance_duration) / (2 * np.pi - stance_duration)
                # Synchronize with femur lift
                lift = 0.4 * np.sin(swing_progress * np.pi)
                tibia_angle = -1.1 + lift
            
            gait[frame, tibia_idx] = tibia_angle
    
    return gait


def main():
    print("Generating smart tripod gait seed...")
    
    # Generate seeds for all three modes
    seed_100 = generate_tripod_gait_seed(100)
    seed_200 = generate_tripod_gait_seed(200)
    seed_300 = generate_tripod_gait_seed(300)
    
    # Save the seeds
    np.save("smart_seed_gait.npy", seed_300)  # Default for mode 1
    np.save("smart_seed_gait_200.npy", seed_200)  # For mode 2
    np.save("smart_seed_gait_100.npy", seed_100)  # For mode 3
    
    print("✓ Created smart_seed_gait.npy (300 frames)")
    print("✓ Created smart_seed_gait_200.npy (200 frames)")
    print("✓ Created smart_seed_gait_100.npy (100 frames)")
    print("\nThese seeds contain proper tripod gait patterns.")
    print("The GA will start with one good walker and evolve from there.")
    print("\nRun main.py now to train with the smart seed!")
    
    # Print sample values
    print("\n--- Sample Joint Angles (Frame 0) ---")
    print(f"L1 (coxa, femur, tibia): {seed_300[0, 0:3]}")
    print(f"R1 (coxa, femur, tibia): {seed_300[0, 21:24]}")


if __name__ == "__main__":
    main()
