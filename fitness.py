import numpy as np
from forward_leg_kinematics import getPositions  # Required for dynamic rules

######################################################################
#### Global Constants and Weights (####)
######################################################################

# --- Pose Quality Weights (cite: 60-64) ---
W_SYMMETRY = 0.12  # Balanced left-right mirroring for stability
W_COUNTERACTION_POSE = 0.10  # Same-side opposition for walking rhythm
W_MIMICRY = 0.08  # Same-side front/rear coordination
W_RANGE = 0.18  # Keep joints within safe anatomical range
W_SMOOTH = 0.12  # Encourage gentle, flowing leg curvature
W_GROUP_MIMICRY = 0.10  # Cohesion within each tripod group
W_GROUP_COUNTERACTION = 0.08  # Opposing group alternation (tripod gait)
W_GROUP_COORD = 0.08  # Synchronize both tripods' rhythm
W_SPATIAL = 0.06  # Prevent leg overlap and excessive crossing
W_ANATOMY = 0.08  # Correct joint hierarchy and natural bending

# Joint Limits for Range Comfort (Rule 4)
# Used to normalize angle penalty.
COXA_LIMIT = np.pi / 4  # ~0.785
FEMUR_LIMIT = np.pi / 3  # ~1.047
TIBIA_LIMIT = np.pi / 2  # ~1.570
LIMITS_PER_LEG = [COXA_LIMIT, FEMUR_LIMIT, TIBIA_LIMIT]
JOINT_LIMITS_VECTOR = np.tile(LIMITS_PER_LEG, 8)  # 24-element vector

# Leg Index Map for a 24-element vector [L1(3), L2(3), ..., R1(3)]
LEG_IDX = {
    'L1': 0, 'L2': 3, 'L3': 6, 'L4': 9,
    'R4': 12, 'R3': 15, 'R2': 18, 'R1': 21
}

# --- Dynamics Constants ---
GROUND_LEVEL = -1.0  # "Ground level is -1"

# Foot position vector order: [L1, L2, L3, L4, R4, R3, R2, R1]
GROUP_A_LABELS = ['R1', 'R3', 'L2', 'L4']
GROUP_B_LABELS = ['R2', 'R4', 'L1', 'L3']
# Map labels to their index in the 8-element foot position array:
LEG_LABELS_ORDER = ['L1', 'L2', 'L3', 'L4', 'R4', 'R3', 'R2', 'R1']
GROUP_A_INDICES = [LEG_LABELS_ORDER.index(label) for label in GROUP_A_LABELS]  # [7, 5, 1, 3]
GROUP_B_INDICES = [LEG_LABELS_ORDER.index(label) for label in GROUP_B_LABELS]  # [6, 4, 0, 2]

# --- Rule Weights (Scales) ---
SCALE_STABILITY = 1.0  # Rule 2
SCALE_COUNTERACTION = 1.5  # Rule 3
SCALE_MOVEMENT = 3.0  # Rule 4 & "no slip" rule

# --- Final Combination Weights (Adaptive) ---
W_POSE_QUALITY = 2.0  # default fallback if needed
W_GAIT_DYNAMICS = 2.0  # default fallback if GA not running adaptively


# ====================================================================
# Main Wrapper Function
# ====================================================================

def calculateReward(population):
    """
    Calculates the fitness for the entire population (list of individuals).
    """
    fitness_scores: list[float] = []
    for idx, individual_frames in enumerate(population):
        fitness = calculateIndividualReward(individual_frames)
        fitness_scores.append(fitness)

    fitness_array = np.array(fitness_scores, dtype=np.float64)
    print(f"Average Fitness: {np.mean(fitness_array):.3f}, Best Fitness: {np.max(fitness_array):.3f}")
    return fitness_array


def calculateIndividualReward(individual_frames):
    """
    CalculATES the total reward for a single individual (a sequence of frames).
    An individual is a numpy array of shape (num_frames, 24).
    """
    if individual_frames.ndim == 1:
        individual_frames = np.array([individual_frames])  # Wrap single pose in a frame array

    num_frames = individual_frames.shape[0]
    if num_frames < 2:
        avg_pose_score = _calculate_pose_quality_score(individual_frames[0])
        return W_POSE_QUALITY * avg_pose_score

    # 1. Get all foot positions for all frames (Needed for Gait Dynamics)
    try:
        positions_data = [getPositions(frame_angles) for frame_angles in individual_frames]
        foot_positions = np.array([[leg_joints[3] for leg_joints in frame] for frame in positions_data])
    except ValueError as e:
        print(f"Error in forward kinematics, individual is invalid: {e}")
        return -np.inf  # Return negative infinity for an invalid individual

    # 2. Calculate Pose Quality Score
    all_pose_scores = [_calculate_pose_quality_score(frame_angles) for frame_angles in individual_frames]
    avg_pose_score = np.mean(all_pose_scores)  # Normalized 0 to 1

    # 3. Calculate Gait Dynamics Reward
    total_dynamic_reward = _calculate_gait_dynamics_reward(foot_positions)  # Raw score (+/-)

    # 4. Final Weighted Combination
    # The global multiplier has been removed.
    final_fitness = (W_POSE_QUALITY * avg_pose_score) + (W_GAIT_DYNAMICS * total_dynamic_reward)

    return final_fitness


# ====================================================================
# Pose Quality (Applied per frame/pose)
# ====================================================================

def _calculate_pose_quality_score(angles_24):
    """
    Calculates the weighted fitness score for a single 24-element static pose. Returns a score from 0.0 to 1.0.
    """

    # --- Angle Extraction ---
    L1 = angles_24[LEG_IDX['L1']: LEG_IDX['L1'] + 3]
    L2 = angles_24[LEG_IDX['L2']: LEG_IDX['L2'] + 3]
    L3 = angles_24[LEG_IDX['L3']: LEG_IDX['L3'] + 3]
    L4 = angles_24[LEG_IDX['L4']: LEG_IDX['L4'] + 3]
    R1 = angles_24[LEG_IDX['R1']: LEG_IDX['R1'] + 3]
    R2 = angles_24[LEG_IDX['R2']: LEG_IDX['R2'] + 3]
    R3 = angles_24[LEG_IDX['R3']: LEG_IDX['R3'] + 3]
    R4 = angles_24[LEG_IDX['R4']: LEG_IDX['R4'] + 3]

    # 1. Range Comfort
    joint_weights = np.tile([0.5, 1.0, 0.8], 8)  # coxa less strict, femur stricter
    comfort_ratios = np.abs(angles_24) / JOINT_LIMITS_VECTOR
    midrange_penalty = np.clip(np.abs(comfort_ratios - 0.5), 0, 1)
    range_score = np.exp(-1.5 * np.average(midrange_penalty, weights=joint_weights))

    # 2. Smooth Leg Shape
    all_leg_smooth_errors = []
    for leg_angles in [L1, L2, L3, L4, R4, R3, R2, R1]:
        coxa, femur, tibia = leg_angles
        curve_variation = (femur - coxa) ** 2 + (tibia - femur) ** 2
        direction_change = abs((femur - coxa) * (tibia - femur))
        leg_err = np.sqrt(curve_variation + 0.5 * direction_change)
        all_leg_smooth_errors.append(leg_err)

    smooth_error = np.mean(all_leg_smooth_errors)
    global_smooth = np.std(all_leg_smooth_errors)
    smooth_score = np.exp(-1.2 * (smooth_error + 0.3 * global_smooth))

    # 3. Anatomical Consistency
    anatomical_errors = []
    for leg_angles in [L1, L2, L3, L4, R4, R3, R2, R1]:
        coxa, femur, tibia = leg_angles
        femur_upward_penalty = max(0, femur - (coxa - 0.10))
        tibia_inversion_penalty = max(0, (femur - tibia) - 0.6)
        curvature_penalty = abs((femur - coxa) - 0.6 * (tibia - femur))
        total_err = (1.5 * femur_upward_penalty) + (1.0 * tibia_inversion_penalty) + (0.5 * curvature_penalty)
        anatomical_errors.append(total_err)

    anatomical_error = np.mean(anatomical_errors)
    anatomical_score = np.exp(-3.0 * anatomical_error)

    # 4. Left-Right Symmetry
    pair_errors = []
    for L, R in zip([L1, L2, L3, L4], [R1, R2, R3, R4]):
        coxa_diff = abs(L[0] + R[0])  # Coxa should be mirrored (L = -R)
        femur_diff = abs(L[1] - R[1])  # Femur should bend similarly
        tibia_diff = abs(L[2] - R[2])  # Tibia should bend similarly
        pair_errors.append(coxa_diff + femur_diff + tibia_diff)

    sym_error = np.mean(pair_errors)
    sym_score = 1.0 / (1.0 + sym_error)

    # 5. Same-Side Counteraction (Opposite)
    pair_errors = []
    for (A, B) in [(L1, L2), (L3, L4), (R1, R2), (R3, R4)]:
        coxa_diff = abs(A[0] + B[0])  # opposite swing desired
        femur_diff = abs(A[1] - B[1])  # similar lift desired
        tibia_diff = abs(A[2] - B[2])  # similar bend desired
        pair_errors.append(coxa_diff + femur_diff + tibia_diff)

    opp_error = np.mean(pair_errors)
    opp_score = np.exp(-2.5 * opp_error)

    # 6. Same-Side Mimicry (Same)
    pair_errors = []
    for (A, B) in [(L1, L3), (L2, L4), (R1, R3), (R2, R4)]:
        coxa_diff = abs(A[0] - B[0])  # similar swing desired
        femur_diff = abs(A[1] - B[1])  # similar lift desired
        tibia_diff = abs(A[2] - B[2])  # similar bend desired
        pair_errors.append(coxa_diff + femur_diff + tibia_diff)

    mimic_error = np.mean(pair_errors)
    mimic_score = np.exp(-1.5 * mimic_error)

    # 7. Spatial Awareness
    left_sets = [L1, L2, L3, L4]
    right_sets = [R1, R2, R3, R4]
    left_avg = np.mean([[leg[0], leg[1], leg[2]] for leg in left_sets], axis=0)
    right_avg = np.mean([[leg[0], leg[1], leg[2]] for leg in right_sets], axis=0)

    cross_penalty = np.mean([
        max(0, abs(left_avg[0] + right_avg[0]) - 0.2),
        max(0, abs(left_avg[1] + right_avg[1]) - 0.3),
        max(0, abs(left_avg[2] + right_avg[2]) - 0.3)
    ])
    spread_penalty = np.mean([
        max(0, abs(left_avg[0] - right_avg[0]) - 1.0),
        max(0, abs(left_avg[1] - right_avg[1]) - 1.0),
        max(0, abs(left_avg[2] - right_avg[2]) - 1.0)
    ])

    spatial_error = 0.8 * cross_penalty + 0.5 * spread_penalty
    spatial_score = np.exp(-2.0 * spatial_error)

    # 8. Group-Level Static Coordination
    A_angles = [angles_24[i * 3:i * 3 + 3] for i in GROUP_A_INDICES]
    B_angles = [angles_24[i * 3:i * 3 + 3] for i in GROUP_B_INDICES]
    A_avg = np.mean(A_angles, axis=0)
    B_avg = np.mean(B_angles, axis=0)
    joint_weights = np.array([1.0, 0.8, 0.5])
    A_mimic_error = np.average(
        [np.sum(np.abs(a - A_avg)) for a in A_angles], weights=[1] * len(A_angles)
    )
    B_mimic_error = np.average(
        [np.sum(np.abs(b - B_avg)) for b in B_angles], weights=[1] * len(B_angles)
    )
    group_mimic_score = np.exp(-2.5 * np.mean([A_mimic_error, B_mimic_error]))
    A_motion = np.std(A_angles)
    B_motion = np.std(B_angles)
    activity_bonus = np.exp(-1.5 * abs(A_motion - B_motion))
    group_counter_error = np.average([
        abs(A_avg[0] + B_avg[0]) * 1.0,  # coxa: opposite motion
        abs(A_avg[1] - B_avg[1]) * 0.8,  # femur: similar bending
        abs(A_avg[2] - B_avg[2]) * 0.6  # tibia: similar extension
    ], weights=joint_weights)
    group_counter_score = np.exp(-2.5 * group_counter_error)
    group_coord_score = np.sqrt(group_mimic_score * group_counter_score * activity_bonus)

    # Final Weighted Score (0 to 1)
    pose_quality_score = (W_SYMMETRY * sym_score) + \
                         (W_COUNTERACTION_POSE * opp_score) + \
                         (W_MIMICRY * mimic_score) + \
                         (W_RANGE * range_score) + \
                         (W_SMOOTH * smooth_score) + \
                         (W_GROUP_MIMICRY * group_mimic_score) + \
                         (W_GROUP_COUNTERACTION * group_counter_score) + \
                         (W_GROUP_COORD * group_coord_score) + \
                         (W_SPATIAL * spatial_score) + \
                         (W_ANATOMY * anatomical_score)

    return pose_quality_score


# ====================================================================
# Gait Dynamics (Applied across frames/movement)
# ====================================================================

def _calculate_gait_dynamics_reward(foot_positions):
    """
    Calculates the total raw reward (not normalized).
    This function contains the critical fixes for the "jerky" walk.
    """
    total_reward = 0
    num_frames = foot_positions.shape[0]
    contact_threshold = GROUND_LEVEL + 0.05

    # --- 1. Tripod Stability Score ---
    # This rewards gaits that approximate a 4-leg-contact stability.
    tripod_score = 0
    for t in range(num_frames):
        z = foot_positions[t, :, 2]
        n_ground = np.sum(z <= contact_threshold)
        # Use an exponential reward, peaking at 4 legs on the ground
        tripod_score += np.exp(-((n_ground - 4) ** 2) / 8)
    tripod_score /= num_frames
    total_reward += SCALE_STABILITY * tripod_score * 2.0  # Extra weight

    # --- 2. Group Alternating Stability ---
    # This rewards a proper tripod gait (Group A up, Group B down, etc.)
    for t in range(num_frames):
        frame_feet_z = foot_positions[t, :, 2]
        A_contacts = np.sum(frame_feet_z[GROUP_A_INDICES] <= contact_threshold)
        B_contacts = np.sum(frame_feet_z[GROUP_B_INDICES] <= contact_threshold)
        A_stability = A_contacts / len(GROUP_A_INDICES)
        B_stability = B_contacts / len(GROUP_B_INDICES)

        if (A_stability > 0.5 and B_stability < 0.5) or (B_stability > 0.5 and A_stability < 0.5):
            stability_score = 1.0  # good alternating tripod
        elif (A_stability > 0.3 and B_stability > 0.3):
            stability_score = 0.5  # both partly touching (ok)
        else:
            stability_score = -0.8  # unstable: both off or wrong balance
        total_reward += SCALE_STABILITY * stability_score

    # --- 3. Group Counteraction (Opposing X-Movement) ---
    avg_X_A = foot_positions[:, GROUP_A_INDICES, 0].mean(axis=1)
    avg_X_B = foot_positions[:, GROUP_B_INDICES, 0].mean(axis=1)

    for t in range(num_frames - 1):
        delta_X_A = avg_X_A[t + 1] - avg_X_A[t]
        delta_X_B = avg_X_B[t + 1] - avg_X_B[t]
        mag_A = abs(delta_X_A)
        mag_B = abs(delta_X_B)

        if mag_A < 1e-4 and mag_B < 1e-4: continue

        if np.sign(delta_X_A) != np.sign(delta_X_B):  # Good: opposite directions
            ratio = 1.0 - abs(mag_A - mag_B) / (mag_A + mag_B + 1e-8)
            frame_reward = 1.0 * ratio
        else:  # Bad: same direction
            frame_reward = -0.5 * (mag_A + mag_B)
        total_reward += SCALE_COUNTERACTION * frame_reward

    # --- 4. Moving & Slip Penalty (FIXED) ---
    for t in range(num_frames - 1):
        pos_t = foot_positions[t]
        pos_t1 = foot_positions[t + 1]

        for i in range(foot_positions.shape[1]):  # iterate over 8 legs
            is_on_ground = (pos_t[i, 2] <= contact_threshold) and (pos_t1[i, 2] <= contact_threshold)

            if is_on_ground:
                delta_X = pos_t1[i, 0] - pos_t[i, 0]  # forward/back
                delta_Y = pos_t1[i, 1] - pos_t[i, 1]  # sideways

                # 1. Penalize forward slipping (delta_X > 0)
                if delta_X > 0:
                    total_reward -= SCALE_MOVEMENT * (delta_X ** 2) * 1.5  # Quadratic slip penalty

                # 2. Reward backward pushing (delta_X < 0)
                elif delta_X < 0:
                    # This incentivizes strong pushes, not "taps".
                    reward = (delta_X ** 2)  # (negative * negative = positive)
                    total_reward += SCALE_MOVEMENT * reward * 2.0

                # 3. Penalize sideways drift (delta_Y)
                total_reward -= SCALE_MOVEMENT * (delta_Y ** 2) * 1.0

                # 4. Penalize jerky ground contact
                if t > 0:
                    prev_delta_X = foot_positions[t, i, 0] - foot_positions[t - 1, i, 0]
                    smoothness_penalty = abs(delta_X - prev_delta_X)
                    total_reward -= 0.5 * smoothness_penalty

    # --- 5. Forward Progress Reward ---
    # This rewards the *net result* of all pushes.
    body_disp = foot_positions[-1, :, 0].mean() - foot_positions[0, :, 0].mean()
    # We want negative X movement (pushing "backward" on the treadmill)
    progress_reward = -body_disp
    total_reward += SCALE_MOVEMENT * progress_reward * 3.0

    # --- 6. Smooth Motion Reward ---
    # Penalize big per-frame velocity swings (jerky motion)
    delta_positions = np.diff(foot_positions[:, :, 0], axis=0)  # (frames-1, 8)
    frame_variation = np.mean(np.abs(np.diff(delta_positions, axis=0)))
    motion_smoothness = np.exp(-6 * frame_variation)
    total_reward += motion_smoothness

    return total_reward

