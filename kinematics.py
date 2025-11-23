#translated using https://www.codeconvert.ai
import math
import numpy as np

def axis_angle_rotation_matrix(axis, angle):
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    C = 1 - c
    R = np.array([
        [x*x*C + c,   x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, y*y*C + c,   y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, z*z*C + c]
    ])
    return R

def rotate_vector(v, axis, angle):
    R = axis_angle_rotation_matrix(axis, angle)
    v_rot = R.dot(v)
    return v_rot

def forward_leg_kinematics(base_pos, base_angle, joint_angles, segment_lengths):
    # Unpack joint angles
    theta1 = joint_angles[0]  # Coxa yaw (rotation about vertical axis)
    theta2 = joint_angles[1]  # Femur pitch
    theta3 = joint_angles[2]  # Tibia pitch

    # Unpack segment lengths
    L1 = segment_lengths[0]  # Coxa length
    L2 = segment_lengths[1]  # Femur length
    L3 = segment_lengths[2]  # Tibia length

    # Joint 1: leg base on body
    j1 = np.array(base_pos)  # starting point

    # --- Compute Coxa direction with elevation ---
    coxa_elevation = math.radians(30)  # fixed 30 degree upward pitch for coxa

    # Horizontal direction of coxa in XY plane based on base_angle + theta1
    coxa_horiz_dir = np.array([math.cos(base_angle + theta1), math.sin(base_angle + theta1), 0])

    # Rotation axis for pitch up: perpendicular to coxa horizontal direction in XY plane
    rot_axis = np.cross(coxa_horiz_dir, [0, 0, 1])

    # Rotation matrix around rot_axis by coxa_elevation
    R = axis_angle_rotation_matrix(rot_axis, coxa_elevation)

    # Rotate horizontal coxa direction upward
    coxa_dir = R.dot(coxa_horiz_dir)

    # Joint 2 position: end of coxa segment
    j2 = j1 + L1 * coxa_dir

    # --- Femur rotation ---
    # Femur pitch is relative to coxa direction, rotate in plane defined by coxa_dir
    # To simplify, rotate femur around axis perpendicular to coxa_dir and Z

    # Define femur rotation axis (perpendicular to coxa_dir and vertical axis)
    femur_rot_axis = np.cross(coxa_dir, [0, 0, 1])
    femur_rot_axis = femur_rot_axis / np.linalg.norm(femur_rot_axis)

    # Femur direction vector starts aligned with coxa_dir
    femur_dir = rotate_vector(coxa_dir, femur_rot_axis, theta2)

    # Joint 3 position: end of femur segment
    j3 = j2 + L2 * femur_dir

    # --- Tibia rotation ---
    # Tibia pitch is relative to femur direction
    # Rotate tibia around axis perpendicular to femur_dir and vertical axis

    tibia_rot_axis = np.cross(femur_dir, [0, 0, 1])
    tibia_rot_axis = tibia_rot_axis / np.linalg.norm(tibia_rot_axis)

    # Tibia direction vector
    tibia_dir = rotate_vector(femur_dir, tibia_rot_axis, theta3)

    # Joint 4 position: end of tibia segment (foot)
    j4 = j3 + L3 * tibia_dir

    return j1, j2, j3, j4

def getPositions(angles):
    # Parameters
    foot_positions = []
    n_legs = 8
    segment_lengths = [1.2, 0.7, 1.0]  # [Coxa, Femur, Tibia]
    a, b = 1.5, 1.0  # Ellipse axes for body (oval shape)

    # Base angles (L1 front-left to L4 rear-left, R4 rear-right to R1 front-right)
    left_leg_angles = np.deg2rad([45, 75, 105, 135])
    right_leg_angles = np.deg2rad([-135, -105, -75, -45])
    base_angles = np.concatenate((left_leg_angles, right_leg_angles))
    
    # Validate input
    if len(angles) != n_legs * 3:
        raise ValueError('Input angles must be a 1x24 vector (3 angles per leg for 8 legs).')

    # Loop over legs
    for i in range(n_legs):
        # Indices for this leg's angles
        idx = i * 3
        theta1 = angles[idx]
        theta2 = angles[idx + 1]
        theta3 = angles[idx + 2]

        # print(f'Leg {leg_labels[i]}: theta1 = {theta1:.3f} rad, theta2 = {theta2:.3f} rad, theta3 = {theta3:.3f} rad')

        # Compute leg base position on body ellipse
        angle = base_angles[i]
        x_base = a * np.cos(angle)
        y_base = b * np.sin(angle)
        base_pos = np.array([x_base, y_base, 0])

        # Compute FK for this leg (now uses theta1 correctly)
        j1, j2, j3, j4 = forward_leg_kinematics(base_pos, angle, [theta1, theta2, theta3], segment_lengths)
        foot_positions.append([j1, j2, j3, j4])
    
    return foot_positions