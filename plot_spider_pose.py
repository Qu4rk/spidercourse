import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from forward_leg_kinematics import forward_leg_kinematics

# Try to import scipy for advanced interpolation
try:
    from scipy.interpolate import CubicSpline
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Note: scipy not available. Using eased linear interpolation instead of cubic splines.")

def plot_spider_pose(ax, angles):
    # plot_spider_pose - Plot a static 3D spider pose based on joint angles
    #
    # Input:
    #   angles: 1x24 vector of joint angles in radians
    #           [theta1_1, theta2_1, theta3_1, ..., theta1_8, theta2_8, theta3_8]
    # Legs are arranged in this configuration: ['L1', 'L2', 'L3', 'L4','R4', 'R3', 'R2', 'R1']

    # Parameters
    n_legs = 8
    segment_lengths = [1.2, 0.7, 1.0]  # [Coxa, Femur, Tibia]
    a, b = 1.5, 1.0  # Ellipse axes for body (oval shape)

    # Base angles (L1 front-left to L4 rear-left, R4 rear-right to R1 front-right)
    left_leg_angles = np.deg2rad([45, 75, 105, 135])
    right_leg_angles = np.deg2rad([-135, -105, -75, -45])
    base_angles = np.concatenate([left_leg_angles, right_leg_angles])

    # Leg labels
    leg_labels = ['L1', 'L2', 'L3', 'L4', 'R4', 'R3', 'R2', 'R1']

    # Validate input
    # if len(angles) != n_legs * 3:
    #     raise ValueError('Input angles must be a 1x24 vector (3 angles per leg for 8 legs).')

    # Setup figure
    # fig = plt.figure(1)
    # fig.clf()
    # fig.patch.set_facecolor('w')
    # ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1,1,0.5])
    ax.grid(True)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(45, 45)
    ax.set_xlim([-4, 4])
    ax.set_ylim([-4, 4])
    ax.set_zlim([-2, 2])

    # Plot body (oval shape)
    t = np.linspace(0, 2*np.pi, 100)
    body_x = a * np.cos(t)
    body_y = b * np.sin(t)
    body_z = np.zeros_like(t)
    ax.plot(body_x, body_y, body_z, 'k-', linewidth=3)

    # Head marker (front of spider at +X)
    ax.plot([a + 0.2], [0], [0], 'r^', markersize=10, markerfacecolor='r')

    # Print joint angles for all legs
    # print('--- Spider Pose ---')

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

        # Compute FK for this leg
        j1, j2, j3, j4 = forward_leg_kinematics(base_pos, angle, [theta1, theta2, theta3], segment_lengths)

        # Plot leg segments
        ax.plot([j1[0], j2[0]], [j1[1], j2[1]], [j1[2], j2[2]], 'k-', linewidth=2)
        ax.plot([j2[0], j3[0]], [j2[1], j3[1]], [j2[2], j3[2]], 'b-', linewidth=2)
        ax.plot([j3[0], j4[0]], [j3[1], j4[1]], [j3[2], j4[2]], 'r-', linewidth=2)
        ax.plot([j4[0]], [j4[1]], [j4[2]], 'ro', markersize=5, markerfacecolor='r')

        # Label leg
        offset = 0.2
        label_pos = base_pos + offset * np.array([np.cos(angle), np.sin(angle), 0])
        ax.text(label_pos[0], label_pos[1], label_pos[2] + 0.05, leg_labels[i],
                fontsize=12, fontweight='bold')

    # plt.show()

# def update(frame):
#     # Update the line data for the current frame
#     line.set_data(xs[:frame], ys[:frame])
#     line.set_3d_properties(zs[:frame])
#     return line,

# def visualiseWalk(frames):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection="3d")
#     ani = FuncAnimation(fig, update, frames=frames, interval=100)
#     plt.show()

def visualiseWalk(frames, interpolation_steps=5, frame_delay=0.03, use_spline=True):
    """
    Visualize spider walk with smooth interpolation between frames.

    Args:
        frames: (n_frames, 24) array of joint angles
        interpolation_steps: number of sub-frames to interpolate between each keyframe
        frame_delay: delay in seconds between displayed frames (default 0.03s = ~33 FPS)
        use_spline: if True, use cubic spline interpolation (smoother); if False, use eased linear
    """
    fig = plt.figure(1)
    fig.clf()
    fig.patch.set_facecolor('w')
    ax = fig.add_subplot(111, projection='3d')

    n_keyframes = frames.shape[0]

    if use_spline and SCIPY_AVAILABLE and n_keyframes > 3:
        # Advanced: Use cubic spline interpolation for ultra-smooth motion
        # Time points for keyframes
        keyframe_times = np.arange(n_keyframes)

        # Create new time points for interpolated frames
        n_interpolated = (n_keyframes - 1) * interpolation_steps + 1
        interpolated_times = np.linspace(0, n_keyframes - 1, n_interpolated)

        # Interpolate each joint angle separately using cubic splines
        interpolated_frames = np.zeros((n_interpolated, 24))
        for joint_idx in range(24):
            # Extract the trajectory of this joint across all keyframes
            joint_trajectory = frames[:, joint_idx]

            # Create cubic spline interpolator
            cs = CubicSpline(keyframe_times, joint_trajectory, bc_type='natural')

            # Evaluate spline at interpolated time points
            interpolated_frames[:, joint_idx] = cs(interpolated_times)

    else:
        # Simple: Use eased linear interpolation
        interpolated_frames = []

        # For each pair of consecutive keyframes, create interpolation
        for i in range(n_keyframes - 1):
            # Get current and next keyframe
            frame1 = frames[i]
            frame2 = frames[i + 1]

            # Create sub-frames between keyframes using linear interpolation with easing
            for step in range(interpolation_steps):
                # Smooth easing function (ease-in-out cubic)
                t = step / interpolation_steps
                # Apply smoothstep function for more natural motion
                smooth_t = t * t * (3.0 - 2.0 * t)

                # Interpolate all 24 joint angles
                interpolated_frame = frame1 * (1 - smooth_t) + frame2 * smooth_t
                interpolated_frames.append(interpolated_frame)

        # Add the last keyframe
        interpolated_frames.append(frames[-1])

        # Convert to numpy array
        interpolated_frames = np.array(interpolated_frames)

    # Display all interpolated frames
    total_frames = len(interpolated_frames)
    for i, angles in enumerate(interpolated_frames):
        ax.cla()
        keyframe_idx = int(i / interpolation_steps)
        plt.title(f"Frame {i+1} of {total_frames} (Keyframe region {min(keyframe_idx+1, n_keyframes)}/{n_keyframes})")
        plot_spider_pose(ax, angles)
        plt.pause(frame_delay)

    plt.show()


#modified from plot_spider_pose
def getPositions(angles):
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

    for i in range(n_legs):
        idx = i * 3
        theta1 = angles[idx]
        theta2 = angles[idx + 1]
        theta3 = angles[idx + 2]

        angle = base_angles[i]
        x_base = a * np.cos(angle)
        y_base = b * np.sin(angle)
        base_pos = np.array([x_base, y_base, 0])

        # Compute FK for this leg (now uses theta1 correctly)
        j1, j2, j3, j4 = forward_leg_kinematics(base_pos, angle, [theta1, theta2, theta3], segment_lengths)

        return j1, j2, j3, j4