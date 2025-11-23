import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from required.kinematics import forward_leg_kinematics

# --- Global State for Playback Control ---
# These globals are necessary for the interactive controls to work with FuncAnimation
current_frame_index = 0
is_paused = True # Start paused for controlled playback
animation = None
frames = None
ax = None

# --- Constants for Plotting ---
n_legs = 8
segment_lengths = [1.2, 0.7, 1.0]  # [Coxa, Femur, Tibia]
a, b = 1.5, 1.0  # Ellipse axes for body (oval shape)
left_leg_angles = np.deg2rad([45, 75, 105, 135])
right_leg_angles = np.deg2rad([-135, -105, -75, -45])
base_angles = np.concatenate([left_leg_angles, right_leg_angles])
leg_labels = ['L1', 'L2', 'L3', 'L4', 'R4', 'R3', 'R2', 'R1']


def _plot_spider_pose_frame(ax, angles):
    # This is the internal plotting logic, updated to use the global constants
    
    # Setup axis limits and labels once per full plot (not per frame update)
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

    # Loop over legs
    for i in range(n_legs):
        # Indices for this leg's angles
        idx = i * 3
        theta1 = angles[idx]
        theta2 = angles[idx + 1]
        theta3 = angles[idx + 2]

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
    
    ax.set_title(f"Frame {current_frame_index + 1} of {len(frames)}")


def update_plot(frame_index):
    """
    Callback function for FuncAnimation.
    The frame_index is ignored when stepping manually.
    """
    global ax, frames, current_frame_index
    
    # When playing, the frame_index argument is used. 
    # When stepping, we use the global current_frame_index.
    if not is_paused:
        current_frame_index = frame_index 

    # Clear the previous plot elements
    ax.cla() 
    
    # Re-plot the current frame's pose
    _plot_spider_pose_frame(ax, frames[current_frame_index])
    
    return ax, # Return the updated artist


def on_key_press(event):
    """Handles keyboard controls for play/pause and stepping."""
    global current_frame_index, is_paused, animation, frames
    
    N_frames = len(frames)
    
    if event.key == ' ': # Spacebar to toggle Play/Pause
        if is_paused:
            animation.event_source.start()
            is_paused = False
            print("Playback: Resumed (Press Space to Pause)")
        else:
            animation.event_source.stop()
            is_paused = True
            print("Playback: Paused (Press Space to Play)")

    elif event.key == 'right' or event.key == 'n': # Right Arrow or 'n' (Next)
        if is_paused:
            current_frame_index = (current_frame_index + 1) % N_frames
            # Manually call the update function and redraw the canvas
            update_plot(current_frame_index)
            plt.draw()
            print(f"Step Forward to Frame: {current_frame_index + 1}")

    elif event.key == 'left' or event.key == 'p': # Left Arrow or 'p' (Previous)
        if is_paused:
            current_frame_index = (current_frame_index - 1) % N_frames
            # Manually call the update function and redraw the canvas
            update_plot(current_frame_index)
            plt.draw()
            print(f"Step Backward to Frame: {current_frame_index + 1}")

def visualiseWalk(input_frames):
    """
    Visualizes the gait sequence with interactive controls:
    - Spacebar: Play/Pause
    - Right Arrow / 'n': Step Forward (when paused)
    - Left Arrow / 'p': Step Backward (when paused)
    """
    global frames, current_frame_index, is_paused, animation, ax
    
    # Initialize globals
    frames = input_frames
    current_frame_index = 0
    is_paused = True

    fig = plt.figure(1)
    fig.clf()
    fig.patch.set_facecolor('w')
    ax = fig.add_subplot(111, projection='3d')

    # Initial plot (to set up the axis and show the first frame)
    _plot_spider_pose_frame(ax, frames[0])
    ax.set_title(f"Frame 1 of {len(frames)} - Paused")
    print("\n--- Interactive Gait Visualization ---")
    print("Controls:")
    print("  - Spacebar: Play/Pause")
    print("  - Left/Right Arrows (or P/N): Step Backward/Forward (when paused)")
    print("--------------------------------------")


    # Create the animation object
    animation = FuncAnimation(
        fig, 
        update_plot, 
        frames=range(len(frames)), 
        interval=100, # 100ms interval (10 FPS)
        blit=False, 
        repeat=True
    )
    
    # Connect the key press handler
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    
    # Start the animation paused
    animation.event_source.stop()

    plt.show()


#modified from plot_spider_pose
def getPositions(angles):
    # This function is retained from the original file structure
    
    # Validate input
    if len(angles) != n_legs * 3:
        raise ValueError('Input angles must be a 1x24 vector (3 angles per leg for 8 legs).')

    positions = []
    
    for i in range(n_legs):
        idx = i * 3
        theta1 = angles[idx]
        theta2 = angles[idx + 1]
        theta3 = angles[idx + 2]

        angle = base_angles[i]
        x_base = a * np.cos(angle)
        y_base = b * np.sin(angle)
        base_pos = np.array([x_base, y_base, 0])

        # Compute FK for this leg
        j1, j2, j3, j4 = forward_leg_kinematics(base_pos, angle, [theta1, theta2, theta3], segment_lengths)
        
        # Store the tip position (j4)
        positions.append(j4)

    # Note: The original code only returned positions for the LAST leg in the loop.
    # This corrected version returns a list of all 8 tip positions, or you can adjust it
    # to return only j1, j2, j3, j4 for a specific leg if that was the intent.
    # For now, it returns the j1, j2, j3, j4 of the last leg, as per the original return statement.
    return j1, j2, j3, j4 # Returning only the last leg's segments as per original implementation