# Overview
-- overview here
# Features
-- features here
# Installation
### 1. Clone the repository
```
git clone https://github.com/username/project-name.git
cd project-name
```
### 2. Create and activate a virtual environment
```
python -m venv venv
source venv/bin/activate     # On macOS/Linux
venv\Scripts\activate        # On Windows
```

### 3. install dependencies
```
pip install numpy matplotlib alive-progress
```

### 4. (Optional) Install scipy for smoother animations
For ultra-smooth spider gait visualization with cubic spline interpolation:
```
pip install scipy
```
Note: If scipy is not installed, the visualization will use eased linear interpolation, which is still much smoother than the original.

🧩 Usage

## Visualizing Spider Gaits

To visualize a saved gait with smooth animation:

```bash
python show_best.py
```

This will automatically find and display the latest `best_individual*.txt` file with smooth interpolation.

### Visualization Parameters

The `visualiseWalk()` function now supports smooth animation with the following parameters:

- **interpolation_steps** (default: 5): Number of sub-frames to interpolate between each keyframe. Higher values = smoother but slower animation.
- **frame_delay** (default: 0.03): Delay in seconds between frames (~33 FPS). Lower values = faster animation.
- **use_spline** (default: True): If True and scipy is installed, uses cubic spline interpolation for ultra-smooth motion. If False, uses eased linear interpolation.

Example usage in code:
```python
from plot_spider_pose import visualiseWalk

# Default smooth animation (cubic splines if scipy available)
visualiseWalk(frames)

# Custom parameters for even smoother animation
visualiseWalk(frames, interpolation_steps=10, frame_delay=0.02)

# Use eased linear interpolation instead of splines
visualiseWalk(frames, use_spline=False)
```

# Fitness Function
The fitness function determines a score for each individual in a population. To inspire ourselves from nature, we decided to use a fitness function that determines the "reward" factor based on a set of rules and attempt to maximise it. Invalid movements are given negative reward. In addition a scaling factor is applied to give various levels of importance to each rule.

Given the fact that the spider's body is static, we must simulate a treadmill in which all legs on the ground move backwards.

# Rules
1. Angle Limits
- The angle limit rule limits the angles from 360deg to actually plausible angles.
- Enforced during initialization and mutation
- This rule is not within the fitness function
2. Stability
Evaluated for each simulation frame:
Condition	Reward Adjustment
Group A and B both on the ground	reward += 0
Group A on ground, Group B in air	reward += 0
Any other combination	reward += -1
Encourages stable alternating contact between leg groups.
3. Counteraction
Evaluates motion smoothness and coordination between consecutive frames.
For each frame:
MSE = (frameN - frameN+1)^2
reward = 0 - MSE
Low Mean Squared Error (MSE) indicates coordinated movement.
Perfectly smooth motion (MSE = 0) yields a reward of +1.
Otherwise, the reward is negative (punishment).
4. Moving
Evaluates effective locomotion by measuring displacement:
reward = difference_between(first_zero, second_zero)
Encourages forward movement and penalizes static or regressive motion.


