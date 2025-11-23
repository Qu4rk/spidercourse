import numpy as np

ANGLE_MIN = -5.0
ANGLE_MAX = 5.0

TOURNAMENT_SIZE = 7 

# ==========================================
# 2. FITNESS FUNCTION WEIGHTS
# ==========================================
W_CONSTRAINT = 10.0

# ==========================================
# 3. ROBOT PHYSICAL CONSTANTS (Offsets)
# ==========================================
# The center-point (C) for the wave equations
COXA_MIDPOINT  = 0.0      
FEMUR_MIDPOINT = -1.04    
TIBIA_MIDPOINT = -0.785   

# ==========================================
# 4. GAIT PHYSICS PARAMETERS (Wave Equation)
# ==========================================
# Form: y(t) = Offset + Amplitude * sin(Frequency * t + Phase)

# --- Frequency ---
# One full cycle per normalized time unit [0, 1]
ANGULAR_FREQUENCY = 2 * np.pi

# --- Amplitudes (A) ---
# Explicit values pre-calculated from mechanical limits.
COXA_AMPLITUDE  = 0.3925  # ~22.5 deg (Horizontal Stride)
FEMUR_AMPLITUDE = 0.2600  # ~15 deg   (Vertical Lift)
TIBIA_AMPLITUDE = 0.19625 # ~11 deg   (Vertical Lift)

# --- Phase Shifts (phi) ---
# Defined in Radians.

# Tripod Offset: 180 degrees (pi) difference between leg groups
PHASE_TRIPOD = np.pi 

# Ellipse Lag: 90 degrees (pi/2) delay. 
# Lags lift behind swing to create an elliptical step.
PHASE_ELLIPSE_LAG = -np.pi / 2

# Lift Timing: 180 degrees (pi) delay.
# Ensures lift occurs during the swing half, not the stance half.
PHASE_LIFT_TIMING = -np.pi