





## Fitness Function
The fitness function is a measure of the reward which can be measured like so:
reward = (scaleN * ruleN) + ....
where scale is the importance of the rule and rule is the total reward (negative or positive) of the rule
The reward for each rule may be calculate

Ground level is -1

Group A is R1, R3, L2, L4
Group B is R2, R4, L1, L3

leg not move forward when its on the ground

## starting angles:
    left_leg_angles = np.deg2rad([45, 75, 105, 135])
    right_leg_angles = np.deg2rad([-135, -105, -75, -45])

## Rules
1. Angle limit
    - enforced at initialisation
    - enforced at mutation # needs to be done
2. Stability
    For each frame the following is checked:
    - if Group A and B are on the ground:
    reward += 0
    - if Group A is on the ground and group B is in the air:
    reward += 0
    - if any other combination:
    reward += -1
3. Counteraction
    For each consecutive frames the following is checked:
    X movement on each group
    if the MSE is low:
    MSE = (frameN - frameN+1)^2
    reward = 0 - MSE (Only punishment)
    if 0 - MSE = 0 reward += 1
4. Moving
    calculate the length between first 0 and second 0
    reward = difference