import numpy as np
from GA import GeneticAlgorithm
from plot_spider_pose import plot_spider_pose

def main():
    print("Select mode:")
    print(" [1] Maximum Performance  (1000 pop, 100 gens, 300 frames)")
    print(" [2] Balanced Performance (500 pop, 50 gens, 200 frames)")
    print(" [3] Testing / Debugging  (200 pop, 10 gens, 100 frames)")
    print("===============================")

    # Get user input
    choice = input("Enter mode number (1 / 2 / 3): ").strip()

    # Select GA settings based on mode
    if choice == "1":
        print("\nMode: MAXIMUM PERFORMANCE\n")
        ga = GeneticAlgorithm(
            populationSize=1000,
            individualLength=24,
            generations=100,
            mutationRate=0.02,
            frames=300,
            seed=42
        )
    elif choice == "2":
        print("\nMode: BALANCED PERFORMANCE\n")
        ga = GeneticAlgorithm(
            populationSize=500,
            individualLength=24,
            generations=25,
            mutationRate=0.02,
            frames=200,
            seed=42
        )
    elif choice == "3":
        print("\nMode: TESTING / DEBUGGING\n")
        ga = GeneticAlgorithm(
            populationSize=200,
            individualLength=24,
            generations=10,
            mutationRate=0.10,
            frames=100,
            seed=42
        )
    else:
        print("\n Invalid selection. Defaulting to [2] Balanced Mode.\n")
        ga = GeneticAlgorithm(
            populationSize=500,
            individualLength=24,
            generations=50,
            mutationRate=0.20,
            frames=200,
            seed=42
        )


    ga.start()

    # Set random joint angles for all 8 legs (3 joints per leg), not symmetrical
    # Each joint gets its own random value
    # angles = np.empty(24)
    # angles[:] = np.random.uniform(
    #     [-np.pi/4, -np.pi/2, -np.pi/2] * 8,   # lower bounds for each joint
    #     [ np.pi/4,  np.pi/2,  np.pi/2] * 8,   # upper bounds for each joint
    # )

    # print(angles)


    # plot_spider_pose(angles)

if __name__ == "__main__":
    main()