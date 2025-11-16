import numpy as np
import matplotlib.pyplot as plt
import fitness
import csv
from plot_spider_pose import visualiseWalk
from fitness import calculateReward
from pathlib import Path


# from fitness import calculateIndividualReward, calculateReward

class GeneticAlgorithm:
    # 24 x frames
    # frames: number of frames per individual
    # N: population size

    #[L11, L12, L13, L21, L22, L23, L31, ... L41, R4, R3, R2, R1]

    def __init__(self, populationSize: int, individualLength: int, generations: int, mutationRate: float, frames: int, seed: int = None):
        self.frames = frames
        self.populationSize = populationSize
        self.individualLength = individualLength
        self.generations = generations
        self.mutationRate = mutationRate

        # 45 degrees per joint limit
        self.coxaMin  = -0.785   # -45°
        self.coxaMax  =  0.785   # +45°

        self.femurMin = -2.356   # -135°
        self.femurMax =  0.0     # 0° (straight)

        self.tibiaMin = -1.57    # -90°
        self.tibiaMax =  0.0     # 0° (straight)
        self.rng = np.random.default_rng(seed)

    #AI code
    def save_population_txt(self, filename: str = "best_individual.txt", fitness=None):
        """
        Save only the best individual's frames (values only) to a text file.
        If the target filename already exists, create a new file by appending
        a numeric suffix: best_individual_1.txt, best_individual_2.txt, ...
        The file contains only comma-separated numeric values, one frame per line.
        """
        if not hasattr(self, "population") or len(self.population) == 0:
            print("No population to save.")
            
        # use provided fitness; if missing compute it (fallback)
        if fitness is None:
            fitness = calculateReward(self.population)

        fitness_arr = np.array(fitness)
        best_idx = int(np.argmax(fitness_arr))
        best_ind = self.population[best_idx]
        best_arr = np.array(best_ind)

        out_path = Path(filename)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # make filename unique if it exists
        stem = out_path.stem
        suffix = out_path.suffix
        parent = out_path.parent
        candidate = parent / f"{stem}{suffix}"
        counter = 1
        while candidate.exists():
            candidate = parent / f"{stem}_{counter}{suffix}"
            counter += 1

        # write values only: one frame per line, comma-separated
        with candidate.open("w", encoding="utf-8") as f:
            for frame in best_arr:
                f.write(",".join(f"{float(v):.6g}" for v in frame) + "\n")

        print(f"Saved best individual (idx={best_idx}) to {candidate.resolve()}")

    def start(self):
        #generate population
        self.population = self.generatePopulation()

        best_hist = []   # store best fitness per generation

        # Create or overwrite generation log
        with open("generation_log.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["generation", "phase", "w_pose", "w_gait","pose_score", "gait_score", "best_fitness"])


        for i in range(self.generations):
            print(f"Generation {i}/{self.generations}")
   
            # ------------------------------------------------
            # Adaptive Weighting (Phase-based)
            # ------------------------------------------------

            # Normalized progress (0 → 1)
            t = i / self.generations

            # Weight Transition
            fitness.W_POSE_QUALITY = 3 * (1 - t) + 1.0 * t
            fitness.W_GAIT_DYNAMICS = 1.0 * (1 - t) + 3 * t

            # Label the current phase for logging
            if t < 0.33:
                phase_label = "Posture Training"
            elif t < 0.66:
                phase_label = "Balance & Motion"
            else:
                phase_label = "Gait Optimization"

            print(f"→ Phase: {phase_label} | t={t:.2f} | W_pose={fitness.W_POSE_QUALITY:.2f}, W_gait={fitness.W_GAIT_DYNAMICS:.2f}")

            # ------------------------------------------------
            # Evaluate fitness for current population
            # ------------------------------------------------
            fit_scores = calculateReward(self.population)
            self.save_population_txt(fitness=fit_scores)

            # Find the best individual before selection
            best_idx = int(np.argmax(fit_scores))

            # ------------------------------------------------
            # Debug: compare static vs dynamic contributions
            # ------------------------------------------------
            best_idx = int(np.argmax(fit_scores))
            best_ind = self.population[best_idx]

            # Pose (static) score
            pose_scores = [fitness._calculate_pose_quality_score(frame) for frame in best_ind]
            pose_static = np.mean(pose_scores)

            # Dynamic score
            positions_data = [fitness.getPositions(frame_angles) for frame_angles in best_ind]
            foot_positions = np.array([[leg_joints[3] for leg_joints in frame] for frame in positions_data])
            dyn_dynamic = fitness._calculate_gait_dynamics_reward(foot_positions)

            # Weighted total 
            final_score = (fitness.W_POSE_QUALITY * pose_static) + (fitness.W_GAIT_DYNAMICS * dyn_dynamic)

            print(f" [Static (Pose): {pose_static:+.3f} | Dynamic (Gait): {dyn_dynamic:+.3f}] "
                f"| Final = {final_score:+.3f}")


            best_individual = self.population[best_idx].copy()
            best_fitness = np.max(fit_scores)
            best_hist.append(best_fitness)

            print(f'Generation {i}: Best Fitness = {best_fitness}')
            print("-" * 50)

            # Log generation stats (append)
            with open("generation_log.csv", "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    i, phase_label, fitness.W_POSE_QUALITY, fitness.W_GAIT_DYNAMICS,
                      pose_static, dyn_dynamic, best_fitness
                ])

            # ------------------------------------------------
            # GA operations: selection → crossover → mutation
            # ------------------------------------------------
            selectedParents = self.tournamentSelection(fit_scores)
            #crossover
            offspring = self.crossover(selectedParents)
            #mutate
            mutated_offspring = self.mutate(offspring, self.mutationRate)
            # Keep the best individual (elitism)
            mutated_offspring[0] = best_individual
            self.population = mutated_offspring
            
    
        # Show the best individual
        input("Type anything if ready")
        best_idx = int(np.argmax(fitness))
        best = self.population[best_idx]

        print("Opening spider animation for best individual...")
        visualiseWalk(best)

    #generates 100 x 300 x 24 frames
    #the numpy way!

    def generatePopulation(self) -> np.ndarray[np.float64]:
        print("Generating Population...")
        population = np.zeros((self.populationSize, self.frames, self.individualLength), dtype=np.float64)
        size = (self.populationSize, self.frames, self.individualLength // 3)

        population[:, :, 0::3] = self.rng.uniform(self.coxaMin, self.coxaMax, size=size)
        population[:, :, 1::3] = self.rng.uniform(self.femurMin, self.femurMax, size=size)
        population[:, :, 2::3] = self.rng.uniform(self.tibiaMin, self.tibiaMax, size=size)

        seed_file = Path("smart_seed_gait.npy")
        if seed_file.exists():
            try:
                print(f"Loading seed file: {seed_file}")
                seed_gait = np.load(seed_file)
                if seed_gait.shape == (self.frames, self.individualLength):
                    population[0] = seed_gait  # Insert seed as the first individual
                    print("Successfully seeded population with smart_seed_gait.npy")
                else:
                    # This handles if user runs Mode 2/3 (100/200 frames)
                    print(
                        f"Seed file shape mismatch. Expected {(self.frames, self.individualLength)}, got {seed_gait.shape}. Using partial seed.")
                    # Use as much of the seed as possible
                    frames_to_copy = min(self.frames, seed_gait.shape[0])
                    population[0, :frames_to_copy, :] = seed_gait[:frames_to_copy, :]

            except Exception as e:
                print(f"Error loading seed file: {e}")
        else:
            print("smart_seed_gait.npy not found. Starting from random.")

        return population

    def tournamentSelection(self, fitness) -> np.ndarray[np.float64]:
        if not isinstance(fitness, np.ndarray):
            print(f"\033[93m[WARNING]: Fitness was not np.ndarray — auto-converting.\033[0m")
            fitness_array = np.array(fitness, dtype=np.float64)
        else:
            fitness_array = fitness

        popSize = len(self.population)
        selected = []
        tournament_size = 5

        for _ in range(popSize):
            # Select 'tournament_size' random indices
            indices = self.rng.integers(0, popSize, tournament_size)

            # Get the fitness for those individuals
            tournament_fitnesses = [fitness_array[i] for i in indices]

            # Find the index of the winner (best fitness) within that group
            winner_local_idx = np.argmax(tournament_fitnesses)
            winner_global_idx = indices[winner_local_idx]

            # Add a copy of the winner to the selected pool
            selected.append(np.copy(self.population[winner_global_idx]))

        return np.array(selected)


    def crossover(self, parents):
        """
        Performs uniform crossover at the joint level for each frame to prevent twitches and jumps of the joints
        """
        num_parents, num_frames, num_joints = parents.shape
        offspring = np.zeros_like(parents)

        for i in range(0, num_parents - 1, 2):
            parent1 = parents[i]
            parent2 = parents[i + 1]

            child1 = np.zeros_like(parent1)
            child2 = np.zeros_like(parent2)

            for f in range(num_frames):
                mask = self.rng.integers(0, 2, size=num_joints)

                child1[f] = np.where(mask, parent1[f], parent2[f])
                child2[f] = np.where(mask, parent2[f], parent1[f])

            offspring[i] = child1
            offspring[i + 1] = child2

        if num_parents % 2 == 1:
            offspring[-1, :] = parents[-1, :]  # Carry over last parent if odd

        return offspring

        # REPLACE your existing 'mutate' function with this:

    def mutate(self, offspring, mutation_rate):
        """
        Apply gaussian noise mutation to fine tune joint movements
        """
        mutated = offspring.copy()
        if mutated.ndim != 3:
            return mutated

        # Standard deviation for noise (e.g., ~0.087 rad = 5 degrees)
        # This controls "how big" the creep/mutation is.
        progress = getattr(self, "current_gen", 0) / max(self.generations - 1, 1)
        mutation_strength = 0.1 * (1 - progress) + 0.02 * progress


        num_ind, num_frames, num_joints = mutated.shape
        for ind in range(num_ind):
            for fr in range(num_frames):
                for j in range(num_joints):
                    if self.rng.random() < mutation_rate:
                        # 1. Add Gaussian noise
                        noise = self.rng.normal(0, mutation_strength)
                        new_angle = mutated[ind, fr, j] + noise

                        # 2. Clamp the angle to its valid limit
                        jt = j % 3
                        if jt == 0:
                            new_angle = np.clip(new_angle, self.coxaMin, self.coxaMax)
                        elif jt == 1:
                            new_angle = np.clip(new_angle, self.femurMin, self.femurMax)
                        else:
                            new_angle = np.clip(new_angle, self.tibiaMin, self.tibiaMax)

                        mutated[ind, fr, j] = new_angle
        return mutated