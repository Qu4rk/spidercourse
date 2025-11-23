import sys, os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
# from visualiser import visualiseWalk
from fitness import calculateFitness
from config import *

class GeneticAlgorithm:
    def __init__(self, populationSize: int, individualLength: int, generations: int, mutationRate: float, frames: int, seed: int | None = None):
        self.frames = 100 # forced to 100 and then cloned up to outputFrames
        self.outputFrames = frames 
        self.populationSize = populationSize
        self.individualLength = individualLength
        self.generations = generations
        self.mutationRate = mutationRate
        self.rng = np.random.default_rng(seed)
        self.bestFitnessHistory = []
        self.avgFitnessHistory = []

    def logFitness(self, fitness, gen):
        bestIndex = int(np.argmax(fitness))
        bestFitness = float(fitness[bestIndex])
        avgFitness = float(np.mean(fitness))
        print(f"Generation {gen}: Best Fitness = {bestFitness:.4f}, Avg Fitness = {avgFitness:.4f}")
            
        self.bestFitnessHistory.append(bestFitness)
        self.avgFitnessHistory.append(bestFitness)

    def start(self):
        self.clearFiles(path="output/runs")
        self.clearFiles(path="output/logs")

        self.population = self.generatePopulation()

        for gen in range(self.generations):
            print(f"Generation {gen}/{self.generations}")

            fitness = calculateFitness(self.population)
            self.logFitness(fitness, gen)
            parents = self.tournamentSelection(fitness)
            offspring = self.crossover(parents)
            mutated_offspring = self.mutate(offspring, self.mutationRate)

            self.population = mutated_offspring

        # Final steps
        final_fitness = calculateFitness(self.population)
        final_arr = np.array(final_fitness, dtype=np.float64)
        final_arr[~np.isfinite(final_arr)] = -1e300
        best_idx = int(np.argmax(final_arr))
        
        # Get the best 100-frame individual
        best_low_res = self.population[best_idx]

        # UPSCALING: Convert 100 frames -> 300 frames (or whatever user requested)
        best_high_res = self.interpolate_genes(best_low_res, self.outputFrames)

        # Save the High-Res version
        self.save_population_txt(individual=best_high_res, filename="best_individual.txt")

        print(f"Opening spider animation for best individual ({self.outputFrames} frames)...")
        visualiseWalk(best_high_res)
        
        self.plot_fitness_history()

    def generatePopulation(self) -> np.ndarray[np.float64]:
        print(f"Generating Population with Wide Bounds ({ANGLE_MIN} to {ANGLE_MAX})...")
        population = np.zeros((self.populationSize, self.frames, self.individualLength), dtype=np.float64)
        size = (self.populationSize, self.frames, self.individualLength//3)
        
        population[:, :, 0::3] = self.rng.uniform(ANGLE_MIN, ANGLE_MAX, size=size)
        population[:, :, 1::3] = self.rng.uniform(ANGLE_MIN, ANGLE_MAX, size=size)
        population[:, :, 2::3] = self.rng.uniform(ANGLE_MIN, ANGLE_MAX, size=size)

        return population

    def tournamentSelection(self, fitness) -> np.ndarray[np.float64]:
        competitors = self.rng.integers(0, self.populationSize, size=(self.populationSize, TOURNAMENT_SIZE))
        winnerIndices = np.argmax(fitness[competitors], axis=1)
        winners = competitors[np.arange(self.populationSize), winnerIndices]
        return self.population[winners]

    def crossover(self, parents) -> np.ndarray:
        numParents, numFrames, numJoints = parents.shape
        individualLength = numFrames * numJoints
        
        flatParents = parents.reshape(numParents, individualLength)
        flatOffspring = np.empty_like(flatParents)

        pairs = numParents // 2

        parents_a = flatParents[0 : 2*pairs : 2]
        parents_b = flatParents[1 : 2*pairs : 2]

        cut_frames = self.rng.integers(1, numFrames, size=(pairs, 1))
        cut_indices = cut_frames * numJoints 
        range_indices = np.arange(individualLength)
        mask = range_indices < cut_indices

        flatOffspring[0 : 2*pairs : 2] = np.where(mask, parents_a, parents_b)
        flatOffspring[1 : 2*pairs : 2] = np.where(mask, parents_b, parents_a)

        if numParents % 2 == 1:
            flatOffspring[-1] = flatParents[-1]

        return flatOffspring.reshape(numParents, numFrames, numJoints)

    def mutate(self, offspring, mutation_rate):
        """
        Uses gentle Gaussian noise for fine-tuning.
        """
        mutated = offspring.copy()
        mutation_strength = 0.25 
        mutation_mask = self.rng.random(mutated.shape) < mutation_rate
        noise = self.rng.normal(0, mutation_strength, size=mutated.shape)
        mutated += np.where(mutation_mask, noise, 0)
        
        return mutated

    def clearFiles(self, path):
        os.makedirs(path, exist_ok=True)
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")

    def save_population_txt(self, individual, filename: str = "best_individual.txt"):
        """
        Saves the specific individual passed to it (already upscaled).
        """
        if individual is None:
            print("No individual to save.")
            return

        out_path = Path(f"output/runs/{filename}")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        stem = out_path.stem
        suffix = out_path.suffix
        parent = out_path.parent
        candidate = parent / f"{stem}{suffix}"
        counter = 1
        while candidate.exists():
            candidate = parent / f"{stem}_{counter}{suffix}"
            counter += 1

        with candidate.open("w", encoding="utf-8") as f:
            n_frames = individual.shape[0]
            for frame_idx in range(n_frames):
                angles = individual[frame_idx]  
                row = np.concatenate(([frame_idx], angles))
                f.write(",".join(f"{float(v):.6g}" for v in row) + "\n")

        print(f"Saved best individual to {candidate.resolve()}")

    def plot_fitness_history(self):
        """Plots the best and average fitness history."""
        plt.figure(figsize=(10, 6))
        generations = range(len(self.best_fitness_history))

        plt.plot(generations, self.best_fitness_history, label='Best Fitness', color='green', linewidth=2)
        plt.plot(generations, self.avg_fitness_history, label='Average Fitness', color='blue', linestyle='--')

        plt.title('Genetic Algorithm Convergence (Fitness vs. Generation)')
        plt.xlabel('Generation')
        plt.ylabel('Fitness Score (Max 100.0)')
        plt.grid(True, alpha=0.5)
        plt.legend()
        plt.tight_layout()

        plot_path = Path("output/logs/fitness_convergence.png")
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path)
        plt.close()

        print(f"\nSaved convergence plot to {plot_path.resolve()}")