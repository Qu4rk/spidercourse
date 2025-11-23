import sys
import os
from GA import GeneticAlgorithm
# from NN import Network

DATA_DIR = "data"
OUTPUT_DIR = "output"

def printMenu():
    print("\n" + "="*15)
    print(" AI Coursework")
    print("="*15)
    print("1. [GA] View Final Generation (Pre-computed)")
    print("2. [GA] Run Full Genetic Algorithm")
    print("3. [NN] View Neural Network Results (Pre-trained)")
    print("4. [NN] Run Full Neural Network Training")
    print("q. Quit")
    print("-" * 40)

def main():
    while True:
        printMenu()
        choice = input("Select an option: ")

        if choice == "1":
            pass
        elif choice == "2":
            ga = GeneticAlgorithm(
            populationSize=4000,
            individualLength=24, #number of joints
            generations=5000,
            mutationRate=0.001,
            frames=100,
            seed=42
            )
            ga.start()
            print(f"\n[i] Run complete. Best individual saved to {OUTPUT_DIR}")
        elif choice == "3":
            pass
        elif choice == "4":
            pass
        elif choice == "q":
            pass
        else:
            print("Please select a valid choice")
            

if __name__ == "__main__":
    main()
