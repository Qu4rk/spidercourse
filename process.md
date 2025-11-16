Here are the specifics of our project

# Genetic algorithm:
The population is a ```npt.NDArray[np.float64]``` of shape ```(N, 300, 24)``` where N is an arbitrary population size
An Individual is a ```npt.NDArray[np.float64]``` of shape ```(300, 24)``` where 300 represents the number of frames
A Frame is a ```npt.NDArray[np.float64]``` of shape ```(24, )``` where 24 represents an angle in radians corresponding to angles on the spider


1. Generate Population
```python
def GeneratePopulation()
```
2. Rank fitness
3. use a tournamentSelection to select parents
4. Crossover
5. Mutate
6. step 2 N times