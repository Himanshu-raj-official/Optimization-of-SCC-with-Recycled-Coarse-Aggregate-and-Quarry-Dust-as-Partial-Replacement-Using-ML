"""
genetic_algorithm.py
---------------------
A lightweight, dependency-free real-valued Genetic Algorithm, used
as a direct Python replacement for MATLAB's `ga()` (Global Optimization
Toolbox) call in the original script. It optimizes continuous
hyperparameters by minimizing a user-supplied fitness function.

This mirrors:
    ga(fitnessFunction, 3, [], [], [], [], lb, ub, [], options)
"""

from dataclasses import dataclass, field
from typing import Callable, List

import numpy as np


@dataclass
class GAResult:
    best_solution: np.ndarray
    best_fitness: float
    history: List[float] = field(default_factory=list)  # best fitness per generation


class GeneticAlgorithm:
    """Real-valued GA with tournament selection, blend (BLX-alpha)
    crossover, gaussian mutation, and elitism."""

    def __init__(
        self,
        bounds_low,
        bounds_high,
        pop_size: int = 20,
        generations: int = 50,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.2,
        elite_fraction: float = 0.1,
        tournament_size: int = 3,
        seed: int = 28,
        verbose: bool = True,
    ):
        self.lb = np.asarray(bounds_low, dtype=float)
        self.ub = np.asarray(bounds_high, dtype=float)
        self.n_genes = len(self.lb)
        self.pop_size = pop_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.n_elite = max(1, int(round(elite_fraction * pop_size)))
        self.tournament_size = tournament_size
        self.rng = np.random.default_rng(seed)
        self.verbose = verbose

    def _init_population(self) -> np.ndarray:
        return self.rng.uniform(self.lb, self.ub, size=(self.pop_size, self.n_genes))

    def _clip(self, pop: np.ndarray) -> np.ndarray:
        return np.clip(pop, self.lb, self.ub)

    def _tournament_select(self, pop: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        idx = self.rng.integers(0, len(pop), size=self.tournament_size)
        best_idx = idx[np.argmin(fitness[idx])]  # minimization
        return pop[best_idx].copy()

    def _blend_crossover(self, p1: np.ndarray, p2: np.ndarray, alpha: float = 0.5):
        if self.rng.random() > self.crossover_rate:
            return p1.copy(), p2.copy()
        lo = np.minimum(p1, p2) - alpha * np.abs(p1 - p2)
        hi = np.maximum(p1, p2) + alpha * np.abs(p1 - p2)
        c1 = self.rng.uniform(lo, hi)
        c2 = self.rng.uniform(lo, hi)
        return c1, c2

    def _mutate(self, ind: np.ndarray) -> np.ndarray:
        mask = self.rng.random(self.n_genes) < self.mutation_rate
        sigma = 0.1 * (self.ub - self.lb)
        ind = ind.copy()
        ind[mask] += self.rng.normal(0, sigma[mask])
        return ind

    def optimize(self, fitness_fn: Callable[[np.ndarray], float]) -> GAResult:
        population = self._init_population()
        history = []
        best_solution, best_fitness = None, np.inf

        for gen in range(1, self.generations + 1):
            fitness = np.array([fitness_fn(ind) for ind in population])

            gen_best_idx = np.argmin(fitness)
            if fitness[gen_best_idx] < best_fitness:
                best_fitness = fitness[gen_best_idx]
                best_solution = population[gen_best_idx].copy()
            history.append(best_fitness)

            if self.verbose:
                print(f"  GA generation {gen:3d}/{self.generations} "
                      f"| best fitness = {best_fitness:.6f}")

            # Elitism
            elite_idx = np.argsort(fitness)[: self.n_elite]
            new_population = [population[i].copy() for i in elite_idx]

            # Breed the rest
            while len(new_population) < self.pop_size:
                p1 = self._tournament_select(population, fitness)
                p2 = self._tournament_select(population, fitness)
                c1, c2 = self._blend_crossover(p1, p2)
                c1, c2 = self._mutate(c1), self._mutate(c2)
                new_population.append(c1)
                if len(new_population) < self.pop_size:
                    new_population.append(c2)

            population = self._clip(np.array(new_population))

        return GAResult(best_solution=best_solution, best_fitness=best_fitness, history=history)
