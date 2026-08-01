#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 17:27:28 2026

@author: chamaelazrak
"""


import numpy as np
from .base_algorithm import BaseMetaheuristic

from experiments.config import ExperimentConfig
from src.utils.binary_utils import BinaryConverter

class PSO(BaseMetaheuristic):
    """Particle Swarm Optimization binaire"""
    
    def __init__(self, n_features: int, pop_size: int, max_iter: int,
                 fitness_func: callable, config=ExperimentConfig, w: float = None, c1: float = None, 
                 c2: float = None,  **kwargs):
        super().__init__(n_features, pop_size, max_iter, fitness_func, **kwargs)
        pso_cfg=config.get_algorithm_params("PSO")
        
        
        self.w = pso_cfg["w"]  if w is  None else w  # Inertie
        self.c1 = pso_cfg["c1"] if c1 is  None else c1   # Cognitive
        self.c2 = pso_cfg["c2"] if c2 is  None else c2   # Social
        
        
        self.velocities = None
        self.pbest = None
        self.pbest_fitness = None
        self.gbest = None
        self.gbest_fitness = float('-inf')
        
    def initialize_population(self) -> np.ndarray:
        """Initialize particles"""
        population = np.random.randint(0, 2, (self.pop_size, self.n_features))
        self.velocities = np.random.uniform(-1, 1, (self.pop_size, self.n_features))
        self.pbest = population.copy()
        self.pbest_fitness = np.array([self.evaluate_fitness(ind) for ind in population])
        
        best_idx = np.argmax(self.pbest_fitness)
        self.gbest = self.pbest[best_idx].copy()
        self.gbest_fitness = self.pbest_fitness[best_idx]
        
        return population
    
    def update_population(self, population: np.ndarray, iteration: int) -> np.ndarray:
        """Update particles"""
        for i in range(self.pop_size):
            # Update velocity
            r1, r2 = np.random.rand(2)
            cognitive = self.c1 * r1 * (self.pbest[i] - population[i])
            social = self.c2 * r2 * (self.gbest - population[i])
            self.velocities[i] = self.w * self.velocities[i] + cognitive + social
            
            # Update position (binary)
            population[i] = BinaryConverter.to_binary(self.velocities[i])
            
            # Update pbest
            fitness = self.evaluate_fitness(population[i])
            if fitness > self.pbest_fitness[i]:
                self.pbest[i] = population[i].copy()
                self.pbest_fitness[i] = fitness
                
                # Update gbest
                if fitness > self.gbest_fitness:
                    self.gbest = population[i].copy()
                    self.gbest_fitness = fitness
        
        return population
