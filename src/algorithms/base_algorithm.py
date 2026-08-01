#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 17:16:16 2026

@author: chamaelazrak
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Dict, Any

class BaseMetaheuristic(ABC):
    """Classe abstraite de base pour tous les algorithmes"""
    
    def __init__(self, n_features, pop_size, max_iter, fitness_func, **kwargs):
        self.n_features = n_features
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.fitness_func = fitness_func
        self.params = kwargs
        
        # Historique
        self.best_solution = None
        self.best_fitness = float('-inf')
        self.convergence_curve = []
        
    @abstractmethod
    def initialize_population(self) :
        """Initialise la population"""
        pass
    
    @abstractmethod
    def update_population(self, population, iteration) :
        """Mise à jour de la population"""
        pass
    
    def evaluate_fitness(self, solution) :
        """Évalue la fitness d'une solution"""
        return self.fitness_func(solution)
    
    def run(self) :
        """Exécute l'algorithme"""
        population = self.initialize_population()
        
        for iteration in range(self.max_iter):
            # Évaluation
            fitness_values = np.array([self.evaluate_fitness(ind) for ind in population])
            
            # Mise à jour du meilleur
            best_idx = np.argmax(fitness_values)
            if fitness_values[best_idx] > self.best_fitness:
                self.best_fitness = fitness_values[best_idx]
                self.best_solution = population[best_idx].copy()
            
            self.convergence_curve.append(self.best_fitness)
            
            # Mise à jour de la population
            population = self.update_population(population, iteration)
            
            # Callback pour logging
            if iteration % 10 == 0:
                print(f"Iteration {iteration}: Best Fitness = {self.best_fitness:.4f}")
        
        return self.best_solution, self.best_fitness, self.convergence_curve