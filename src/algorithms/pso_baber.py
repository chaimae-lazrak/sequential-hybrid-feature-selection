#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PSO-bABER Hybrid Algorithm - Sequential Strategy
Conforme à l'architecture existante

@author: chamaelazrak
Created: February 2026
"""

import numpy as np
from typing import Tuple
from .base_algorithm import BaseMetaheuristic
from src.utils.binary_utils import BinaryConverter
from experiments.config import ExperimentConfig


class PSO_bABER(BaseMetaheuristic):
    """
    Hybride PSO-bABER avec Stratégie Séquentielle
    
    Stratégie:
    - Phase 1 (0-60%): PSO uniquement
    - Phase 2 (60-100%): bABER uniquement
    - Transition rigide à iteration = 60% * max_iter
    
    Configuration Phase 1: 15/100/15
    - Transition à iteration 60 (60% de 100)
    """
    
    def __init__(self, n_features, pop_size, max_iter, fitness_func,
                 config=ExperimentConfig, 
                 transition_ratio=None,
                 # PSO params
                 w=None, c1=None, c2=None,
                 # bABER params
                 exploration_ratio_init=None, exploration_ratio_final=None,
                 h_range=None, mutation_threshold=None,
                 **kwargs):
        """Initialise PSO-bABER"""
        super().__init__(n_features, pop_size, max_iter, fitness_func, **kwargs)
        
        # Charger paramètres hybride
        try:
            hybrid_cfg = config.get_algorithm_params("PSO_bABER")
            self.transition_ratio = hybrid_cfg.get('transition_ratio', 0.6) if transition_ratio is None else transition_ratio
        except:
            self.transition_ratio = 0.6 if transition_ratio is None else transition_ratio
        
        self.transition_iter = int(self.transition_ratio * self.max_iter)
        
        # ====== PARAMÈTRES PSO ======
        pso_cfg = config.get_algorithm_params("PSO")
        self.w = pso_cfg.get("w", 0.9) if w is None else w
        self.c1 = pso_cfg.get("c1", 2.0) if c1 is None else c1
        self.c2 = pso_cfg.get("c2", 2.0) if c2 is None else c2
        
        # État PSO
        self.velocities = None
        self.pbest = None
        self.pbest_fitness = None
        
        # ====== PARAMÈTRES bABER ======
        baber_cfg = config.get_algorithm_params("bABER")
        self.exploration_ratio_init = baber_cfg.get('exploration_ratio_init', 0.7) if exploration_ratio_init is None else exploration_ratio_init
        self.exploration_ratio_final = baber_cfg.get('exploration_ratio_final', 0.3) if exploration_ratio_final is None else exploration_ratio_final
        self.h_range = baber_cfg.get('h_range', [0, 2]) if h_range is None else h_range
        self.mutation_threshold = baber_cfg.get('mutation_threshold', 3) if mutation_threshold is None else mutation_threshold
        
        # État bABER
        self.no_improvement_count = None
        
        # ====== ÉTAT GLOBAL ======
        self.gbest = None
        self.gbest_fitness = -np.inf
        
    def initialize_population(self):
        """Initialise la population et les états PSO et bABER"""
        population = np.random.randint(0, 2, (self.pop_size, self.n_features))
        
        for i in range(self.pop_size):
            if population[i].sum() == 0:
                population[i][np.random.randint(self.n_features)] = 1
        
        # ====== INITIALISATION PSO ======
        self.velocities = np.random.uniform(-1, 1, (self.pop_size, self.n_features))
        self.pbest = population.copy()
        self.pbest_fitness = np.array([self.evaluate_fitness(ind) for ind in population])
        
        # ====== INITIALISATION bABER ======
        self.no_improvement_count = np.zeros(self.pop_size, dtype=int)
        
        # ====== INITIALISATION GBEST ======
        best_idx = np.argmax(self.pbest_fitness)
        self.gbest = population[best_idx].copy()
        self.gbest_fitness = self.pbest_fitness[best_idx]
        
        return population
    
    def update_population(self, population: np.ndarray, iteration: int) -> np.ndarray:
        """Mise à jour selon stratégie séquentielle"""
        if iteration <= self.transition_iter:
            # PHASE 1: PSO
            return self._pso_update(population, iteration)
        else:
            # PHASE 2: bABER
            return self._baber_update(population, iteration)
    
    # ========================================================================
    # PHASE PSO
    # ========================================================================
    
    def _pso_update(self, population: np.ndarray, iteration: int) -> np.ndarray:
        """Mise à jour PSO standard"""
        for i in range(self.pop_size):
            # Update velocity
            r1, r2 = np.random.rand(2)
            cognitive = self.c1 * r1 * (self.pbest[i] - population[i])
            social = self.c2 * r2 * (self.gbest - population[i])
            self.velocities[i] = self.w * self.velocities[i] + cognitive + social
            
            # Update position (binary)
            population[i] = BinaryConverter.to_binary(self.velocities[i])
            
            # Assurer au moins une feature
            if population[i].sum() == 0:
                population[i][np.random.randint(self.n_features)] = 1
            
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
    
    # ========================================================================
    # PHASE bABER (copié de votre baber.py)
    # ========================================================================
    
    def _baber_update(self, population: np.ndarray, iteration: int) -> np.ndarray:
        """Mise à jour bABER"""
        n_exploration, n_exploitation = self._get_exploration_exploitation_sizes(iteration)
        
        indices = np.random.permutation(self.pop_size)
        population = population[indices]
        self.no_improvement_count = self.no_improvement_count[indices]
        
        fitness_values = np.array([self.evaluate_fitness(ind) for ind in population])
        new_population = []
        
        # Groupe exploration
        for i in range(n_exploration):
            new_solution = self._exploration_update(population[i])
            new_fitness = self.evaluate_fitness(new_solution)
            
            if new_fitness > fitness_values[i]:
                new_population.append(new_solution)
                self.no_improvement_count[i] = 0
                
                if new_fitness > self.gbest_fitness:
                    self.gbest = new_solution.copy()
                    self.gbest_fitness = new_fitness
            else:
                self.no_improvement_count[i] += 1
                
                if self.no_improvement_count[i] >= self.mutation_threshold:
                    mutated_solution = self._mutation_operation()
                    mutated_fitness = self.evaluate_fitness(mutated_solution)
                    
                    if mutated_fitness > fitness_values[i]:
                        new_population.append(mutated_solution)
                        if mutated_fitness > self.gbest_fitness:
                            self.gbest = mutated_solution.copy()
                            self.gbest_fitness = mutated_fitness
                    else:
                        new_population.append(population[i])
                    
                    self.no_improvement_count[i] = 0
                else:
                    new_population.append(population[i])
        
        # Groupe exploitation
        for i in range(n_exploration, self.pop_size):
            if np.random.rand() < 0.5:
                new_solution = self._exploitation_update_strategy1(population[i])
            else:
                new_solution = self._exploitation_update_strategy2(iteration)
            
            new_fitness = self.evaluate_fitness(new_solution)
            
            if new_fitness > fitness_values[i]:
                new_population.append(new_solution)
                if new_fitness > self.gbest_fitness:
                    self.gbest = new_solution.copy()
                    self.gbest_fitness = new_fitness
            else:
                new_population.append(population[i])
        
        return np.array(new_population)
    
    def _get_exploration_exploitation_sizes(self, iteration: int) -> Tuple[int, int]:
        exploration_ratio = (self.exploration_ratio_init - 
                           (self.exploration_ratio_init - self.exploration_ratio_final) * 
                           (iteration / self.max_iter))
        n_exploration = int(self.pop_size * exploration_ratio)
        n_exploitation = self.pop_size - n_exploration
        return n_exploration, n_exploitation
    
    def _calculate_radius(self) -> float:
        h = np.random.uniform(self.h_range[0], self.h_range[1])
        x = np.random.uniform(0, 180)
        x_rad = np.deg2rad(x)
        cos_x = np.cos(x_rad)
        if cos_x >= 1:
            cos_x = 0.99
        radius = h * cos_x / (1 - cos_x)
        return radius
    
    def _exploration_update(self, solution: np.ndarray) -> np.ndarray:
        r1 = np.random.rand(self.n_features)
        r2 = np.random.rand(self.n_features)
        S_continuous = solution.astype(float)
        D = r1 * (S_continuous - 1)
        S_new_continuous = S_continuous + D * (2 * r2 - 1)
        S_new_binary = BinaryConverter.to_binary(S_new_continuous)
        if S_new_binary.sum() == 0:
            S_new_binary[np.random.randint(self.n_features)] = 1
        return S_new_binary
    
    def _exploitation_update_strategy1(self, solution: np.ndarray) -> np.ndarray:
        r2 = np.random.rand()
        r3 = np.random.rand(self.n_features)
        S_continuous = solution.astype(float)
        L_continuous = self.gbest.astype(float)
        D = r3 * (L_continuous - S_continuous)
        S_new_continuous = r2 * (S_continuous + D)
        S_new_binary = BinaryConverter.to_binary(S_new_continuous)
        if S_new_binary.sum() == 0:
            S_new_binary[np.random.randint(self.n_features)] = 1
        return S_new_binary
    
    def _exploitation_update_strategy2(self, iteration: int) -> np.ndarray:
        radius = self._calculate_radius()
        z = np.random.rand()
        k = z + 2 * (iteration ** 2) / (self.max_iter ** 2)
        S_star_continuous = self.gbest.astype(float)
        S_new_continuous = radius * (S_star_continuous + k)
        S_new_binary = BinaryConverter.to_binary(S_new_continuous)
        if S_new_binary.sum() == 0:
            S_new_binary[np.random.randint(self.n_features)] = 1
        return S_new_binary
    
    def _mutation_operation(self) -> np.ndarray:
        radius = self._calculate_radius()
        z2 = np.random.rand(self.n_features)
        k = np.random.rand(self.n_features)
        S_new_continuous = k * z2 - radius
        S_new_binary = BinaryConverter.to_binary(S_new_continuous)
        if S_new_binary.sum() == 0:
            S_new_binary[np.random.randint(self.n_features)] = 1
        return S_new_binary
