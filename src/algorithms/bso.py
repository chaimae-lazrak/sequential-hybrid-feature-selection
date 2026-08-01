#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 17:34:10 2026

@author: chamaelazrak
"""





import numpy as np
from .base_algorithm import BaseMetaheuristic
from experiments.config import ExperimentConfig
from src.utils.binary_utils import BinaryConverter


class BSO(BaseMetaheuristic):
    """Brain Storm Optimization pour sélection de features"""
    
    def __init__(self, n_features: int, pop_size: int, max_iter: int,
                 fitness_func: callable,config=ExperimentConfig, n_clusters: int = None, 
                 p_replace: float = None, p_one: float = None, **kwargs):
        super().__init__(n_features, pop_size, max_iter, fitness_func, **kwargs)
        
        bso_cfg=config.get_algorithm_params("BSO")
        self.n_clusters = min(bso_cfg["n_clusters"],pop_size) if n_clusters is None else min(n_clusters, pop_size)
        self.p_replace = bso_cfg["p_replace"] if p_replace is None else p_replace
        self.p_one = bso_cfg["p_one"] if p_one is None else p_one
        
    def initialize_population(self) -> np.ndarray:
        """Initialise la population"""
        return np.random.randint(0, 2, (self.pop_size, self.n_features))
    
    def cluster_population(self, population: np.ndarray, fitness_values: np.ndarray):
        """Clustering simple de la population"""
        # Tri par fitness
        sorted_indices = np.argsort(fitness_values)[::-1]
        
        # Division en clusters
        cluster_size = self.pop_size // self.n_clusters
        clusters = []
        
        for i in range(self.n_clusters):
            start = i * cluster_size
            end = start + cluster_size if i < self.n_clusters - 1 else self.pop_size
            cluster_indices = sorted_indices[start:end]
            clusters.append(population[cluster_indices])
        
        return clusters
    
    def update_population(self, population: np.ndarray, iteration: int) -> np.ndarray:
        """Mise à jour avec stratégie de brainstorming"""
        fitness_values = np.array([self.evaluate_fitness(ind) for ind in population])
        clusters = self.cluster_population(population, fitness_values)
        
        new_population = []
        
        for i in range(self.pop_size):
            if np.random.rand() < self.p_replace:
                # Générer nouvelle solution
                new_solution = np.random.randint(0, 2, self.n_features)
            else:
                # Sélectionner un cluster
                if np.random.rand() < self.p_one:
                    # Un seul cluster
                    cluster_idx = np.random.randint(self.n_clusters)
                    cluster = clusters[cluster_idx]
                    selected = cluster[np.random.randint(len(cluster))]
                else:
                    # Deux clusters
                    c1, c2 = np.random.choice(self.n_clusters, 2, replace=False)
                    ind1 = clusters[c1][np.random.randint(len(clusters[c1]))]
                    ind2 = clusters[c2][np.random.randint(len(clusters[c2]))]
                    # Crossover
                    mask = np.random.randint(0, 2, self.n_features)
                    selected = np.where(mask == 1, ind1, ind2)
                
                # Mutation
                mutation_rate = 0.1 * (1 - iteration / self.max_iter)
                mutation_mask = np.random.rand(self.n_features) < mutation_rate
                new_solution = selected.copy()
                new_solution[mutation_mask] = 1 - new_solution[mutation_mask]
            
            new_population.append(new_solution)
        
        return np.array(new_population)

