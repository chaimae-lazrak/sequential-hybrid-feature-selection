#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BSO-bABER Sequential Hybrid Algorithm
Created on February 2026

@author: chamaelazrak

Hybridation séquentielle:
- Phase 1 (60%): BSO pour exploration cluster-based brainstorming
- Phase 2 (40%): bABER pour raffinement géométrique local
"""

import numpy as np
from .base_algorithm import BaseMetaheuristic

from src.algorithms.bso import BSO
from src.algorithms.baber import bABER

from experiments.config import ExperimentConfig
from src.utils.binary_utils import BinaryConverter


class BSO_bABER(BaseMetaheuristic):
    """
    Hybride séquentiel BSO-bABER
    
    Stratégie:
    - Phase 1 (iterations 1-60): BSO exploration cluster-based brainstorming
    - Transfer (iteration 60): Direct initialization
    - Phase 2 (iterations 61-100): bABER raffinement géométrique
    """
    
    def __init__(self, n_features: int, pop_size: int, max_iter: int,
                 fitness_func: callable, config=ExperimentConfig,
                 transition_ratio: float = 0.6, **kwargs):
        """
        Initialise BSO-bABER
        
        Args:
            n_features: Nombre de features
            pop_size: Taille de la population
            max_iter: Nombre total d'itérations
            fitness_func: Fonction de fitness
            config: Configuration
            transition_ratio: Ratio de transition Phase 1→Phase 2 (default: 0.6 = 60%)
            **kwargs: Paramètres supplémentaires
        """
        super().__init__(n_features, pop_size, max_iter, fitness_func, **kwargs)
        
        self.transition_ratio = transition_ratio
        self.transition_iter = int(max_iter * transition_ratio)
        
        # Paramètres BSO
        bso_params = config.get_algorithm_params("BSO")
        self.bso_n_clusters = bso_params.get('n_clusters', 5)
        self.bso_p_replace = bso_params.get('p_replace', 0.2)
        self.bso_p_one = bso_params.get('p_one', 0.8)
        
        # Paramètres bABER
        baber_params = config.get_algorithm_params("bABER")
        self.baber_exploration_init = baber_params.get('exploration_ratio_init', 0.7)
        self.baber_exploration_final = baber_params.get('exploration_ratio_final', 0.3)
        self.baber_h_range = baber_params.get('h_range', [0, 2])
        self.baber_mutation_threshold = baber_params.get('mutation_threshold', 3)
        
        # Algorithmes Phase 1 et Phase 2
        self.phase1_algo = None  # BSO
        self.phase2_algo = None  # bABER
        
        # Phase actuelle
        self.current_phase = 1
        
    def initialize_population(self) -> np.ndarray:
        """
        Initialise la population pour Phase 1 (BSO)
        """
        # Créer instance BSO pour Phase 1
        self.phase1_algo = BSO(
            n_features=self.n_features,
            pop_size=self.pop_size,
            max_iter=self.transition_iter,
            fitness_func=self.fitness_func,
            n_clusters=self.bso_n_clusters,
            p_replace=self.bso_p_replace,
            p_one=self.bso_p_one
        )
        
        # Initialiser population BSO
        population = self.phase1_algo.initialize_population()
        
        print(f"\n   🔵 PHASE 1: BSO (iterations 1-{self.transition_iter})")
        print(f"      Exploration cluster-based brainstorming ({self.bso_n_clusters} clusters)")
        
        return population
    
    def _transfer_to_phase2(self, population: np.ndarray) -> np.ndarray:
        """
        Transfer de Phase 1 (BSO) vers Phase 2 (bABER)
        Stratégie: Direct Initialization
        
        Args:
            population: Population finale de BSO
            
        Returns:
            Population initiale pour bABER (même solutions)
        """
        print(f"\n   🔄 TRANSFER (iteration {self.transition_iter})")
        print(f"      Stratégie: Direct Initialization")
        print(f"      Transfert des {self.pop_size} idées (de {self.bso_n_clusters} clusters) → bABER")
        
        # Créer instance bABER pour Phase 2
        # Ajuster max_iter pour bABER (iterations restantes)
        remaining_iter = self.max_iter - self.transition_iter
        
        self.phase2_algo = bABER(
            n_features=self.n_features,
            pop_size=self.pop_size,
            max_iter=remaining_iter,
            fitness_func=self.fitness_func,
            exploration_ratio_init=self.baber_exploration_init,
            exploration_ratio_final=self.baber_exploration_final,
            h_range=self.baber_h_range,
            mutation_threshold=self.baber_mutation_threshold
        )
        
        # Direct initialization: population BSO → population bABER
        
        # Initialiser compteur de non-amélioration bABER
        self.phase2_algo.no_improvement_count = np.zeros(self.pop_size, dtype=int)
        
        # Initialiser gbest bABER avec meilleure solution BSO
        fitness_values = np.array([self.fitness_func(ind) for ind in population])
        best_idx = np.argmax(fitness_values)
        self.phase2_algo.gbest = population[best_idx].copy()
        self.phase2_algo.gbest_fitness = fitness_values[best_idx]
        
        print(f"\n   🟢 PHASE 2: bABER (iterations {self.transition_iter+1}-{self.max_iter})")
        print(f"      Raffinement géométrique Al-Biruni")
        
        self.current_phase = 2
        
        # Retourner population inchangée (direct transfer)
        return population.copy()
    
    def update_population(self, population: np.ndarray, iteration: int) -> np.ndarray:
        """
        Met à jour la population selon la phase actuelle
        
        Args:
            population: Population actuelle
            iteration: Itération actuelle (1-based)
            
        Returns:
            Population mise à jour
        """
        
        # PHASE 1: BSO (iterations 1 à transition_iter)
        if iteration <= self.transition_iter:
            # Utiliser update de BSO
            population = self.phase1_algo.update_population(population, iteration)
            return population
        
        # TRANSFER à l'iteration transition_iter + 1
        elif iteration == self.transition_iter + 1:
            # Transfer BSO → bABER
            population = self._transfer_to_phase2(population)
            
            # Première update bABER (iteration 1 pour bABER)
            baber_iteration = 1
            population = self.phase2_algo.update_population(population, baber_iteration)
            return population
        
        # PHASE 2: bABER (iterations transition_iter+2 à max_iter)
        else:
            # Utiliser update de bABER
            # Calculer iteration relative pour bABER
            baber_iteration = iteration - self.transition_iter
            population = self.phase2_algo.update_population(population, baber_iteration)
            return population


# ============================================================================
# TEST DU HYBRIDE
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print(" "*20 + "TEST BSO-bABER HYBRID")
    print("="*70)
    
    # Fonction fitness simple pour test
    def test_fitness(solution):
        """Fitness simple: favorise 50% de features"""
        n_selected = solution.sum()
        target = len(solution) / 2
        fitness = 1.0 - abs(n_selected - target) / len(solution)
        return fitness
    
    # Paramètres de test
    n_features = 20
    pop_size = 10
    max_iter = 100
    transition_ratio = 0.6
    
    print(f"\nConfiguration:")
    print(f"  Features: {n_features}")
    print(f"  Population: {pop_size}")
    print(f"  Itérations totales: {max_iter}")
    print(f"  Transition ratio: {transition_ratio} (Phase 1: {int(max_iter*transition_ratio)}, Phase 2: {max_iter - int(max_iter*transition_ratio)})")
    
    # Créer et exécuter hybride
    print("\nExécution de BSO-bABER...")
    hybrid = BSO_bABER(
        n_features=n_features,
        pop_size=pop_size,
        max_iter=max_iter,
        fitness_func=test_fitness,
        transition_ratio=transition_ratio
    )
    
    # Exécuter
    best_solution, best_fitness, convergence = hybrid.run()
    
    print("\n" + "="*70)
    print("RÉSULTATS")
    print("="*70)
    print(f"Meilleure solution: {best_solution}")
    print(f"Features sélectionnées: {np.where(best_solution == 1)[0]}")
    print(f"Nombre de features: {best_solution.sum()}/{n_features}")
    print(f"Meilleure fitness: {best_fitness:.6f}")
    
    # Afficher convergence phases
    print(f"\nConvergence Phase 1 (BSO, 5 premières):")
    for i in range(min(5, int(max_iter*transition_ratio))):
        print(f"  Iter {i+1}: {convergence[i]:.6f}")
    
    print(f"\nConvergence Phase 2 (bABER, 5 premières après transfer):")
    start_idx = int(max_iter*transition_ratio)
    for i in range(min(5, max_iter - start_idx)):
        print(f"  Iter {start_idx + i + 1}: {convergence[start_idx + i]:.6f}")
    
    print("\n" + "="*70)
    print("✅ Test BSO-bABER réussi!")
    print("="*70)