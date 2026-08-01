#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 23 08:05:25 2026

@author: chamaelazrak
"""



import numpy as np
from typing import Tuple
from .base_algorithm import BaseMetaheuristic


from src.utils.binary_utils import BinaryConverter
from experiments.config import ExperimentConfig


class bABER(BaseMetaheuristic):
    """
    Binary Al-Biruni Earth Radius (bABER) - Version conforme à l'algorithme original
    
    Référence: El-kenawy et al. (2023) "Al-Biruni Earth Radius (BER) 
    Metaheuristic Search Optimization Algorithm"
    """
   
    def __init__(self, n_features, pop_size, max_iter, fitness_func,
                 config=ExperimentConfig,exploration_ratio_init=None,
                 exploration_ratio_final=None,
                 h_range=None,mutation_threshold=None, **kwargs):
        
        
        """
        Initialise bABER
        
        Args:
            n_features: Nombre de features
            pop_size: Taille de la population
            max_iter: Nombre maximum d'itérations
            fitness_func: Fonction de fitness
        """
        super().__init__(n_features, pop_size, max_iter, fitness_func, **kwargs)
        
        
        

        
        
        baber_cfg = config.get_algorithm_params("bABER")
        # Paramètres BER originaux
        self.exploration_ratio_init =baber_cfg['exploration_ratio_init'] if exploration_ratio_init is None else exploration_ratio_init # 70% exploration au début
        self.exploration_ratio_final =baber_cfg['exploration_ratio_final'] if exploration_ratio_final is None else exploration_ratio_final 
        self.h_range = baber_cfg['h_range'] if h_range is None else h_range  # [0,2]
        self.mutation_threshold = baber_cfg['mutation_threshold'] if mutation_threshold is None else mutation_threshold  # Mutation après 3 itérations sans amélioration
        
        # Historique pour la mutation
        self.no_improvement_count = None
        
        # Meilleure solution globale
        self.gbest = None
        self.gbest_fitness = -np.inf
        
    def initialize_population(self):
        """Initialise la population binaire"""
        population = np.random.randint(0, 2, (self.pop_size, self.n_features))
        
        # Assurer au moins une feature par solution
        for i in range(self.pop_size):
            if population[i].sum() == 0:
                population[i][np.random.randint(self.n_features)] = 1
                
        # Initialiser le compteur de non-amélioration
        self.no_improvement_count = np.zeros(self.pop_size, dtype=int)
        
        # Initialiser la meilleure solution
        fitness_values = np.array([self.evaluate_fitness(ind) for ind in population])
        best_idx = np.argmax(fitness_values)
        self.gbest = population[best_idx].copy()
        self.gbest_fitness = fitness_values[best_idx]
        
        return population
    
    
    def _calculate_radius(self) -> float:
        """
        Calcule le rayon selon la méthode Al-Biruni
        r = h * cos(x) / (1 - cos(x))
        
        Returns:
            Rayon calculé
        """
        h = np.random.uniform(self.h_range[0], self.h_range[1])
        x = np.random.uniform(0, 180)
        x_rad = np.deg2rad(x)
        cos_x = np.cos(x_rad)
        
        # Éviter division par zéro
        if cos_x >= 1:
            cos_x = 0.99
        
        radius = h * cos_x / (1 - cos_x)
        return radius
    
    def _get_exploration_exploitation_sizes(self, iteration: int) -> Tuple[int, int]:
        """
        Calcule dynamiquement la taille des groupes exploration/exploitation
        
        Args:
            iteration: Itération actuelle
            
        Returns:
            (n_exploration, n_exploitation)
        """
        # Ratio d'exploration décroissant linéairement
        exploration_ratio = (self.exploration_ratio_init - 
                           (self.exploration_ratio_init - self.exploration_ratio_final) * 
                           (iteration / self.max_iter))
        
        n_exploration = int(self.pop_size * exploration_ratio)
        n_exploitation = self.pop_size - n_exploration
        
        return n_exploration, n_exploitation

    
    
    
 
    def _exploration_update(self, solution: np.ndarray) -> np.ndarray:
        """
        Mise à jour exploration selon formule Al-Biruni
        
        Équations 3-5 de l'article:
        r = h*cos(x)/(1-cos(x))
        D = r1 * (S(t) - 1)
        S(t+1) = S(t) + D * (2*r2 - 1)
        
        Args:
            solution: Solution à mettre à jour
            
        Returns:
            Nouvelle solution binaire
        """
        # Vecteurs aléatoires
        r1 = np.random.rand(self.n_features)
        r2 = np.random.rand(self.n_features)
        
        # Conversion en continu pour calculs
        S_continuous = solution.astype(float)
        
        # Formule d'exploration BER originale (équations 4-5)
        D = r1 * (S_continuous - 1)
        S_new_continuous = S_continuous + D * (2 * r2 - 1)
        
        # Conversion binaire avec fonction de transfert
        S_new_binary = BinaryConverter.to_binary(S_new_continuous)
        
        # Assurer au moins une feature
        if S_new_binary.sum() == 0:
            S_new_binary[np.random.randint(self.n_features)] = 1
        
        return S_new_binary
    
    def _exploitation_update_strategy1(self, solution: np.ndarray) -> np.ndarray:
        """
        Stratégie d'exploitation 1: Se déplacer vers la meilleure solution
        
        Équations 6-7 de l'article:
        D = r3 * (L(t) - S(t))
        S(t+1) = r2 * (S(t) + D)
        
        Args:
            solution: Solution à mettre à jour
            
        Returns:
            Nouvelle solution binaire
        """
        # Vecteurs aléatoires
        r2 = np.random.rand()
        r3 = np.random.rand(self.n_features)
        
        # Conversion continue
        S_continuous = solution.astype(float)
        L_continuous = self.gbest.astype(float)
        
        # Formule exploitation 1 (équations 6-7)
        D = r3 * (L_continuous - S_continuous)
        S_new_continuous = r2 * (S_continuous + D)
        
        # Conversion binaire
        S_new_binary = BinaryConverter.to_binary(S_new_continuous)
        
        if S_new_binary.sum() == 0:
            S_new_binary[np.random.randint(self.n_features)] = 1
        
        return S_new_binary
    
    def _exploitation_update_strategy2(self, iteration: int) -> np.ndarray:
        """
        Stratégie d'exploitation 2: Chercher autour de la meilleure solution
        
        Équations 8-9 de l'article:
        k = z + 2*t²/N²
        S(t+1) = r * (S*(t) + k)
        
        Args:
            iteration: Itération actuelle
            
        Returns:
            Nouvelle solution binaire
        """
        # Calcul du rayon
        radius = self._calculate_radius()
        
        # Paramètre k (équation 9)
        z = np.random.rand()
        k = z + 2 * (iteration ** 2) / (self.max_iter ** 2)
        
        # Conversion continue
        S_star_continuous = self.gbest.astype(float)
        
        # Formule exploitation 2 (équation 8)
        S_new_continuous = radius * (S_star_continuous + k)
        
        # Conversion binaire
        S_new_binary = BinaryConverter.to_binary(S_new_continuous)
        
        if S_new_binary.sum() == 0:
            S_new_binary[np.random.randint(self.n_features)] = 1
        
        return S_new_binary
    
    def _mutation_operation(self) -> np.ndarray:
        """
        Opération de mutation
        
        Équation 10 de l'article:
        S(t+1) = k * z² - r
        où r = h*cos(x)/(1-cos(x))
        
        Returns:
            Nouvelle solution binaire mutée
        """
        # Calcul du rayon
        radius = self._calculate_radius()
        
        # Paramètres aléatoires
        z2 = np.random.rand(self.n_features)
        k = np.random.rand(self.n_features)
        
        # Formule de mutation (équation 10)
        S_new_continuous = k * z2 - radius
        
        # Conversion binaire
        S_new_binary = BinaryConverter.to_binary(S_new_continuous)
        
        if S_new_binary.sum() == 0:
            S_new_binary[np.random.randint(self.n_features)] = 1
        
        return S_new_binary 
    
    def update_population(self, population: np.ndarray, iteration: int) -> np.ndarray:
        """
        Met à jour la population selon l'algorithme BER
        
        Cette méthode est appelée par BaseMetaheuristic.run()
        
        Args:
            population: Population actuelle
            iteration: Itération actuelle
            
        Returns:
            Population mise à jour
        """
        # Calculer les tailles des groupes
        n_exploration, n_exploitation = self._get_exploration_exploitation_sizes(iteration)
        
        # Mélanger la population pour diversité (comme dans l'article)
        indices = np.random.permutation(self.pop_size)
        population = population[indices]
        self.no_improvement_count = self.no_improvement_count[indices]
        
        # Évaluation de la population
        fitness_values = np.array([self.evaluate_fitness(ind) for ind in population])
        
        # Nouvelle population
        new_population = []
        
        # ========== GROUPE D'EXPLORATION ==========
        for i in range(n_exploration):
            # Mise à jour exploration
            new_solution = self._exploration_update(population[i])
            
            # Évaluation
            new_fitness = self.evaluate_fitness(new_solution)
            
            # Sélection gloutonne
            if new_fitness > fitness_values[i]:
                new_population.append(new_solution)
                self.no_improvement_count[i] = 0
                
                # Mise à jour gbest
                if new_fitness > self.gbest_fitness:
                    self.gbest = new_solution.copy()
                    self.gbest_fitness = new_fitness
            else:
                # Incrémenter compteur de non-amélioration
                self.no_improvement_count[i] += 1
                
                # Vérifier si mutation nécessaire
                if self.no_improvement_count[i] >= self.mutation_threshold:
                    # Appliquer mutation
                    mutated_solution = self._mutation_operation()
                    mutated_fitness = self.evaluate_fitness(mutated_solution)
                    
                    if mutated_fitness > fitness_values[i]:
                        new_population.append(mutated_solution)
                        
                        # Mise à jour gbest
                        if mutated_fitness > self.gbest_fitness:
                            self.gbest = mutated_solution.copy()
                            self.gbest_fitness = mutated_fitness
                    else:
                        new_population.append(population[i])
                    
                    # Réinitialiser compteur
                    self.no_improvement_count[i] = 0
                else:
                    new_population.append(population[i])
        
        # ========== GROUPE D'EXPLOITATION ==========
        for i in range(n_exploration, self.pop_size):
            # Alterner entre les deux stratégies d'exploitation
            if np.random.rand() < 0.5:
                # Stratégie 1: Vers la meilleure solution
                new_solution = self._exploitation_update_strategy1(population[i])
            else:
                # Stratégie 2: Autour de la meilleure solution
                new_solution = self._exploitation_update_strategy2(iteration)
            
            # Évaluation
            new_fitness = self.evaluate_fitness(new_solution)
            
            # Sélection gloutonne
            if new_fitness > fitness_values[i]:
                new_population.append(new_solution)
                
                # Mise à jour gbest
                if new_fitness > self.gbest_fitness:
                    self.gbest = new_solution.copy()
                    self.gbest_fitness = new_fitness
            else:
                new_population.append(population[i])
        
        return np.array(new_population)


    
 
# Exemple d'utilisation
if __name__ == "__main__":
    print("="*70)
    print(" "*20 + "TEST bABER")
    print("="*70)
    
    # Fonction fitness simple pour test
    def test_fitness(solution):
        """Fitness simple: favorise les solutions avec 50% de features"""
        n_selected = solution.sum()
        target = len(solution) / 2
        fitness = 1.0 - abs(n_selected - target) / len(solution)
        return fitness
    
    # Paramètres de test
    n_features = 20
    pop_size = 10
    max_iter = 50
    
    print(f"\nConfiguration:")
    print(f"  Features: {n_features}")
    print(f"  Population: {pop_size}")
    print(f"  Itérations: {max_iter}")
    
    # Créer et exécuter bABER
    print("\nExécution de bABER...")
    optimizer = bABER(
        n_features=n_features,
        pop_size=pop_size,
        max_iter=max_iter,
        fitness_func=test_fitness
    )
    
    # Utiliser la méthode run() de BaseMetaheuristic
    best_solution, best_fitness, convergence = optimizer.run()
    
    print("\n" + "="*70)
    print("RÉSULTATS")
    print("="*70)
    print(f"Meilleure solution: {best_solution}")
    print(f"Features sélectionnées: {np.where(best_solution == 1)[0]}")
    print(f"Nombre de features: {best_solution.sum()}/{n_features}")
    print(f"Meilleure fitness: {best_fitness:.6f}")
    
    # Afficher convergence
    print(f"\nConvergence (5 premières itérations):")
    for i, fit in enumerate(convergence[:5]):
        print(f"  Iter {i+1}: {fit:.6f}")
    
    print("\n" + "="*70)
    print("✅ Test réussi!")
    print("="*70)