#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 17:44:53 2026

@author: chamaelazrak
"""

"""
Harris Hawks Optimization pour sélection de features
src/algorithms/hho.py

Basé sur: Heidari et al. (2019) - Harris hawks optimization: 
Algorithm and applications
"""


import math
import numpy as np
from .base_algorithm import BaseMetaheuristic

from src.utils.binary_utils import BinaryConverter
from experiments.config import ExperimentConfig


class HHO(BaseMetaheuristic):
    """
    Harris Hawks Optimization pour sélection de features
    
    Inspiré du comportement de chasse coopérative des faucons de Harris
    """
    
    def __init__(self, n_features, pop_size, max_iter, fitness_func,config=ExperimentConfig, levy_beta=None, levy_scale=None, **kwargs):
        """
        Initialise HHO
        
        Args:
            n_features: Nombre de features
            pop_size: Taille de la population
            max_iter: Nombre maximum d'itérations
            fitness_func: Fonction de fitness
        """
        super().__init__(n_features, pop_size, max_iter, fitness_func, **kwargs)
        hho_cfg = config.get_algorithm_params("HHO")
        self.levy_beta  = hho_cfg["levy_beta"]  if levy_beta  is None else levy_beta
        self.levy_scale = hho_cfg["levy_scale"] if levy_scale is None else levy_scale
        
        self.rabbit_position = None  # Position de la proie (meilleure solution)
        self.rabbit_energy = None    # Énergie de la proie
        
    def initialize_population(self):
        """Initialise les hawks (faucons)"""
        # Population binaire aléatoire
        population = np.random.randint(0, 2, (self.pop_size, self.n_features))
        
        # Assurer qu'au moins une feature est sélectionnée
        for i in range(self.pop_size):
            if population[i].sum() == 0:
                population[i][np.random.randint(self.n_features)] = 1
        
        return population
    
    def update_population(self, population, iteration):
        """
        Mise à jour basée sur la stratégie de chasse des faucons
        
        Args:
            population: Population actuelle
            iteration: Itération actuelle
            
        Returns:
            Nouvelle population
        """
        
        # Calculer les fitness
        fitness_values = np.array([self.evaluate_fitness(ind) for ind in population])
        
        # Identifier la proie (meilleure solution)
        best_idx = np.argmax(fitness_values)
        self.rabbit_position = population[best_idx].copy()
        
        # Énergie du lapin (décroît avec les itérations)
        E0 = 2 * np.random.rand() - 1  # Énergie initiale [-1, 1]
        E = 2 * E0 * (1 - iteration / self.max_iter)  # Énergie décroissante
        self.rabbit_energy = E
        
        new_population = []
        
        for i in range(self.pop_size):
            # Position continue pour les calculs
            X_continuous = population[i].astype(float)
            
            # Phase d'exploration (|E| >= 1)
            if abs(E) >= 1:
                q = np.random.rand()
                rand_idx_1 = np.random.randint(0, self.pop_size)
                #rand_idx_2 = np.random.randint(0, self.pop_size)
                
                if q >= 0.5:
                    # Perching basé sur position aléatoire d'autres hawks
                    r1, r2 = np.random.rand(2)
                    X_rand = population[rand_idx_1].astype(float)
                    X_new = X_rand - r1 * np.abs(X_rand - 2 * r2 * X_continuous)
                else:
                    # Perching basé sur position du lapin et moyenne
                    X_rabbit = self.rabbit_position.astype(float)
                    X_m = np.mean(population.astype(float), axis=0)
                    r3, r4 = np.random.rand(2)
                    X_new = (X_rabbit - X_m) - r3 * (np.ones(self.n_features) - r4 * 2)
            
            # Phase d'exploitation (|E| < 1)
            else:
                r = np.random.rand()  # Probabilité de fuite du lapin
                
                if r >= 0.5 and abs(E) >= 0.5:
                    # Soft besiege
                    X_rabbit = self.rabbit_position.astype(float)
                    Delta_X = X_rabbit - X_continuous
                    X_new = Delta_X - E * np.abs(np.random.rand() * X_rabbit - X_continuous)
                
                elif r >= 0.5 and abs(E) < 0.5:
                    # Hard besiege
                    X_rabbit = self.rabbit_position.astype(float)
                    X_new = X_rabbit - E * np.abs(X_rabbit - X_continuous)
                
                elif r < 0.5 and abs(E) >= 0.5:
                    # Soft besiege with progressive rapid dives
                    X_rabbit = self.rabbit_position.astype(float)
                    Y = X_rabbit - E * np.abs(np.random.rand() * X_rabbit - X_continuous)
                    
                    # Levy flight
                    LF = self.levy_flight(self.n_features)
                    S = np.random.rand(self.n_features) * X_continuous
                    Z = Y + S * LF
                    
                    # Choisir la meilleure entre Y et Z
                    Y_bin = BinaryConverter.to_binary(np.clip(Y, 0, 1))
                    Z_bin = BinaryConverter.to_binary(np.clip(Z, 0, 1))
                    
                    # Assurer au moins une feature
                    if Y_bin.sum() == 0:
                        Y_bin[np.random.randint(self.n_features)] = 1
                    if Z_bin.sum() == 0:
                        Z_bin[np.random.randint(self.n_features)] = 1
                    
                    if self.evaluate_fitness(Y_bin) > self.evaluate_fitness(Z_bin):
                        X_new = Y
                    else:
                        X_new = Z
                
                else:  # r < 0.5 and abs(E) < 0.5
                    # Hard besiege with progressive rapid dives
                    X_rabbit = self.rabbit_position.astype(float)
                    X_m = np.mean(population.astype(float), axis=0)
                    Y = X_rabbit - E * np.abs(np.random.rand() * X_rabbit - X_m)
                    
                    # Levy flight
                    LF = self.levy_flight(self.n_features)
                    S = np.random.rand(self.n_features) * X_continuous
                    Z = Y + S * LF
                    
                    Y_bin = BinaryConverter.to_binary(np.clip(Y, 0, 1))
                    Z_bin = BinaryConverter.to_binary(np.clip(Z, 0, 1))
                    
                    if Y_bin.sum() == 0:
                        Y_bin[np.random.randint(self.n_features)] = 1
                    if Z_bin.sum() == 0:
                        Z_bin[np.random.randint(self.n_features)] = 1
                    
                    if self.evaluate_fitness(Y_bin) > self.evaluate_fitness(Z_bin):
                        X_new = Y
                    else:
                        X_new = Z
            
            # Convertir en binaire
            X_new_binary = BinaryConverter.to_binary(np.clip(X_new, 0, 1))
            
            # S'assurer qu'au moins une feature est sélectionnée
            if X_new_binary.sum() == 0:
                X_new_binary[np.random.randint(self.n_features)] = 1
            
            new_population.append(X_new_binary)
        
        return np.array(new_population)
    
    def levy_flight(self, dim):
        """
        Génère un vol de Lévy pour la recherche
        
        Args:
            dim: Dimension du vecteur
            
        Returns:
            Vecteur de vol de Lévy
        """
        beta = self.levy_beta
        
        # Calcul de sigma
        numerator = math.gamma(1 + beta) * np.sin(np.pi * beta / 2)
        denominator = math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)
        sigma = (numerator / denominator)**(1 / beta)
        
        # Génération du vol de Lévy
        u = np.random.randn(dim) * sigma
        v = np.random.randn(dim)
        step = u / np.abs(v)**(1 / beta)
        
        return self.levy_scale * step


# Test rapide
if __name__ == "__main__":
    print("✅ Module HHO chargé avec succès!")
    print("Classe disponible: HHO")