#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration Unifiée - Algorithmes de Base + Hybrides
Created on Fri Jan 16 17:29:18 2026
Updated on February 2026 - Ajout des hybrides séquentiels

@author: chamaelazrak
"""


class ExperimentConfig:
    """Configuration des expériences - Base + Hybrides"""
    
    # ========================================================================
    # DATASETS
    # ========================================================================
    DATASETS = {
        'parkinsons': 'data/raw/parkinsons.csv',
        'diabetes': 'data/raw/diabetes.csv',
        'heart': 'data/raw/heart.csv',
        'breast_cancer': 'data/raw/breast_cancer.csv'
    }
    
    # ========================================================================
    # PARAMÈTRES GÉNÉRAUX DES ALGORITHMES
    # ========================================================================
    ALGORITHM_PARAMS = {
        'pop_size': 15,      # Taille de la population
        'max_iter': 100,     # Nombre d'itérations maximum
        'n_runs': 15         # Nombre d'exécutions indépendantes
    }
    
    # ========================================================================
    # PARAMÈTRES ALGORITHMES DE BASE
    # ========================================================================
    
    # ACO (Ant Colony Optimization)
    ACO_PARAMS = {
        'alpha': 1.0,   # Importance phéromones
        'beta': 2.0,    # Importance heuristique
        'rho': 0.1,     # Évaporation
        'q0': 0.9       # Exploitation vs exploration
    }
    
    # PSO (Particle Swarm Optimization)
    PSO_PARAMS = {
        'w': 0.7,    # Inertia weight
        'c1': 1.5,   # Cognitive coefficient
        'c2': 1.5    # Social coefficient
    }
    
    # BSO (Brain Storm Optimization)
    BSO_PARAMS = {
        'n_clusters': 3,
        'p_replace': 0.2,
        'p_one': 0.8
    }
    
    # HHO (Harris Hawks Optimization)
    HHO_PARAMS = {
        'levy_beta': 1.5, 
        'levy_scale': 0.01
    }
    
    # bABER (Binary Al-Biruni Earth Radius)
    BABER_PARAMS = {
        'exploration_ratio_init': 0.7,
        'exploration_ratio_final': 0.3, 
        'h_range': [0, 2],
        'mutation_threshold': 3
    }
    
    # ========================================================================
    # PARAMÈTRES HYBRIDES SÉQUENTIELS
    # ========================================================================
    
    # Paramètre commun à tous les hybrides
    HYBRID_COMMON_PARAMS = {
        'transition_ratio': 0.6,  # 60% Phase 1, 40% Phase 2
    }
    
    # PSO-bABER
    # Utilise PSO_PARAMS pour Phase 1 et BABER_PARAMS pour Phase 2
    PSO_bABER_PARAMS = {
        'transition_ratio': 0.6,  # Hérité de HYBRID_COMMON_PARAMS
    }
    
    # HHO-bABER
    # Utilise HHO_PARAMS pour Phase 1 et BABER_PARAMS pour Phase 2
    HHO_bABER_PARAMS = {
        'transition_ratio': 0.6,  # Hérité de HYBRID_COMMON_PARAMS
    }
    
    # BSO-bABER
    # Utilise BSO_PARAMS pour Phase 1 et BABER_PARAMS pour Phase 2
    BSO_bABER_PARAMS = {
        'transition_ratio': 0.6,  # Hérité de HYBRID_COMMON_PARAMS
    }
    
    # ========================================================================
    # RANDOM FOREST CLASSIFIER
    # ========================================================================
    RF_PARAMS = {
        'n_estimators': 50,         # Nombre d'arbres
        'max_depth': 10,            # Profondeur maximale
        'min_samples_split': 2,     # Minimum d'échantillons pour split
        'min_samples_leaf': 1,      # Minimum d'échantillons par feuille
        'random_state': 42,         # Seed pour reproductibilité
        'n_jobs': -1                # Utiliser tous les CPU disponibles
    }
    
    # ========================================================================
    # FONCTION DE FITNESS
    # ========================================================================
    FITNESS_PARAMS = {
        'alpha': 0.99   # Poids pour l'accuracy
                        # fitness = alpha * accuracy - (1-alpha) * feature_ratio
                        # alpha = 0.99 : priorité à l'accuracy (99%)
                        # alpha = 0.50 : équilibre accuracy/réduction (50/50)
    }
    
    # ========================================================================
    # PRÉTRAITEMENT DES DONNÉES
    # ========================================================================
    PREPROCESSING_PARAMS = {
        'test_size': 0.3,               # Proportion pour le test set (30%)
        'random_state': 42,             # Seed pour reproductibilité
        'scaling_method': 'standard'    # 'standard' (StandardScaler) ou 'minmax' (MinMaxScaler)
    }
    
    # ========================================================================
    # VALIDATION CROISÉE
    # ========================================================================
    CV_PARAMS = {
        'n_splits': 5,      # Nombre de folds pour la validation croisée
        'shuffle': True,    # Mélanger les données
        'random_state': 42  # Seed pour reproductibilité
    }
    
    # ========================================================================
    # SAUVEGARDE ET LOGGING
    # ========================================================================
    OUTPUT_PARAMS = {
        'results_dir': 'results',
        'save_convergence': True,   # Sauvegarder les courbes de convergence
        'save_solutions': True,     # Sauvegarder les meilleures solutions
        'save_models': False,       # Sauvegarder les modèles RF entraînés
        'verbose': True             # Afficher les logs détaillés
    }
    
    # ========================================================================
    # MÉTHODES UTILITAIRES
    # ========================================================================
    
    @classmethod
    def get_algorithm_params(cls, algorithm_name: str) -> dict:
        """
        Récupère les paramètres spécifiques d'un algorithme (base ou hybride)
        
        Args:
            algorithm_name: Nom de l'algorithme 
                          ('ACO', 'PSO', 'BSO', 'HHO', 'bABER',
                           'PSO-bABER', 'HHO-bABER', 'BSO-bABER')
        
        Returns:
            dict: Paramètres de l'algorithme
        """
        params_map = {
            # Algorithmes de base
            'ACO': cls.ACO_PARAMS,
            'PSO': cls.PSO_PARAMS,
            'BSO': cls.BSO_PARAMS,
            'HHO': cls.HHO_PARAMS,
            'bABER': cls.BABER_PARAMS,
            # Hybrides séquentiels
            'PSO-bABER': cls.PSO_bABER_PARAMS,
            'HHO-bABER': cls.HHO_bABER_PARAMS,
            'BSO-bABER': cls.BSO_bABER_PARAMS,
        }
        return params_map.get(algorithm_name, {})
    
    @classmethod
    def print_config(cls):
        """Affiche la configuration actuelle"""
        print("\n" + "="*80)
        print(" "*30 + "CONFIGURATION")
        print("="*80)
        
        print(f"\n 📊 ALGORITHMES:")
        print(f"   - Taille de population: {cls.ALGORITHM_PARAMS['pop_size']}")
        print(f"   - Itérations max: {cls.ALGORITHM_PARAMS['max_iter']}")
        print(f"   - Runs par algorithme: {cls.ALGORITHM_PARAMS['n_runs']}")
        
        print(f"\n 🌳 RANDOM FOREST:")
        print(f"   - Nombre d'arbres: {cls.RF_PARAMS['n_estimators']}")
        print(f"   - Profondeur max: {cls.RF_PARAMS['max_depth']}")
        
        print(f"\n 🎯 FITNESS:")
        print(f"   - Alpha (accuracy): {cls.FITNESS_PARAMS['alpha']}")
        print(f"   - Beta (features): {1 - cls.FITNESS_PARAMS['alpha']}")
        
        print(f"\n 🔧 PRÉTRAITEMENT:")
        print(f"   - Test size: {cls.PREPROCESSING_PARAMS['test_size']*100}%")
        print(f"   - Scaling: {cls.PREPROCESSING_PARAMS['scaling_method']}")
        
        print(f"\n 💾 OUTPUT:")
        print(f"   - Répertoire: {cls.OUTPUT_PARAMS['results_dir']}/")
        print(f"   - Sauvegarder convergence: {cls.OUTPUT_PARAMS['save_convergence']}")
        
        print("="*80 + "\n")
    
    @classmethod
    def print_hybrid_config(cls):
        """Affiche la configuration spécifique des hybrides"""
        print("\n" + "="*80)
        print(" "*25 + "CONFIGURATION HYBRIDES")
        print("="*80)
        
        print(f"\n 🔀 STRATÉGIE: Hybridation séquentielle")
        print(f"   - Transfer point: {cls.HYBRID_COMMON_PARAMS['transition_ratio']*100:.0f}%")
        print(f"   - Méthode transfer: Direct Initialization")
        
        print(f"\n 🔵 PSO-bABER:")
        print(f"   - Phase 1 (60%): PSO (w={cls.PSO_PARAMS['w']}, c1={cls.PSO_PARAMS['c1']}, c2={cls.PSO_PARAMS['c2']})")
        print(f"   - Phase 2 (40%): bABER (exploration_init={cls.BABER_PARAMS['exploration_ratio_init']})")
        
        print(f"\n 🟢 HHO-bABER:")
        print(f"   - Phase 1 (60%): HHO (levy_beta={cls.HHO_PARAMS['levy_beta']})")
        print(f"   - Phase 2 (40%): bABER (exploration_init={cls.BABER_PARAMS['exploration_ratio_init']})")
        
        print(f"\n 🟠 BSO-bABER:")
        print(f"   - Phase 1 (60%): BSO (n_clusters={cls.BSO_PARAMS['n_clusters']}, p_replace={cls.BSO_PARAMS['p_replace']})")
        print(f"   - Phase 2 (40%): bABER (exploration_init={cls.BABER_PARAMS['exploration_ratio_init']})")
        
        print("="*80 + "\n")
    
    @classmethod
    def print_complete_config(cls):
        """Affiche la configuration complète (base + hybrides)"""
        cls.print_config()
        cls.print_hybrid_config()
    
    @classmethod
    def update_params(cls, **kwargs):
        """
        Met à jour les paramètres de configuration
        
        Usage:
            ExperimentConfig.update_params(
                pop_size=50,
                max_iter=200,
                n_runs=50
            )
        """
        for key, value in kwargs.items():
            if key in cls.ALGORITHM_PARAMS:
                cls.ALGORITHM_PARAMS[key] = value
            else:
                print(f"⚠️ Unknown Parameter : {key}")
    
    @classmethod
    def list_algorithms(cls):
        """Liste tous les algorithmes disponibles"""
        print("\n" + "="*80)
        print(" "*25 + "ALGORITHMS AVAILABLE")
        print("="*80)
        
        print("\n 🔹 BASIC ALGORITHMS (5):")
        base_algos = ['ACO', 'PSO', 'BSO', 'HHO', 'bABER']
        for algo in base_algos:
            print(f"   ✓ {algo}")
        
        print("\n 🔸 SEQUENTIEL HYBRIDS  (3):")
        hybrid_algos = ['PSO-bABER', 'HHO-bABER', 'BSO-bABER']
        for algo in hybrid_algos:
            print(f"   ✓ {algo}")
        
        print(f"\n 📊 TOTAL: {len(base_algos) + len(hybrid_algos)} algorithmes")
        print("="*80 + "\n")


# ============================================================================
# CONFIGURATIONS PRÉDÉFINIES
# ============================================================================

class QuickTestConfig(ExperimentConfig):
    """Configuration pour des tests rapides (debugging)"""
    ALGORITHM_PARAMS = {
        'pop_size': 10,
        'max_iter': 50,   # 30 Phase 1, 20 Phase 2 pour hybrides
        'n_runs': 3
    }


class StandardConfig(ExperimentConfig):
    """Configuration standard pour expériences normales (RECOMMANDÉ)"""
    ALGORITHM_PARAMS = {
        'pop_size': 15,
        'max_iter': 100,  # 60 Phase 1, 40 Phase 2 pour hybrides
        'n_runs': 15
    }


class IntensiveConfig(ExperimentConfig):
    """Configuration intensive pour résultats de publication"""
    ALGORITHM_PARAMS = {
        'pop_size': 50,
        'max_iter': 200,  # 120 Phase 1, 80 Phase 2 pour hybrides
        'n_runs': 50
    }


# ============================================================================
# EXEMPLE D'UTILISATION
# ============================================================================

if __name__ == "__main__":
    # Afficher la configuration complète
    print("\n" + "="*80)
    print(" "*20 + "DÉMONSTRATION DE LA CONFIGURATION")
    print("="*80)
    
    # 1. Configuration par défaut
    print("\n1️⃣  Configuration par défaut:")
    ExperimentConfig.print_config()
    
    # 2. Configuration hybrides
    print("\n2️⃣  Configuration hybrides:")
    ExperimentConfig.print_hybrid_config()
    
    # 3. Liste des algorithmes
    print("\n3️⃣  Liste des algorithmes:")
    ExperimentConfig.list_algorithms()
    
    # 4. Configuration standard (recommandée)
    print("\n4️⃣  Configuration STANDARD (recommandée pour publication):")
    StandardConfig.print_complete_config()
    
    # 5. Configuration test rapide
    print("\n5️⃣  Configuration TEST RAPIDE (debugging):")
    QuickTestConfig.print_config()
    
    # 6. Récupérer paramètres d'un algorithme
    print("\n6️⃣  Exemple récupération paramètres:")
    print("\n   PSO params:", ExperimentConfig.get_algorithm_params('PSO'))
    print("   PSO-bABER params:", ExperimentConfig.get_algorithm_params('PSO-bABER'))
    print("   HHO params:", ExperimentConfig.get_algorithm_params('HHO'))
    
    # 7. Modifier paramètres
    print("\n7️⃣  Modification des paramètres...")
    ExperimentConfig.update_params(
        pop_size=20,
        max_iter=150
    )
    print("   Nouvelle config:")
    print(f"   - Population: {ExperimentConfig.ALGORITHM_PARAMS['pop_size']}")
    print(f"   - Iterations: {ExperimentConfig.ALGORITHM_PARAMS['max_iter']}")
    
    print("\n" + "="*80)
    print("✅ Démonstration terminée!")
    print("="*80 + "\n")