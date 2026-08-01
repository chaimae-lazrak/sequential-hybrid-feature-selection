#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script d'Exécution Unifié - Algorithmes Base + Hybrides
Created on Fri Jan 16 17:30:14 2026
Updated on February 2026 - Intégration des hybrides séquentiels

Exécute 8 algorithmes sur 4 datasets médicaux:
- 5 Base: ACO, PSO, BSO, HHO, bABER
- 3 Hybrides: PSO-bABER, HHO-bABER, BSO-bABER

@author: chaimae lazrak
"""

from pathlib import Path

import numpy as np
import json
import pickle
from datetime import datetime
from time import time
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns

# Imports des modules du projet
from src.preprocessing.data_loader import DataLoader
from src.preprocessing.preprocessor import DataPreprocessor

from src.classifier.rf_classifier import RFClassifierWrapper

from src.utils.fitness import FitnessFunction

from src.evaluation.evaluator import AlgorithmEvaluator

# ============================================================================
# IMPORTS DES ALGORITHMES DE BASE
# ============================================================================
#from src.algorithms.aco import ACO
from src.algorithms.pso import PSO
from src.algorithms.bso import BSO
from src.algorithms.hho import HHO
from src.algorithms.baber import bABER

# ============================================================================
# IMPORTS DES HYBRIDES SÉQUENTIELS
# ============================================================================
from src.algorithms.pso_baber import PSO_bABER
from src.algorithms.hho_baber import HHO_bABER
from src.algorithms.bso_baber import BSO_bABER

# ============================================================================
# IMPORT DE LA CONFIGURATION
# ============================================================================
from experiments.config import ExperimentConfig, QuickTestConfig, StandardConfig, IntensiveConfig


class UnifiedExperimentRunner:
    """Gestionnaire d'expériences unifié - Base + Hybrides"""
    
    def __init__(self, config: ExperimentConfig, 
                 include_base: bool = True, 
                 include_hybrids: bool = True):
        """
        Initialise le runner
        
        Args:
            config: Configuration à utiliser
            include_base: Si True, exécute les 5 algorithmes de base
            include_hybrids: Si True, exécute les 3 hybrides
        """
        self.config = config
        self.include_base = include_base
        self.include_hybrids = include_hybrids
        self.results = {}
        self.setup_directories()
        
    def setup_directories(self):
        """Crée la structure de dossiers pour les résultats"""
        directories = [
            'results/metrics',
            'results/plots',
            'results/models',
            'results/logs',
            'results/hybrids'  # Dossier spécifique pour hybrides
        ]
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def load_datasets(self):
        """Charge tous les datasets"""
        print("\n" + "="*80)
        print("LOADING DATASETS")
        print("="*80)
        
        datasets = {}
        
        try:
            print("\n1. Parkinson's Disease...")
            X, y, features = DataLoader.load_parkinsons()
            datasets['parkinsons'] = (X, y, features)
            print(f"   ✓ Loaded: {X.shape[0]} samples, {X.shape[1]} features")
        except Exception as e:
            print(f"   ✗ Error: {e}")
        
        try:
            print("\n2. Diabetes (PIMA)...")
            X, y, features = DataLoader.load_diabetes()
            datasets['diabetes'] = (X, y, features)
            print(f"   ✓ Loaded: {X.shape[0]} samples, {X.shape[1]} features")
        except Exception as e:
            print(f"   ✗ Error: {e}")
        
        try:
            print("\n3. Heart Disease...")
            X, y, features = DataLoader.load_heart()
            datasets['heart'] = (X, y, features)
            print(f"   ✓ Loaded: {X.shape[0]} samples, {X.shape[1]} features")
        except Exception as e:
            print(f"   ✗ Error: {e}")
        
        try:
            print("\n4. Breast Cancer Wisconsin...")
            X, y, features = DataLoader.load_breast_cancer()
            datasets['breast_cancer'] = (X, y, features)
            print(f"   ✓ Loaded: {X.shape[0]} samples, {X.shape[1]} features")
        except Exception as e:
            print(f"   ✗ Error: {e}")
        
        try:
            print("\n5. Cervical Cancer (Risk Factors)...")
            X, y, features = DataLoader.load_cervical_cancer()
            datasets['cervical_cancer'] = (X, y, features)
            print(f"   ✓ Loaded: {X.shape[0]} samples, {X.shape[1]} features")
        except Exception as e:
            print(f"   ✗ Error: {e}")   
        
        try:
            print("\n6. Alzheimer's Disease...")
            X, y, features = DataLoader.load_alzheimers()
            datasets['alzheimers'] = (X, y, features)
            print(f"   ✓ Loaded: {X.shape[0]} samples, {X.shape[1]} features")
        except Exception as e:
            print(f"   ✗ Error: {e}")
            

        
        return datasets
    
    def run_single_experiment(self, algorithm_class, algorithm_name,
                             dataset_name, X, y,
                             run_number, **algo_params):
        """Exécute une seule expérience (un algorithme sur un dataset)"""
        
        # Prétraitement
        preprocessor = DataPreprocessor(
            scaling_method=self.config.PREPROCESSING_PARAMS['scaling_method']
        )
        data = preprocessor.preprocess(
            X, y,
            test_size=self.config.PREPROCESSING_PARAMS['test_size'],
            random_state=self.config.PREPROCESSING_PARAMS['random_state'] + run_number
        )
        
        # Classifier
        classifier = RFClassifierWrapper(**self.config.RF_PARAMS)
        
        # Fitness SANS test set - utilise cross_validate()
        fitness = FitnessFunction(
            data['X_train'], 
            data['y_train'],
            classifier,
            alpha=self.config.FITNESS_PARAMS['alpha'],
            cv=self.config.CV_PARAMS['n_splits']
        )
        
        # Initialisation de l'algorithme
        algorithm = algorithm_class(
            n_features=X.shape[1],
            pop_size=self.config.ALGORITHM_PARAMS['pop_size'],
            max_iter=self.config.ALGORITHM_PARAMS['max_iter'],
            fitness_func=fitness,
            config=self.config,
            **algo_params
        )
        
        # Exécution (test set jamais vu)
        print(f"      Run {run_number + 1}/{self.config.ALGORITHM_PARAMS['n_runs']}...", end=" ")
        #Mesurer le temps d'execution
        start_time = time()
        best_solution, best_fitness, convergence = algorithm.run()
        end_time = time()
        execution_time=end_time-start_time
        # Évaluation finale sur test set (APRÈS optimisation)
        metrics = fitness.evaluate_on_test(
            best_solution,
            data['X_test'],
            data['y_test']
        )
        metrics["execution_time"]=execution_time
        
        print(f"Accuracy: {metrics['accuracy']:.4f}, Features: {metrics['n_features']}, Time: {execution_time:.2f}s")
        
        return {
            'best_solution': best_solution,
            'best_fitness': best_fitness,
            'convergence': convergence,
            'metrics': metrics,
            'run_number': run_number
        }
    
    def run_algorithm_on_dataset(self, algorithm_class, algorithm_name,
                                dataset_name, X, y,
                                feature_names, **algo_params):
        """Exécute un algorithme plusieurs fois sur un dataset"""
        
        # Identifier type d'algorithme
        algo_type = "HYBRIDE" if '-' in algorithm_name else "BASE"
        print(f"\n   → {algorithm_name} ({algo_type})...")
        
        run_results = []
        
        for run in range(self.config.ALGORITHM_PARAMS['n_runs']):
            try:
                result = self.run_single_experiment(
                    algorithm_class, algorithm_name,
                    dataset_name, X, y, run,
                    **algo_params
                )
                run_results.append(result)
            except Exception as e:
                print(f"      ✗ Run {run + 1} failed: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Agrégation des résultats
        if len(run_results) == 0:
            print(f"      ✗ All runs failed!")
            return None
        
        accuracies = [r['metrics']['accuracy'] for r in run_results]
        precisions = [r['metrics']['precision'] for r in run_results]
        recalls = [r['metrics']['recall'] for r in run_results]
        f1_scores = [r['metrics']['f1'] for r in run_results]
        n_features = [r['metrics']['n_features'] for r in run_results]
        
        execution_times = [r['metrics']['execution_time'] for r in run_results]
        
        # Extraction correcte des features sélectionnées
        best_run_idx = np.argmax(accuracies)
        best_solution = run_results[best_run_idx]['best_solution']
        selected_indices = np.where(best_solution == 1)[0]
        
        if isinstance(feature_names, (list, np.ndarray)):
            try:
                selected_features = [feature_names[i] for i in selected_indices]
            except:
                selected_features = selected_indices.tolist()
        else:
            selected_features = selected_indices.tolist()
        
        aggregated_results = {
            'algorithm_type': algo_type,
            'accuracy_mean': np.mean(accuracies),
            'accuracy_std': np.std(accuracies),
            'accuracy_min': np.min(accuracies),
            'accuracy_max': np.max(accuracies),
            'accuracy_median': np.median(accuracies),
            'precision_mean': np.mean(precisions),
            'precision_std': np.std(precisions),
            'recall_mean': np.mean(recalls),
            'recall_std': np.std(recalls),
            'f1_mean': np.mean(f1_scores),
            'f1_std': np.std(f1_scores),
            'n_features_mean': np.mean(n_features),
            'n_features_std': np.std(n_features),
            'n_features_min': np.min(n_features),
            'n_features_max': np.max(n_features),
            'execution_time_mean': np.mean(execution_times),
            'execution_time_std': np.std(execution_times),
            'execution_time_min': np.min(execution_times),
            'execution_time_max': np.max(execution_times),
            'all_accuracies': accuracies,
            'all_n_features': n_features,
            'all_execution_times': execution_times,
            'best_run': run_results[best_run_idx],
            'selected_features': selected_features,
            'n_successful_runs': len(run_results),
            'n_total_runs': self.config.ALGORITHM_PARAMS['n_runs']
        }
        
        print(f"      Average: {aggregated_results['accuracy_mean']:.4f} ± {aggregated_results['accuracy_std']:.4f}")
        
        return aggregated_results
    
    def run_all_experiments(self):
        """Exécute toutes les expériences (base et/ou hybrides)"""
        
        print("\n" + "="*80)
        print("BEGINNING OF THE EXPERIMENTS - UNIFIED MODE")
        print("="*80)
        print(f"Paramètres:")
        print(f"  - Population: {self.config.ALGORITHM_PARAMS['pop_size']}")
        print(f"  - Iterations: {self.config.ALGORITHM_PARAMS['max_iter']}")
        print(f"  - Runs by algorithm: {self.config.ALGORITHM_PARAMS['n_runs']}")
        print(f"  - Include Base: {self.include_base} (5 algorithmes)")
        print(f"  - Include Hybrides: {self.include_hybrids} (3 algorithmes)")
        
        if self.include_hybrids:
            print(f"  - Transition hybrides: {self.config.HYBRID_COMMON_PARAMS['transition_ratio']*100:.0f}%")
        
        # Chargement des datasets
        datasets = self.load_datasets()
        
        if len(datasets) == 0:
            print("\n✗ No dataset Loaded. Stop of experiments.")
            return
        
        # ====================================================================
        # DÉFINITION DES ALGORITHMES À TESTER
        # ====================================================================
        algorithms = {}
        
        # ALGORITHMES DE BASE
        if self.include_base:
            algorithms.update({
                #'ACO': (ACO, self.config.ACO_PARAMS),
                'PSO': (PSO, self.config.PSO_PARAMS),
                'BSO': (BSO, self.config.BSO_PARAMS),
                'HHO': (HHO, self.config.HHO_PARAMS),
                'bABER': (bABER, self.config.BABER_PARAMS),
            })
        
        # HYBRIDES SÉQUENTIELS
        if self.include_hybrids:
            algorithms.update({
                'PSO-bABER': (PSO_bABER, self.config.PSO_bABER_PARAMS),
                'HHO-bABER': (HHO_bABER, self.config.HHO_bABER_PARAMS),
                'BSO-bABER': (BSO_bABER, self.config.BSO_bABER_PARAMS),
            })
        
        print(f"\n📊 Algorithms to test ({len(algorithms)}):")
        base_algos = [k for k in algorithms.keys() if '-' not in k]
        hybrid_algos = [k for k in algorithms.keys() if '-' in k]
        if base_algos:
            print(f"   BASE: {', '.join(base_algos)}")
        if hybrid_algos:
            print(f"   HYBRIDES: {', '.join(hybrid_algos)}")
        
        # ====================================================================
        # BOUCLE SUR LES DATASETS
        # ====================================================================
        for dataset_name, (X, y, feature_names) in datasets.items():
            print("\n" + "="*80)
            print(f"DATASET: {dataset_name.upper()}")
            print("="*80)
            print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}, Classes: {len(np.unique(y))}")
            
            self.results[dataset_name] = {}
            
            # Boucle sur les algorithmes
            for algo_name, (algo_class, algo_params) in algorithms.items():
                result = self.run_algorithm_on_dataset(
                    algo_class, algo_name,
                    dataset_name, X, y, 
                    feature_names,
                    **algo_params
                )
                
                if result is not None:
                    self.results[dataset_name][algo_name] = result
        
        # Sauvegarde et génération du rapport
        self.save_results()
        self.generate_comparison_report()
        self.generate_report()
        
        # Si hybrides inclus, générer rapport spécifique
        if self.include_hybrids and self.include_base:
            self.generate_hybrid_vs_base_report()
    
    def save_results(self):
        """Sauvegarde les résultats"""
        
        print("\n" + "="*80)
        print("SAVING OF RESULTS")
        print("="*80)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Déterminer préfixe selon mode
        if self.include_base and self.include_hybrids:
            prefix = "unified"
        elif self.include_hybrids:
            prefix = "hybrids_only"
        else:
            prefix = "base_only"
        
        # 1. JSON (résultats numériques)
        print("\n1. Fichier JSON...")
        json_results = {}
        for dataset, algos in self.results.items():
            json_results[dataset] = {}
            for algo, metrics in algos.items():
                json_results[dataset][algo] = {
                    'type': metrics.get('algorithm_type', 'BASE'),
                    'accuracy_mean': float(metrics['accuracy_mean']),
                    'accuracy_std': float(metrics['accuracy_std']),
                    'precision_mean': float(metrics['precision_mean']),
                    'recall_mean': float(metrics['recall_mean']),
                    'f1_mean': float(metrics['f1_mean']),
                    'n_features_mean': float(metrics['n_features_mean']),
                    'n_features_std': float(metrics['n_features_std']),
                    'selected_features': metrics.get('selected_features', [])
                }
        
        json_path = f"results/metrics/results_{prefix}_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"   ✓ {json_path}")
        
        # 2. Pickle (résultats complets avec convergence)
        print("\n2. Complet Pickle File...")
        pickle_path = f"results/metrics/results_{prefix}_full_{timestamp}.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(self.results, f)
        print(f"   ✓ {pickle_path}")
        
        # 3. CSV (pour Excel)
        print("\n3. Fichier CSV...")
        import pandas as pd
        
        data = []
        for dataset, algos in self.results.items():
            for algo, metrics in algos.items():
                data.append({
                    'Dataset': dataset,
                    'Algorithm': algo,
                    'Type': metrics.get('algorithm_type', 'BASE'),
                    'Accuracy_Mean': metrics['accuracy_mean'],
                    'Accuracy_Std': metrics['accuracy_std'],
                    'Precision_Mean': metrics['precision_mean'],
                    'Recall_Mean': metrics['recall_mean'],
                    'F1_Mean': metrics['f1_mean'],
                    'Features_Mean': metrics['n_features_mean'],
                    'Features_Std': metrics['n_features_std'],
                    #ajouter colonne pour xecution time
                    'Execution_Time_Mean': metrics.get('execution_time_mean', 0),
                    'Execution_Time_Std': metrics.get('execution_time_std', 0)
                })
        
        df = pd.DataFrame(data)
        csv_path = f"results/metrics/results_{prefix}_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        print(f"   ✓ {csv_path}")
        
        print("\n✅ All results saved!")
    
    def generate_comparison_report(self):
        """Génère un rapport de comparaison global"""
        
        print("\n" + "="*80)
        print("GLOBAL COMPARISON REPORT")
        print("="*80)
        
        for dataset_name, algos in self.results.items():
            print(f"\n{'='*80}")
            print(f"📊 {dataset_name.upper()}")
            print(f"{'='*80}")
            
            # Trier par accuracy
            sorted_algos = sorted(algos.items(), 
                                key=lambda x: x[1]['accuracy_mean'], 
                                reverse=True)
            
            # Afficher top 5
            print("\n🏆 Top 5:")
            for i, (algo, metrics) in enumerate(sorted_algos[:5], 1):
                algo_type = metrics.get('algorithm_type', 'BASE')
                emoji = "🔸" if algo_type == "HYBRIDE" else "🔹"
                print(f"   {i}. {emoji} {algo:15s} ({algo_type:8s}): "
                      f"{metrics['accuracy_mean']:.4f} ± {metrics['accuracy_std']:.4f} | "
                      f"Features: {metrics['n_features_mean']:.1f}| "
                      f"Time: {metrics.get('execution_time_mean', 0):.2f}s")
    
    def generate_hybrid_vs_base_report(self):
        """Génère un rapport spécifique Hybrides vs Base"""
        
        print("\n" + "="*80)
        print("HYBRIDES vs BASE REPORT ")
        print("="*80)
        
        for dataset_name, algos in self.results.items():
            print(f"\n{'='*80}")
            print(f"📊 {dataset_name.upper()}")
            print(f"{'='*80}")
            
            # Séparer base et hybrides
            base_algos = {k: v for k, v in algos.items() 
                         if v.get('algorithm_type') == 'BASE'}
            hybrid_algos = {k: v for k, v in algos.items() 
                           if v.get('algorithm_type') == 'HYBRIDE'}
            
            if base_algos and hybrid_algos:
                # Meilleur de chaque catégorie
                best_base = max(base_algos.items(), 
                              key=lambda x: x[1]['accuracy_mean'])
                best_hybrid = max(hybrid_algos.items(), 
                                key=lambda x: x[1]['accuracy_mean'])
                
                print(f"\n   🔹 Best BASE:")
                print(f"      {best_base[0]:15s}: {best_base[1]['accuracy_mean']:.4f} ± {best_base[1]['accuracy_std']:.4f}")
                
                print(f"\n   🔸 Best HYBRIDE:")
                print(f"      {best_hybrid[0]:15s}: {best_hybrid[1]['accuracy_mean']:.4f} ± {best_hybrid[1]['accuracy_std']:.4f}")
                
                # Gain
                gain = (best_hybrid[1]['accuracy_mean'] - best_base[1]['accuracy_mean']) * 100
                print(f"\n   📈 GAIN: {gain:+.2f}% ({'✅ Positif' if gain > 0 else '⚠️ Négatif'})")
    
    def generate_report(self):
        """Génère le rapport complet avec visualisations"""
        
        print("\n" + "="*80)
        print("Report Generation with Visualisation")
        print("="*80)
        
        try:
            # Créer l'évaluateur
            evaluator = AlgorithmEvaluator(self.results)
            
            # Générer le rapport complet
            evaluator.generate_full_report("results")
            
            print("\n✅ Full Visualization report generated!")
        except Exception as e:
            print(f"\n⚠️ Error during the generation of visualizations: {e}")
            print("   Successfully saved digital results.")


# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def run_quick_test(include_base=True, include_hybrids=True):
    """Exécute un test rapide pour vérifier que tout fonctionne"""
    
    print("\n" + "="*80)
    print(" "*30 + "MODE QUICK TEST")
    print("="*80)
    
    runner = UnifiedExperimentRunner(
        QuickTestConfig, 
        include_base=include_base,
        include_hybrids=include_hybrids
    )
    runner.run_all_experiments()


def run_standard_experiments(include_base=True, include_hybrids=True):
    """Exécute les expériences avec la configuration standard"""
    
    print("\n" + "="*80)
    print(" "*25 + "STANDARD EXPERIMENTS")
    print("="*80)
    
    runner = UnifiedExperimentRunner(
        StandardConfig,
        include_base=include_base,
        include_hybrids=include_hybrids
    )
    runner.run_all_experiments()


def run_intensive_experiments(include_base=True, include_hybrids=True):
    """Exécute les expériences avec la configuration intensive (publication)"""
    
    print("\n" + "="*80)
    print(" "*20 + "INTENSIVES EXPERIMENTS (PUBLICATION)")
    print("="*80)
    
    runner = UnifiedExperimentRunner(
        IntensiveConfig,
        include_base=include_base,
        include_hybrids=include_hybrids
    )
    runner.run_all_experiments()


# ============================================================================
# POINT D'ENTRÉE PRINCIPAL
# ============================================================================

def main():
    """Point d'entrée principal du programme"""
    
    import argparse
    
    # Parser d'arguments en ligne de commande
    parser = argparse.ArgumentParser(
        description='Exécution unifiée des expériences - Base + Hybrides'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['test', 'standard', 'intensive'],
        default='standard',
        help='Mode d\'exécution (test/standard/intensive)'
    )
    parser.add_argument(
        '--base-only',
        action='store_true',
        help='Exécuter seulement les algorithmes de base'
    )
    parser.add_argument(
        '--hybrids-only',
        action='store_true',
        help='Exécuter seulement les hybrides'
    )
    parser.add_argument(
        '--pop-size',
        type=int,
        help='Taille de la population (override config)'
    )
    parser.add_argument(
        '--max-iter',
        type=int,
        help='Nombre max d\'itérations (override config)'
    )
    parser.add_argument(
        '--n-runs',
        type=int,
        help='Nombre de runs (override config)'
    )
    
    args = parser.parse_args()
    
    # Déterminer quels algorithmes exécuter
    include_base = not args.hybrids_only
    include_hybrids = not args.base_only
    
    # Sélection du mode
    if args.mode == 'test':
        print("\n🚀 Lancement en mode TEST (rapide)...")
        run_quick_test(include_base, include_hybrids)
    elif args.mode == 'intensive':
        print("\n🚀 Lancement en mode INTENSIF (publication)...")
        run_intensive_experiments(include_base, include_hybrids)
    else:
        print("\n🚀 Lancement en mode STANDARD...")
        run_standard_experiments(include_base, include_hybrids)
    
    print("\n" + "="*80)
    print(" "*30 + "TERMINÉ!")
    print("="*80)
    print("\n📁 Results available in:")
    print("   - results/metrics/     : Fichiers JSON, CSV, Pickle")
    print("   - results/plots/       : Graphiques et visualisations")
    print("   - results/logs/        : Logs d'exécution")
    if include_hybrids:
        print("   - results/hybrids/     : Résultats hybrides spécifiques")
    print("\n")


if __name__ == "__main__":
    main()