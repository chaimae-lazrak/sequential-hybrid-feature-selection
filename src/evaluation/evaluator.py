#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 12:39:07 2026

@author: chamaelazrak
"""

"""
Module d'évaluation et de comparaison des algorithmes
src/evaluation/evaluator.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict
import json


class AlgorithmEvaluator:
    """Évaluation complète et comparaison des algorithmes"""
    
    def __init__(self, results: Dict):
        """
        Initialise l'évaluateur
        
        Args:
            results: Dictionnaire des résultats
                {
                    'dataset1': {
                        'algo1': {'accuracy_mean': x, 'all_accuracies': [...], ...},
                        'algo2': {...},
                    },
                    'dataset2': {...}
                }
        """
        self.results = results
        
    def create_comparison_table(self) -> pd.DataFrame:
        """
        Crée un tableau de comparaison des résultats
        
        Returns:
            DataFrame avec tous les résultats
        """
        data = []
        
        for dataset, algos in self.results.items():
            for algo, metrics in algos.items():
                data.append({
                    'Dataset': dataset,
                    'Algorithm': algo,
                    'Accuracy_Mean': metrics.get('accuracy_mean', 0),
                    'Accuracy_Std': metrics.get('accuracy_std', 0),
                    'Accuracy_Min': metrics.get('accuracy_min', 0),
                    'Accuracy_Max': metrics.get('accuracy_max', 0),
                    'Precision_Mean': metrics.get('precision_mean', 0),
                    'Recall_Mean': metrics.get('recall_mean', 0),
                    'F1_Mean': metrics.get('f1_mean', 0),
                    'Features_Mean': metrics.get('n_features_mean', 0),
                    'Features_Std': metrics.get('n_features_std', 0),
                    'Execution_Time_Mean': metrics.get('execution_time_mean', 0),
                    'Execution_Time_Std': metrics.get('execution_time_std', 0),
                    'Successful_Runs': metrics.get('n_successful_runs', 0)
                })
        
        df = pd.DataFrame(data)
        return df
    
    def rank_algorithms(self) -> pd.DataFrame:
        """
        Classe les algorithmes par dataset
        
        Returns:
            DataFrame avec les rankings
        """
        rankings = []
        
        for dataset, algos in self.results.items():
            accuracies = {algo: metrics.get('accuracy_mean', 0) 
                         for algo, metrics in algos.items()}
            
            sorted_algos = sorted(accuracies.items(), 
                                key=lambda x: x[1], reverse=True)
            
            for rank, (algo, acc) in enumerate(sorted_algos, 1):
                rankings.append({
                    'Dataset': dataset,
                    'Rank': rank,
                    'Algorithm': algo,
                    'Accuracy': acc
                })
        
        return pd.DataFrame(rankings)
    
    def plot_accuracy_comparison(self, save_path: str = None):
        """
        Graphique de comparaison des accuracy
        
        Args:
            save_path: Chemin de sauvegarde (optionnel)
        """
        n_datasets = len(self.results)
        if n_datasets == 0:
            print("⚠️  Pas de résultats à visualiser")
            return
        
        # Créer la figure avec sous-graphiques
        n_rows = 2
        n_cols = 3
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 12))
        axes = axes.flatten()
        
        for idx, (dataset, algos) in enumerate(self.results.items()):
            if idx >= len(axes):
                break
            
            ax = axes[idx]
            
            algo_names = list(algos.keys())
            means = [algos[a].get('accuracy_mean', 0) for a in algo_names]
            stds = [algos[a].get('accuracy_std', 0) for a in algo_names]
            
            # Couleurs
            colors = plt.cm.Set3(np.linspace(0, 1, len(algo_names)))
            
            # Barplot
            bars = ax.bar(algo_names, means, yerr=stds, 
                         capsize=5, alpha=0.8, color=colors, 
                         edgecolor='black', linewidth=1.5)
            
            # Ajouter les valeurs sur les barres
            for bar, mean in zip(bars, means):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{mean:.4f}',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            ax.set_title(f"{dataset.upper()}", fontsize=14, fontweight='bold')
            ax.set_ylabel("Accuracy", fontsize=12)
            ax.set_ylim([max(0, min(means) - 0.05), 1.0])
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.tick_params(axis='x', rotation=45)
        
        # Masquer les axes non utilisés
        for idx in range(len(self.results), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Graphique sauvegardé: {save_path}")
        
        plt.close()
    
    def plot_convergence_curves(self, save_path: str = None):
        """
        Graphiques de convergence
        
        Args:
            save_path: Chemin de sauvegarde (optionnel)
        """
        n_datasets = len(self.results)
        if n_datasets == 0:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 12))
        axes = axes.flatten()
        
        for idx, (dataset, algos) in enumerate(self.results.items()):
            if idx >= len(axes):
                break
            
            ax = axes[idx]
            
            for algo, metrics in algos.items():
                if 'best_run' in metrics and 'convergence' in metrics['best_run']:
                    convergence = metrics['best_run']['convergence']
                    ax.plot(convergence, label=algo, linewidth=2, 
                           marker='o', markevery=max(1, len(convergence)//10))
            
            ax.set_title(f"Convergence - {dataset.upper()}", 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel("Iteration", fontsize=12)
            ax.set_ylabel("Fitness", fontsize=12)
            ax.legend(fontsize=10, loc='lower right')
            ax.grid(True, alpha=0.3)
        
        # Masquer les axes non utilisés
        for idx in range(len(self.results), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Graphique sauvegardé: {save_path}")
        
        plt.close()
    
    def plot_boxplot(self, save_path: str = None):
        """
        Boxplots des distributions d'accuracy
        
        Args:
            save_path: Chemin de sauvegarde (optionnel)
        """
        n_datasets = len(self.results)
        if n_datasets == 0:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 12))
        axes = axes.flatten()
        
        for idx, (dataset, algos) in enumerate(self.results.items()):
            if idx >= len(axes):
                break
            
            ax = axes[idx]
            
            data = []
            labels = []
            
            for algo, metrics in algos.items():
                if 'all_accuracies' in metrics:
                    data.append(metrics['all_accuracies'])
                    labels.append(algo)
            
            if len(data) > 0:
                bp = ax.boxplot(data, labels=labels, patch_artist=True,
                              showmeans=True, meanline=True)
                
                # Coloration
                colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
            
            ax.set_title(f"Distribution - {dataset.upper()}", 
                        fontsize=14, fontweight='bold')
            ax.set_ylabel("Accuracy", fontsize=12)
            ax.grid(axis='y', alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
        
        # Masquer les axes non utilisés
        for idx in range(len(self.results), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Graphique sauvegardé: {save_path}")
        
        plt.close()
    
    def plot_feature_reduction(self, save_path: str = None):
        """
        Graphique de réduction de features
        
        Args:
            save_path: Chemin de sauvegarde (optionnel)
        """
        fig, ax = plt.subplots(figsize=(14, 8))
        
        datasets = list(self.results.keys())
        if len(datasets) == 0:
            return
        
        algorithms = list(next(iter(self.results.values())).keys())
        
        x = np.arange(len(datasets))
        width = 0.8 / len(algorithms)
        
        for i, algo in enumerate(algorithms):
            features = [self.results[d][algo].get('n_features_mean', 0) 
                       for d in datasets]
            offset = (i - len(algorithms)/2) * width + width/2
            ax.bar(x + offset, features, width, label=algo, alpha=0.8)
        
        ax.set_xlabel('Dataset', fontsize=12)
        ax.set_ylabel('Number of features selected', fontsize=12)
        ax.set_title('Reduction of features by Algorithm', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([d.upper() for d in datasets])
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved graphic: {save_path}")
        
        plt.close()
    
    def generate_latex_table(self) -> str:
        """
        Génère un tableau LaTeX pour publication
        
        Returns:
            String LaTeX
        """
        df = self.create_comparison_table()
        
        latex = "\\begin{table}[htbp]\n"
        latex += "\\centering\n"
        latex += "\\caption{Comparison of feature selection algorithms}\n"
        latex += "\\label{tab:comparison}\n"
        latex += "\\begin{tabular}{llcccc}\n"
        latex += "\\hline\n"
        latex += "Dataset & Algorithm & Accuracy & Std & Features & Std \\\\\n"
        latex += "\\hline\n"
        
        for dataset in df['Dataset'].unique():
            subset = df[df['Dataset'] == dataset]
            for idx, row in subset.iterrows():
                latex += f"{row['Dataset']} & {row['Algorithm']} & "
                latex += f"{row['Accuracy_Mean']:.4f} & "
                latex += f"{row['Accuracy_Std']:.4f} & "
                latex += f"{row['Features_Mean']:.1f} & "
                latex += f"{row['Features_Std']:.1f} \\\\\n"
                
                latex += f"{row.get('Execution_Time_Mean', 0):.2f} & "  
                latex += f"{row.get('Execution_Time_Std', 0):.2f} \\\\\n"  
            latex += "\\hline\n"
        
        latex += "\\end{tabular}\n"
        latex += "\\end{table}\n"
        
        return latex

    def plot_execution_time_comparison(self, save_path: str = None):
        """Graphique comparaison temps d'exécution"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        datasets = list(self.results.keys())
        algorithms = list(next(iter(self.results.values())).keys())
        
        x = np.arange(len(datasets))
        width = 0.8 / len(algorithms)
        
        for i, algo in enumerate(algorithms):
            times = [self.results[d][algo].get('execution_time_mean', 0) 
                    for d in datasets]
            stds = [self.results[d][algo].get('execution_time_std', 0)
                   for d in datasets]
            offset = (i - len(algorithms)/2) * width + width/2
            ax.bar(x + offset, times, width, label=algo, alpha=0.8, 
                  yerr=stds, capsize=5)
        
        ax.set_xlabel('Dataset', fontsize=12)
        ax.set_ylabel('Execution Time  (secondes)', fontsize=12)
        ax.set_title('Comparison Execution Time', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([d.upper() for d in datasets], rotation=45)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved Graphic: {save_path}")
        
        plt.close()    
    
    def generate_full_report(self, output_dir: str = "results"):
        """
        Génère un rapport complet
        
        Args:
            output_dir: Répertoire de sortie
        """
        Path(output_dir).mkdir(exist_ok=True)
        Path(f"{output_dir}/plots").mkdir(exist_ok=True)
        
        print("\n" + "="*70)
        print(" "*20 + "REPORT GENERATION")
        print("="*70)
        
        # 1. Tableau de comparaison
        print("\n1. Comparison Table...")
        df = self.create_comparison_table()
        df.to_csv(f"{output_dir}/comparison_table.csv", index=False)
        print(f"   ✓ Saved: {output_dir}/comparison_table.csv")
        
        # 2. Rankings
        print("\n2. Rankings...")
        rankings = self.rank_algorithms()
        rankings.to_csv(f"{output_dir}/rankings.csv", index=False)
        print(f"   ✓ Saved: {output_dir}/rankings.csv")
        
        # 3. Graphiques
        print("\n3. Graphics Generation ...")
        self.plot_accuracy_comparison(f"{output_dir}/plots/accuracy_comparison.png")
        self.plot_convergence_curves(f"{output_dir}/plots/convergence.png")
        self.plot_boxplot(f"{output_dir}/plots/boxplot.png")
        self.plot_feature_reduction(f"{output_dir}/plots/feature_reduction.png")
        self.plot_execution_time_comparison(f"{output_dir}/plots/execution_time.png")
        
        # 4. Tableau LaTeX
        print("\n4. LaTeX Table...")
        latex = self.generate_latex_table()
        with open(f"{output_dir}/latex_table.tex", 'w') as f:
            f.write(latex)
        print(f"   ✓ Saved: {output_dir}/latex_table.tex")
        
        # 5. Afficher le tableau dans la console
        print("\n5. Comparison Table :")
        print("="*70)
        print(df.to_string(index=False))
        
        # 6. Afficher le ranking
        print("\n6. Ranking by dataset:")
        print("="*70)
        for dataset in df['Dataset'].unique():
            print(f"\n{dataset.upper()}:")
            subset = rankings[rankings['Dataset'] == dataset]
            for _, row in subset.iterrows():
                print(f"  {row['Rank']}. {row['Algorithm']:10s} - {row['Accuracy']:.4f}")
                
        
        
        print("\n" + "="*70)
        print(" "*15 + "Full REPORT GENERATED!")
        print("="*70)
        
    # Ajouter graphe de temps d'execution


# Test rapide
if __name__ == "__main__":
    print(" Module AlgorithmEvaluator loaed successfully!")