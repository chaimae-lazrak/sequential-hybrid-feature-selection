#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module de tests statistiques pour validation scientifique
src/evaluation/statistical_tests.py

Tests implémentés:
- Wilcoxon signed-rank test (comparaisons pairées)
- Friedman test (comparaisons multiples)
- Nemenyi post-hoc test
- Cohen's d (effect size)

Auteur: Chaimae Lazrak
Date: Février 2026
"""

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare, rankdata
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class StatisticalAnalyzer:
    """
    Analyse statistique complète pour validation scientifique
    """
    
    def __init__(self, results_csv_path: str, alpha: float = 0.05):
        """
        Initialise l'analyseur statistique
        
        Args:
            results_csv_path: Chemin vers le CSV de résultats
                             (généré par run_experiments.py)
            alpha: Niveau de signification (défaut: 0.05)
        """
        self.alpha = alpha
        self.df = pd.read_csv(results_csv_path)
        
        # Séparer base et hybrides
        self.base_algos = self.df[self.df['Type'] == 'BASE']
        self.hybrid_algos = self.df[self.df['Type'] == 'HYBRIDE']
        
        print(f"✅ Data loaded: {results_csv_path}")
        print(f"   Datasets: {self.df['Dataset'].nunique()}")
        print(f"   BASE Algorithms : {len(self.base_algos['Algorithm'].unique())}")
        print(f"   HYBRIDES Algorithms: {len(self.hybrid_algos['Algorithm'].unique())}")
        print(f"   level α: {self.alpha}")
    
    
    def wilcoxon_pairwise_tests(self) -> pd.DataFrame:
        """
        Tests de Wilcoxon pour chaque paire base-hybride
        
        Returns:
            DataFrame avec résultats des tests
        """
        print("\n" + "="*80)
        print("1️⃣  WILCOXON SIGNED-RANK TESTS (Pairwise Comparisons)")
        print("="*80)
        
        results = []
        
        for hybrid_name in self.hybrid_algos['Algorithm'].unique():
            base_name = hybrid_name.replace('-bABER', '')
            
            # Extraire données
            hybrid_data = self.hybrid_algos[
                self.hybrid_algos['Algorithm'] == hybrid_name
            ]
            base_data = self.base_algos[
                self.base_algos['Algorithm'] == base_name
            ]
            
            if base_data.empty:
                print(f"⚠️  {hybrid_name}: No maching base ({base_name})")
                continue
            
            # Accuracies moyennes par dataset
            hybrid_accs = hybrid_data['Accuracy_Mean'].values
            base_accs = base_data['Accuracy_Mean'].values
            
            # Vérifier même nombre de datasets
            if len(hybrid_accs) != len(base_accs):
                print(f"⚠️  {hybrid_name}: Different number of datasets ")
                continue
            
            # Test de Wilcoxon (unilatéral: hybride > base)
            try:
                statistic, p_value = wilcoxon(
                    hybrid_accs, 
                    base_accs, 
                    alternative='greater'
                )
                
                significant = "✅ Yes" if p_value < self.alpha else "❌ No"
                
                # Différence moyenne
                mean_diff = np.mean(hybrid_accs - base_accs)
                
                results.append({
                    'Comparison': f"{hybrid_name} vs {base_name}",
                    'Hybrid_Mean': np.mean(hybrid_accs),
                    'Base_Mean': np.mean(base_accs),
                    'Mean_Diff': mean_diff,
                    'W_Statistic': statistic,
                    'p_value': p_value,
                    f'Significant (α={self.alpha})': significant
                })
                
                print(f"{hybrid_name:15s} vs {base_name:10s}: "
                      f"W={statistic:6.1f}, p={p_value:.6f} {significant}")
                
            except Exception as e:
                print(f"❌ {hybrid_name} vs {base_name}: Erreur - {e}")
        
        df_results = pd.DataFrame(results)
        
        print(f"\n📊 Summary: {len(df_results)} comparisons made")
        significant_count = df_results[
            df_results[f'Significant (α={self.alpha})'] == "✅ Yes"
        ].shape[0]
        print(f"   ✅ Significatifs: {significant_count}/{len(df_results)}")
        
        return df_results
    
    
    def friedman_test(self) -> dict:
        """
        Test de Friedman pour comparer tous les algorithmes
        
        Returns:
            Dictionnaire avec résultats du test
        """
        print("\n" + "="*80)
        print("2️⃣  FRIEDMAN TEST (Multiple Algorithms Comparison)")
        print("="*80)
        
        # Organiser données par algorithme
        algorithms = self.df['Algorithm'].unique()
        datasets = self.df['Dataset'].unique()
        
        print(f"Number of algorithms: {len(algorithms)}")
        print(f"Number of datasets: {len(datasets)}")
        
        # Créer matrice (datasets × algorithmes)
        data_by_algo = {}
        for algo in algorithms:
            data_by_algo[algo] = self.df[
                self.df['Algorithm'] == algo
            ]['Accuracy_Mean'].values
        
        # Vérifier même nombre de datasets pour tous
        lengths = [len(data_by_algo[algo]) for algo in algorithms]
        if len(set(lengths)) > 1:
            print(f"⚠️  Attention: Number of datasets is different between algorithms")
            print(f"   Lengths: {dict(zip(algorithms, lengths))}")
        
        # Test de Friedman
        data_arrays = [data_by_algo[algo] for algo in algorithms]
        statistic, p_value = friedmanchisquare(*data_arrays)
        
        significant = p_value < self.alpha
        
        print(f"\nFriedman χ² = {statistic:.4f}")
        print(f"p-value = {p_value:.6f}")
        print(f"Degree of freedom (df) = {len(algorithms) - 1}")
        
        if significant:
            print(f" SIGNIFICANT (p < {self.alpha})")
            print("   → At least one difference exists between algorithms")
            print("   → Proceed with the post-hoc test of Nemenyi")
        else:
            print(f"❌ NON SIGNIFICANT (p >= {self.alpha})")
            print("   → NO SIGNIFICANT difference detected")
            print("   → Post-hoc Test not necessary")
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'df': len(algorithms) - 1,
            'significant': significant,
            'n_algorithms': len(algorithms),
            'n_datasets': len(datasets)
        }
    
    
    def nemenyi_posthoc_test(self) -> tuple:
        """
        Test post-hoc de Nemenyi
        
        À exécuter UNIQUEMENT si Friedman significatif
        
        Returns:
            (comparisons_df, mean_ranks, CD)
        """
        print("\n" + "="*80)
        print("3️⃣  NEMENYI POST-HOC TEST")
        print("="*80)
        
        algorithms = self.df['Algorithm'].unique()
        datasets = self.df['Dataset'].unique()
        
        n_algorithms = len(algorithms)
        n_datasets = len(datasets)
        
        # Calculer rangs moyens pour chaque algorithme
        ranks_by_algo = {algo: [] for algo in algorithms}
        
        for dataset in datasets:
            # Performances de tous les algos sur ce dataset
            dataset_data = self.df[self.df['Dataset'] == dataset]
            performances = []
            algos_in_dataset = []
            
            for algo in algorithms:
                algo_data = dataset_data[dataset_data['Algorithm'] == algo]
                if not algo_data.empty:
                    performances.append(algo_data['Accuracy_Mean'].values[0])
                    algos_in_dataset.append(algo)
            
            # Calculer rangs (1=meilleur car on maximise accuracy)
            # rankdata avec negative pour que meilleur = rang 1
            ranks = rankdata([-p for p in performances])
            
            # Stocker rangs
            for i, algo in enumerate(algos_in_dataset):
                ranks_by_algo[algo].append(ranks[i])
        
        # Rangs moyens
        mean_ranks = {
            algo: np.mean(ranks) if len(ranks) > 0 else np.nan
            for algo, ranks in ranks_by_algo.items()
        }
        
        # Critical Difference (CD) de Nemenyi
        # CD = q_α * sqrt(k(k+1) / (6N))
        # où k=nombre d'algorithmes, N=nombre de datasets
        
        # Valeurs critiques q pour α=0.05 (Nemenyi)
        q_values = {
            2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
            7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164
        }
        
        q_alpha = q_values.get(n_algorithms, 3.031)
        CD = q_alpha * np.sqrt((n_algorithms * (n_algorithms + 1)) / (6 * n_datasets))
        
        print(f"Number of algorithms: {n_algorithms}")
        print(f"Number of datasets: {n_datasets}")
        print(f"critical Value  q(α={self.alpha}): {q_alpha}")
        print(f"Critical Difference (CD): {CD:.3f}")
        
        print("\n📊 Medium Ranks (1 = meilleur):")
        for algo, rank in sorted(mean_ranks.items(), key=lambda x: x[1]):
            print(f"   {rank:.2f} - {algo}")
        
        # Comparaisons pairées
        comparisons = []
        
        print("\n🔍 Significant paid comparisons  (rank diff > CD):")
        
        for algo1, algo2 in combinations(algorithms, 2):
            rank1 = mean_ranks[algo1]
            rank2 = mean_ranks[algo2]
            
            if np.isnan(rank1) or np.isnan(rank2):
                continue
            
            rank_diff = abs(rank1 - rank2)
            significant = rank_diff > CD
            
            comparisons.append({
                'Algorithm 1': algo1,
                'Algorithm 2': algo2,
                'Rank 1': rank1,
                'Rank 2': rank2,
                'Rank Diff': rank_diff,
                'Critical Diff (CD)': CD,
                'Significant': '✅ Yes' if significant else '❌ No'
            })
            
            if significant:
                better = algo1 if rank1 < rank2 else algo2
                worse = algo2 if rank1 < rank2 else algo1
                print(f"   ✅ {better} > {worse} (diff={rank_diff:.2f} > CD={CD:.2f})")
        
        comparisons_df = pd.DataFrame(comparisons)
        
        significant_count = comparisons_df[
            comparisons_df['Significant'] == '✅ Yes'
        ].shape[0]
        print(f"\n📊 Total comparisons: {len(comparisons_df)}")
        print(f"   ✅ Significant: {significant_count}")
        
        return comparisons_df, mean_ranks, CD
    
    
    def cohens_d_effect_size(self) -> pd.DataFrame:
        """
        Calcule Cohen's d (effect size) pour chaque paire
        
        Returns:
            DataFrame avec effect sizes
        """
        print("\n" + "="*80)
        print("4️⃣  COHEN'S d EFFECT SIZE")
        print("="*80)
        
        results = []
        
        for hybrid_name in self.hybrid_algos['Algorithm'].unique():
            base_name = hybrid_name.replace('-bABER', '')
            
            hybrid_data = self.hybrid_algos[
                self.hybrid_algos['Algorithm'] == hybrid_name
            ]
            base_data = self.base_algos[
                self.base_algos['Algorithm'] == base_name
            ]
            
            if base_data.empty:
                continue
            
            # Accuracies
            hybrid_accs = hybrid_data['Accuracy_Mean'].values
            base_accs = base_data['Accuracy_Mean'].values
            
            # Cohen's d
            n1, n2 = len(base_accs), len(hybrid_accs)
            mean1, mean2 = np.mean(base_accs), np.mean(hybrid_accs)
            var1, var2 = np.var(base_accs, ddof=1), np.var(hybrid_accs, ddof=1)
            
            # Pooled standard deviation
            pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
            
            # Cohen's d
            d = (mean2 - mean1) / pooled_std if pooled_std > 0 else 0
            
            # Interprétation
            abs_d = abs(d)
            if abs_d < 0.2:
                magnitude = "Negligible"
            elif abs_d < 0.5:
                magnitude = "Small"
            elif abs_d < 0.8:
                magnitude = "Medium"
            else:
                magnitude = "Large"
            
            results.append({
                'Comparison': f"{hybrid_name} vs {base_name}",
                'Mean_Diff': mean2 - mean1,
                "Cohen's d": d,
                'Effect Size': magnitude
            })
            
            print(f"{hybrid_name:15s} vs {base_name:10s}: "
                  f"d={d:+.3f} ({magnitude})")
        
        return pd.DataFrame(results)
    
    def execution_time_analysis(self) -> pd.DataFrame:
        """
        Analyse statistique des temps d'exécution
        
        Returns:
            DataFrame avec comparaisons des temps
        """
        print("\n" + "="*80)
        print("5️⃣  EXECUTION TIME ANALYSIS")
        print("="*80)
        
        results = []
        
        for hybrid_name in self.hybrid_algos['Algorithm'].unique():
            base_name = hybrid_name.replace('-bABER', '')
            
            hybrid_data = self.hybrid_algos[
                self.hybrid_algos['Algorithm'] == hybrid_name
            ]
            base_data = self.base_algos[
                self.base_algos['Algorithm'] == base_name
            ]
            
            if base_data.empty:
                continue
            
            # Temps moyens
            hybrid_time = hybrid_data['Execution_Time_Mean'].values
            base_time = base_data['Execution_Time_Mean'].values
            
            if len(hybrid_time) != len(base_time):
                continue
            
            # Test de Wilcoxon sur les temps
            from scipy.stats import wilcoxon
            try:
                statistic, p_value = wilcoxon(
                    hybrid_time, 
                    base_time, 
                    alternative='greater'  # Hybride plus lent?
                )
                
                # Différence moyenne
                mean_diff = np.mean(hybrid_time - base_time)
                percent_increase = (mean_diff / np.mean(base_time)) * 100
                
                results.append({
                    'Comparison': f"{hybrid_name} vs {base_name}",
                    'Base_Time_Mean': np.mean(base_time),
                    'Hybrid_Time_Mean': np.mean(hybrid_time),
                    'Time_Diff': mean_diff,
                    'Percent_Increase': percent_increase,
                    'W_Statistic': statistic,
                    'p_value': p_value,
                    'Significant': "✅ Yes" if p_value < 0.05 else " No"
                })
                
                print(f"{hybrid_name:15s} vs {base_name:10s}: "
                      f"Base={np.mean(base_time):.2f}s, "
                      f"Hybrid={np.mean(hybrid_time):.2f}s, "
                      f"Overhead={percent_increase:+.1f}%")
                
            except Exception as e:
                print(f"❌ {hybrid_name} vs {base_name}: Erreur - {e}")
        
        df_results = pd.DataFrame(results)
        
        print(f"\n📊 Overhead moyen hybrides: "
              f"{df_results['Percent_Increase'].mean():.1f}%")
        
        return df_results  
    
    
    def generate_statistical_report(self, output_dir: str = "results/statistics"):
        """
        Génère un rapport statistique complet
        
        Args:
            output_dir: Répertoire de sortie
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*80)
        print(" "*20 + "GENERATION OF STATISTICAL REPORT ")
        print("="*80)
        
        # 1. Wilcoxon tests
        wilcoxon_results = self.wilcoxon_pairwise_tests()
        wilcoxon_results.to_csv(f"{output_dir}/wilcoxon_tests.csv", index=False)
        print(f"\n✅ Wilcoxon: {output_dir}/wilcoxon_tests.csv")
        
        # 2. Friedman test
        friedman_results = self.friedman_test()
        
        # Sauvegarder résultats Friedman
        with open(f"{output_dir}/friedman_test.txt", 'w') as f:
            f.write("FRIEDMAN TEST RESULTS\n")
            f.write("="*50 + "\n\n")
            f.write(f"Chi-square statistic: {friedman_results['statistic']:.4f}\n")
            f.write(f"p-value: {friedman_results['p_value']:.6f}\n")
            f.write(f"Degrees of freedom: {friedman_results['df']}\n")
            f.write(f"Significant (α={self.alpha}): {friedman_results['significant']}\n")
        
        print(f"✅ Friedman: {output_dir}/friedman_test.txt")
        
        # 3. Nemenyi (si Friedman significatif)
        if friedman_results['significant']:
            nemenyi_comparisons, mean_ranks, CD = self.nemenyi_posthoc_test()
            nemenyi_comparisons.to_csv(f"{output_dir}/nemenyi_posthoc.csv", index=False)
            
            # Sauvegarder rangs moyens
            ranks_df = pd.DataFrame([
                {'Algorithm': algo, 'Mean_Rank': rank}
                for algo, rank in sorted(mean_ranks.items(), key=lambda x: x[1])
            ])
            ranks_df.to_csv(f"{output_dir}/nemenyi_mean_ranks.csv", index=False)
            
            print(f"✅ Nemenyi: {output_dir}/nemenyi_posthoc.csv")
            print(f"✅ Rangs: {output_dir}/nemenyi_mean_ranks.csv")
        else:
            print("⚠️  Nemenyi: Non exécuté (Friedman non significatif)")
        
        # 4. Cohen's d
        cohens_results = self.cohens_d_effect_size()
        cohens_results.to_csv(f"{output_dir}/cohens_d_effect_size.csv", index=False)
        print(f"✅ Cohen's d: {output_dir}/cohens_d_effect_size.csv")
        
        # 5. Tableau récapitulatif LaTeX
        self._generate_latex_statistical_table(
            wilcoxon_results,
            friedman_results,
            cohens_results,
            output_dir
        )
        
        print("\n" + "="*80)
        print(" "*15 + "FULL STATISTICAL REPORT GENERATED!")
        print("="*80)
    
    
    def _generate_latex_statistical_table(self, wilcoxon_df, friedman_dict, 
                                          cohens_df, output_dir):
        """Génère tableau LaTeX pour l'article"""
        
        latex = "\\begin{table}[htbp]\n"
        latex += "\\centering\n"
        latex += "\\caption{Statistical Validation Results}\n"
        latex += "\\label{tab:statistical_tests}\n"
        latex += "\\begin{tabular}{llccc}\n"
        latex += "\\hline\n"
        latex += "Test & Comparison & Statistic & p-value & Significant \\\\\n"
        latex += "\\hline\n"
        
        # Friedman
        latex += f"Friedman & All algorithms & "
        latex += f"$\\chi^2={friedman_dict['statistic']:.2f}$ & "
        latex += f"{friedman_dict['p_value']:.4f} & "
        latex += "Yes \\\\\n" if friedman_dict['significant'] else "No \\\\\n"
        latex += "\\hline\n"
        
        # Wilcoxon (top 3)
        for idx, row in wilcoxon_df.head(3).iterrows():
            latex += f"Wilcoxon & {row['Comparison']} & "
            latex += f"W={row['W_Statistic']:.1f} & "
            latex += f"{row['p_value']:.4f} & "
            sig = "Yes" if "Yes" in row[f'Significant (α={self.alpha})'] else "No"
            latex += f"{sig} \\\\\n"
        
        latex += "\\hline\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}\n"
        
        with open(f"{output_dir}/statistical_table.tex", 'w') as f:
            f.write(latex)
        
        print(f"✅ LaTeX: {output_dir}/statistical_table.tex")


# ============================================================================
# SCRIPT D'EXÉCUTION
# ============================================================================

def main():
    """
    Script principal pour lancer l'analyse statistique
    
    Usage:
        python src/evaluation/statistical_tests.py
    """
    import sys
    
    print("="*80)
    print(" "*20 + "COMPLETE STATISTICAL ANALYSIS")
    print("="*80)
    
    # Chercher le fichier de résultats le plus récent
    from glob import glob
    
    results_files = glob("results/metrics/results_unified_*.csv")
    
    if not results_files:
        print("❌ No results file found!")
        print("   Please execute first:")
        print("   python experiments/run_experiments.py --mode standard --hybrids-only")
        sys.exit(1)
    
    # Prendre le plus récent
    latest_file = max(results_files, key=lambda x: x)
    
    print(f"\n📁 Results File: {latest_file}")
    
    # Demander confirmation
    response = input("\nContinue with this file? [y/N]: ")
    if response.lower() != 'y':
        print("Cancel.")
        sys.exit(0)
    
    # Créer analyseur
    analyzer = StatisticalAnalyzer(latest_file, alpha=0.05)
    
    # Générer rapport complet
    analyzer.generate_statistical_report()
    
    print("\n✅ Analysis compleed!")
    print("\n📚 Files Generated in : results/statistics/")
    print("   - wilcoxon_tests.csv")
    print("   - friedman_test.txt")
    print("   - nemenyi_posthoc.csv (si applicable)")
    print("   - cohens_d_effect_size.csv")
    print("   - statistical_table.tex (pour article)")


if __name__ == "__main__":
    main()