#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 17:26:02 2026

@author: chamaelazrak
"""


from src.classifier.rf_classifier import RFClassifierWrapper
import numpy as np


class FitnessFunction:
    """Fonction de fitness pour la sélection de features"""
    
    def __init__(self, X_train, y_train, classifier, alpha=0.99, cv=5):
        """
        Initialise la fonction de fitness
        
        Args:
            X_train: Features d'entraînement (numpy array)
            y_train: Labels d'entraînement (numpy array)
            X_test: Features de test (numpy array)
            y_test: Labels de test (numpy array)
            classifier: Instance de RFClassifierWrapper
            alpha: Poids pour l'accuracy (par défaut 0.99)
                   fitness = alpha * accuracy - (1-alpha) * ratio_features
        """
        self.X_train = X_train
        self.y_train = y_train
        self.classifier = classifier
        self.alpha = alpha  # Poids pour l'accuracy
        self.cv=cv
        
        # Statistiques pour monitoring
        self.n_evaluations = 0
        
    def __call__(self, solution):
            """
            Calcule la fitness d'une solution via validation croisée
            
            ✅ Utilise RFClassifierWrapper.cross_validate() 
            
            Args:
                solution: Vecteur binaire de sélection de features (0 ou 1)
            
            Returns:
                float: Valeur de fitness (à maximiser)
            """
            self.n_evaluations += 1
            
            # Vérification basique
            selected_indices = np.where(solution == 1)[0]
            if len(selected_indices) == 0:
                return 0.0
            
            try:
                # ✅ UTILISATION de RFClassifierWrapper.cross_validate()
                # Cette méthode fait exactement ce qu'il faut: CV sur train uniquement
                accuracy = self.classifier.cross_validate(
                    self.X_train,
                    self.y_train,
                    solution,
                    cv=self.cv
                )
                
            except Exception as e:
                # En cas d'erreur, retourner 0
                print(f"Erreur dans cross_validate: {e}")
                return 0.0
            
            # Ratio de features sélectionnées
            feature_ratio = len(selected_indices) / len(solution)
            
            # Fitness multi-objectif
            # alpha = 0.99 : priorité à l'accuracy (réduction minime)
            # alpha = 0.50 : équilibre accuracy/réduction
            fitness = self.alpha * accuracy - (1 - self.alpha) * feature_ratio
            
            return fitness
        
    def  evaluate(self, solution):
        """
        Évaluation complète d'une solution via validation croisée
        
        ✅ Utilise RFClassifierWrapper.cross_validate() pour toutes les métriques
        
        Args:
            solution: Vecteur binaire de sélection
        
        Returns:
            dict: Dictionnaire avec toutes les métriques estimées par CV
        """
        from sklearn.model_selection import cross_val_score
        
        selected_indices = np.where(solution == 1)[0]
        
        if len(selected_indices) == 0:
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'n_features': 0,
                'fitness': 0.0,
                'feature_ratio': 0.0,
                'reduction_rate': 1.0
            }
        
        X_selected = self.X_train[:, selected_indices]
        
        try:
            # Validation croisée pour plusieurs métriques
            accuracy_scores = cross_val_score(
                self.classifier.model, X_selected, self.y_train,
                cv=self.cv, scoring='accuracy', n_jobs=-1
            )
            precision_scores = cross_val_score(
                self.classifier.model, X_selected, self.y_train,
                cv=self.cv, scoring='precision_weighted', n_jobs=-1
            )
            recall_scores = cross_val_score(
                self.classifier.model, X_selected, self.y_train,
                cv=self.cv, scoring='recall_weighted', n_jobs=-1
            )
            f1_scores = cross_val_score(
                self.classifier.model, X_selected, self.y_train,
                cv=self.cv, scoring='f1_weighted', n_jobs=-1
            )
            
            accuracy = np.mean(accuracy_scores)
            precision = np.mean(precision_scores)
            recall = np.mean(recall_scores)
            f1 = np.mean(f1_scores)
            
        except Exception as e:
            print(f"Erreur dans evaluate: {e}")
            accuracy = precision = recall = f1 = 0.0
        
        feature_ratio = len(selected_indices) / len(solution)
        fitness = self.alpha * accuracy - (1 - self.alpha) * feature_ratio
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'n_features': len(selected_indices),
            'fitness': fitness,
            'feature_ratio': feature_ratio,
            'reduction_rate': 1 - feature_ratio,
            'cv_std_accuracy': np.std(accuracy_scores) if 'accuracy_scores' in locals() else 0.0
        }
          
      
    def evaluate_on_test(self, solution, X_test, y_test):
        """
        ✅ ÉVALUATION FINALE sur TEST SET
        
        ✅ Utilise RFClassifierWrapper.evaluate_features() de votre architecture
        
        À appeler UNIQUEMENT après l'optimisation complète!
        
        Args:
            solution: Meilleure solution trouvée
            X_test: Features de test
            y_test: Labels de test
        
        Returns:
            dict: Métriques sur le test set
        """
        # ✅ UTILISATION de votre RFClassifierWrapper.evaluate_features()
        # Cette méthode fait exactement ce qu'il faut:
        # 1. Train sur train set complet
        # 2. Évalue sur test set
        # 3. Retourne toutes les métriques
        
        metrics = self.classifier.evaluate_features(
            self.X_train,
            self.y_train,
            X_test,
            y_test,
            solution
        )
        
        return metrics
    
    def get_stats(self):
        """Retourne les statistiques d'utilisation"""
        return {
            'n_evaluations': self.n_evaluations,
            'cv_folds': self.cv,
            'alpha': self.alpha
        }




# Test rapide si exécuté directement
if __name__ == "__main__":
    print("✅ Module fitness (VERSION OPTIMISÉE) chargé avec succès!")
    print("\n🏗️  UTILISE L'ARCHITECTURE EXISTANTE:")
    print("   ✓ RFClassifierWrapper.cross_validate() pour fitness")
    print("   ✓ RFClassifierWrapper.evaluate_features() pour test")
    print("\n🔒 PROTECTION contre data leakage:")
    print("   ✓ Pas de X_test/y_test dans __init__")
    print("   ✓ Validation croisée sur train set uniquement")
    print("   ✓ evaluate_on_test() séparé pour évaluation finale")
    print("Classes disponibles: FitnessFunction")
