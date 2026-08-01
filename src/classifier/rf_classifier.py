#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 17:24:06 2026

@author: chamaelazrak
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class RFClassifierWrapper:
    """Wrapper pour Random Forest avec évaluation"""
    
    def __init__(self, n_estimators: int = 100, random_state: int = 42, **kwargs):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            **kwargs
        )
        
    def evaluate_features(self, X_train: np.ndarray, y_train: np.ndarray,
                         X_test: np.ndarray, y_test: np.ndarray,
                         selected_features: np.ndarray):
        """Évalue un sous-ensemble de features"""
        
        # Sélection des features
        selected_indices = np.where(selected_features == 1)[0]
        
        if len(selected_indices) == 0:
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 
                   'f1': 0.0, 'n_features': 0}
        
        X_train_selected = X_train[:, selected_indices]
        X_test_selected = X_test[:, selected_indices]
        
        # Entraînement
        self.model.fit(X_train_selected, y_train)
        
        # Prédiction
        y_pred = self.model.predict(X_test_selected)
        
        # Métriques
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'n_features': len(selected_indices)
        }
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, 
                       selected_features: np.ndarray, cv: int = 5) -> float:
        """Validation croisée"""
        selected_indices = np.where(selected_features == 1)[0]
        
        if len(selected_indices) == 0:
            return 0.0
        
        X_selected = X[:, selected_indices]
        scores = cross_val_score(self.model, X_selected, y, cv=cv, scoring='accuracy')
        
        return np.mean(scores)
