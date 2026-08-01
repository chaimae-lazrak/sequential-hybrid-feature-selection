#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 17:36:03 2026

@author: chamaelazrak
"""

import pandas as pd
from pathlib import Path


from sklearn.datasets import load_svmlight_file
import numpy as np



class DataLoader:
    """Chargement des datasets médicaux"""
    
    BASE_DIR = Path(__file__).resolve().parents[2]
    DATA_DIR = BASE_DIR / "data" / "raw"
    
    @staticmethod
    def load_parkinsons(filepath: str = None):
        """Charge le dataset Parkinson's"""
        
        
        path = Path(filepath) if filepath else DataLoader.DATA_DIR / "parkinsons.csv"
        df = pd.read_csv(path)
        
        X = df.drop(['name', 'status'], axis=1).values
        y = df['status'].values
        feature_names = df.columns.drop(['name', 'status']).tolist()
        return X, y, feature_names
    
    @staticmethod
    def load_diabetes(filepath: str = None):
        """Charge le dataset Diabetes (PIMA)"""
        
    
            
        path = Path(filepath) if filepath else DataLoader.DATA_DIR / "diabetes.csv"
        df = pd.read_csv(path)
        feature_names = df.columns[:-1].tolist()
        
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        return X, y, feature_names
    
    @staticmethod
    def load_heart(filepath: str = None):
       
            
        path = Path(filepath) if filepath else DataLoader.DATA_DIR / "heart_disease.csv"
        df = pd.read_csv(path)
        
        # Convertir toutes les colonnes en numérique si possible (les non-numériques -> NaN)
        df = df.apply(pd.to_numeric, errors="coerce")


        # Suppression valeurs manquantes si présentes
        df = df.dropna()
        
        # X = toutes colonnes sauf la dernière
        X = df.iloc[:, :-1].values
        #y = (df.iloc[:, -1].values > 0).astype(int)  # Binaire
        
        # y = dernière colonne, binarisée (0 -> 0 ; 1.. -> 1)
        y_raw = df.iloc[:, -1].astype(int).values
        y = (y_raw > 0).astype(int)
        
        feature_names = df.columns[:-1].tolist()
        return X, y, feature_names
    
    @staticmethod
    def load_breast_cancer(filepath: str = None):
        """Charge le dataset Breast Cancer Wisconsin"""
        #data = load_breast_cancer()
        
        path = Path(filepath) if filepath else DataLoader.DATA_DIR / "breast-cancer.csv"
        df = pd.read_csv(path)
        
        
        y = df["diagnosis"].map({"M": 1, "B": 0})
        X_df = df.drop(columns=["diagnosis"])
        if "id" in X_df.columns:
            X_df = X_df.drop(columns=["id"])
        X_df = X_df.select_dtypes(include=["number"])    
            
        # Sécurité : supprimer lignes invalides
        valid = y.notna()
        X_df = X_df.loc[valid]
        y = y.loc[valid].astype(int)        
    
        return X_df.values, y.values, X_df.columns.tolist()
    
    @staticmethod
    def load_cervical_cancer(filepath: str = None):
        """Charge le dataset Cervical Cancer (Risk Factors)"""
    
        path = Path(filepath) if filepath else DataLoader.DATA_DIR / "kag_risk_factors_cervical_cancer.csv"
        df = pd.read_csv(path)
    
        # Convertir toutes les colonnes en numérique (les erreurs → NaN)
        df = df.apply(pd.to_numeric, errors="coerce")
    
        # Cible
        target = "Biopsy"
    
        # Supprimer lignes sans cible
        df = df.dropna(subset=[target])
    
        # Séparer X / y
        y = df[target].astype(int).values
        X_df = df.drop(columns=[target])
    
        # Imputation simple (moyenne)
        X_df = X_df.fillna(X_df.mean())
    
        feature_names = X_df.columns.tolist()
        X = X_df.values
    
        return X, y, feature_names
    
    @staticmethod
    def load_alzheimers(filepath: str = None):
        """Charge le dataset Alzheimer's Disease"""
    
        path = Path(filepath) if filepath else DataLoader.DATA_DIR / "alzheimers.csv"
        df = pd.read_csv(path)
    
        # Supprimer colonnes non informatives
        cols_to_drop = [c for c in ["PatientID", "DoctorInCharge"] if c in df.columns]
        df = df.drop(columns=cols_to_drop)
    
        # Convertir en numérique
        df = df.apply(pd.to_numeric, errors="coerce")
    
        # Cible
        target = "Diagnosis"
    
        # Supprimer lignes sans cible
        df = df.dropna(subset=[target])
    
        y = df[target].astype(int).values
        X_df = df.drop(columns=[target])
    
        # Imputation simple
        X_df = X_df.fillna(X_df.mean())
    
        feature_names = X_df.columns.tolist()
        X = X_df.values
    
        return X, y, feature_names
