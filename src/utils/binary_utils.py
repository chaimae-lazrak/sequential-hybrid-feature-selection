#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 17:18:35 2026

@author: chamaelazrak
"""
import numpy as np

class  BinaryConverter:
    """Utilitaires pour convertir solutions continues en binaires"""
    
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """Fonction sigmoïde"""
        x_clipped = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x_clipped))
    
    @staticmethod
    def v_shaped(x: np.ndarray) -> np.ndarray:
        """Fonction de transfert en V"""
        return np.abs(2 / np.pi * np.arctan(np.pi / 2 * x))
    
    @staticmethod
    def s_shaped(x: np.ndarray) -> np.ndarray:
        """Fonction de transfert en S"""
        return 1 / (1 + np.exp(-2 * x))
    
    @staticmethod
    def to_binary(x: np.ndarray, transfer_func: str = 'sigmoid') -> np.ndarray:
        """Convertit vecteur continu en binaire"""
        if transfer_func == 'sigmoid':
            proba = BinaryConverter.sigmoid(x)
        elif transfer_func == 'v_shaped':
            proba = BinaryConverter.v_shaped(x)
        else:
            proba = BinaryConverter.s_shaped(x)
        
        return (np.random.rand(len(x)) < proba).astype(int)
