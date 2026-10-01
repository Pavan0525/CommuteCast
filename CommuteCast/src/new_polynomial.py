"""
CommuteCast - New Dataset Polynomial Curve Fitting
Module: src/new_polynomial.py

Uses Traffic dataset.csv.
X variable: day_num (chronological integer 0 to 364)
Y variable: mean Vehicle_Count aggregated per day
Fits Degree 1, 2, and 3 polynomials using OLS.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path = [p for p in sys.path if Path(p).resolve() != Path(__file__).resolve().parent]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.new_preprocessing import preprocess_new_traffic_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_SELECTED_DEGREE = 3


def prepare_daily_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate Vehicle_Count by day_num (mean per day).
    Returns DataFrame with columns ['day_num', 'mean_vehicle_count'].
    """
    if 'day_num' not in df.columns or 'Vehicle_Count' not in df.columns:
        raise KeyError("Input DataFrame must contain 'day_num' and 'Vehicle_Count' columns.")
    daily_df = df.groupby('day_num')['Vehicle_Count'].mean().reset_index()
    daily_df.columns = ['day_num', 'mean_vehicle_count']
    daily_df = daily_df.sort_values('day_num').reset_index(drop=True)
    return daily_df


def fit_polynomial_models(X: np.ndarray, y: np.ndarray, degrees: Tuple = (1, 2, 3)) -> Dict[int, Dict[str, Any]]:
    """Fit polynomial models of given degrees using OLS (np.polyfit)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    fitted = {}
    for deg in degrees:
        coeffs_desc = np.polyfit(X, y, deg)
        p = np.poly1d(coeffs_desc)
        coeffs_asc = coeffs_desc[::-1]
        terms = [f'{coeffs_asc[0]:.2f}']
        if deg >= 1:
            sign = '+' if coeffs_asc[1] >= 0 else '-'
            terms.append(f'{sign} {abs(coeffs_asc[1]):.4f}x')
        if deg >= 2:
            sign = '+' if coeffs_asc[2] >= 0 else '-'
            terms.append(f'{sign} {abs(coeffs_asc[2]):.6f}x2')
        if deg >= 3:
            sign = '+' if coeffs_asc[3] >= 0 else '-'
            terms.append(f'{sign} {abs(coeffs_asc[3]):.8f}x3')
        eq_str = f'y = {" ".join(terms)}'
        fitted[deg] = {
            'degree': deg,
            'coeffs_descending': coeffs_desc,
            'coeffs_ascending': coeffs_asc,
            'poly1d': p,
            'equation_str': eq_str
        }
    return fitted


def evaluate_polynomial_models(models: Dict[int, Dict], X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Evaluate fitted polynomial models: MAE, RMSE, R2, Mean Residual."""
    records = []
    for deg, model in sorted(models.items()):
        p = model['poly1d']
        y_pred = p(X)
        residuals = y - y_pred
        mae = float(mean_absolute_error(y, y_pred))
        rmse = float(np.sqrt(np.mean(residuals ** 2)))
        r2 = float(r2_score(y, y_pred))
        mean_res = float(np.mean(residuals))
        model['y_pred'] = y_pred
        model['residuals'] = residuals
        model['mae'] = mae
        model['rmse'] = rmse
        model['r2'] = r2
        model['mean_residual'] = mean_res
        records.append({'Model': f'Degree {deg} Polynomial', 'Degree': deg,
                        'Equation': model['equation_str'], 'MAE': mae, 'RMSE': rmse, 'R2': r2,
                        'Mean Residual': mean_res})
    return pd.DataFrame(records)


class NewTrafficPolynomialPredictor:
    """Polynomial predictor for daily Vehicle_Count based on day_num."""
    def __init__(self, daily_df: Optional[pd.DataFrame] = None, degrees: Tuple = (1, 2, 3)):
        if daily_df is None:
            df = preprocess_new_traffic_data()
            daily_df = prepare_daily_data(df)
        self.daily_df = daily_df
        self.X = daily_df['day_num'].values
        self.y = daily_df['mean_vehicle_count'].values
        self.models = fit_polynomial_models(self.X, self.y, degrees)
        self.eval_df = evaluate_polynomial_models(self.models, self.X, self.y)
        self.selected_degree = DEFAULT_SELECTED_DEGREE

    def predict(self, day_num: Union[int, float], degree: Optional[int] = None) -> float:
        """Estimate Vehicle_Count for a given day_num."""
        if degree is None:
            degree = self.selected_degree
        if degree not in self.models:
            raise ValueError(f'Degree {degree} not available.')
        if not isinstance(day_num, (int, float, np.integer, np.floating)):
            raise TypeError(f'day_num must be numeric, got {type(day_num).__name__}.')
        p = self.models[degree]['poly1d']
        return max(0.0, float(p(day_num)))


_PREDICTOR_INSTANCE: Optional[NewTrafficPolynomialPredictor] = None


def get_new_predictor() -> NewTrafficPolynomialPredictor:
    global _PREDICTOR_INSTANCE
    if _PREDICTOR_INSTANCE is None:
        _PREDICTOR_INSTANCE = NewTrafficPolynomialPredictor()
    return _PREDICTOR_INSTANCE


def predict_vehicle_count(day_num: Union[int, float], degree: int = DEFAULT_SELECTED_DEGREE) -> float:
    """Functional interface to predict Vehicle_Count for a given day_num."""
    return get_new_predictor().predict(day_num, degree=degree)
