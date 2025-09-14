from __future__ import annotations
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pmdarima as pm

from .utils import parse_date_series


class Forecaster:
    def __init__(self, df: pd.DataFrame, date_col: str, value_col: str):
        self.raw = df.copy()
        self.date_col = date_col
        self.value_col = value_col
        self.df = df.dropna(subset=[date_col, value_col]).copy()
        self.df[date_col] = parse_date_series(self.df[date_col])
        self.df = self.df.dropna(subset=[date_col])
        self.df = self.df.sort_values(date_col)

    def _infer_freq_and_season(self) -> Tuple[str, int]:
        freq = pd.infer_freq(self.df[self.date_col]) or 'D'
        season_len = 12 if 'M' in freq else 7 if 'D' in freq else 12
        return freq, season_len

    def fit_predict(self, model: str = 'ETS', horizon: int = 12, season: Optional[int] = None) -> Tuple[pd.DataFrame, Dict]:
        freq, season_len = self._infer_freq_and_season()
        if season:
            season_len = season
        y = self.df[self.value_col].astype(float).values
        if len(y) < 3:
            raise ValueError("Not enough data to forecast.")

        last_date = self.df[self.date_col].iloc[-1]
        idx = pd.date_range(last_date, periods=horizon+1, freq=freq)[1:]

        diag: Dict = {"model": model, "freq": freq, "season_len": season_len}

        if model.upper() == 'ETS':
            try:
                est = ExponentialSmoothing(y, trend='add', seasonal='add', seasonal_periods=season_len)
                fit = est.fit(optimized=True)
            except Exception:
                est = ExponentialSmoothing(y, trend='add', seasonal=None)
                fit = est.fit(optimized=True)
            fc = fit.forecast(horizon)
            resid = y - fit.fittedvalues
            diag.update({"mae": float(np.mean(np.abs(resid))), "mape_%": float(np.mean(np.abs(resid/(y+1e-9)))*100)})
            out = pd.DataFrame({self.date_col: idx, 'forecast': fc})
            return out, diag

        if model.upper() in ('ARIMA', 'ARIMAX'):
            m = season_len if season_len > 1 else 1
            try:
                fit = pm.auto_arima(y, seasonal=(m>1), m=m, error_action='ignore', suppress_warnings=True)
                fc = fit.predict(n_periods=horizon)
                resid = y - fit.predict_in_sample()
                diag.update({"order": str(fit.order), "seasonal_order": str(fit.seasonal_order)})
                diag.update({"mae": float(np.mean(np.abs(resid))), "mape_%": float(np.mean(np.abs(resid/(y+1e-9)))*100)})
                out = pd.DataFrame({self.date_col: idx, 'forecast': fc})
                return out, diag
            except Exception as e:
                raise RuntimeError(f"ARIMA failed: {e}")

        if model.upper() in ('RF', 'RANDOM_FOREST'):
            # Basic date features
            ds = self.df[[self.date_col]].copy()
            ds['t'] = np.arange(len(ds))
            ds['month'] = ds[self.date_col].dt.month
            ds['quarter'] = ds[self.date_col].dt.quarter
            ds['dow'] = ds[self.date_col].dt.dayofweek
            X = ds[['t','month','quarter','dow']].values
            yv = self.df[self.value_col].astype(float).values
            rf = RandomForestRegressor(n_estimators=400, random_state=42)
            rf.fit(X, yv)
            future = pd.DataFrame({self.date_col: idx})
            future['t'] = np.arange(len(ds), len(ds)+horizon)
            future['month'] = future[self.date_col].dt.month
            future['quarter'] = future[self.date_col].dt.quarter
            future['dow'] = future[self.date_col].dt.dayofweek
            fc = rf.predict(future[['t','month','quarter','dow']].values)
            pred_in = rf.predict(X)
            resid = yv - pred_in
            diag.update({"mae": float(np.mean(np.abs(resid))), "mape_%": float(np.mean(np.abs(resid/(yv+1e-9)))*100)})
            out = pd.DataFrame({self.date_col: idx, 'forecast': fc})
            return out, diag

        raise ValueError("Unknown model: choose ETS, ARIMA, or RF")