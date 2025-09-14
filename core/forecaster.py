from __future__ import annotations
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import STL


class Forecaster:
    def __init__(self, df: pd.DataFrame, date_col: str, value_col: str, season: Optional[int] = None):
        self.df = df.dropna(subset=[date_col, value_col]).copy()
        self.df[date_col] = pd.to_datetime(self.df[date_col])
        self.df = self.df.sort_values(date_col)
        self.date_col = date_col
        self.value_col = value_col
        self.season = season

    def fit_predict(self, horizon: int = 12) -> Tuple[pd.DataFrame, Dict]:
        y = self.df[self.value_col].astype(float).values
        if len(y) < 3:
            raise ValueError("Not enough data to forecast.")

        freq = pd.infer_freq(self.df[self.date_col]) or 'D'
        season_len = self.season or (12 if 'M' in freq else 7 if 'D' in freq else 12)
        try:
            model = ExponentialSmoothing(y, trend='add', seasonal='add', seasonal_periods=season_len)
            fit = model.fit(optimized=True)
            fc = fit.forecast(horizon)
            residuals = y - fit.fittedvalues
        except Exception:
            model = ExponentialSmoothing(y, trend='add', seasonal=None)
            fit = model.fit(optimized=True)
            fc = fit.forecast(horizon)
            residuals = y - fit.fittedvalues

        last_date = self.df[self.date_col].iloc[-1]
        idx = pd.date_range(last_date, periods=horizon+1, freq=freq)[1:]
        out = pd.DataFrame({self.date_col: idx, 'forecast': fc})

        diag = {}
        if len(y) >= 2*season_len+1:
            try:
                stl = STL(self.df.set_index(self.date_col)[self.value_col], period=season_len).fit()
                diag = {
                    'trend_var': float(np.var(stl.trend)),
                    'seasonal_var': float(np.var(stl.seasonal)),
                    'resid_var': float(np.var(stl.resid)),
                }
            except Exception:
                pass
        diag['mae'] = float(np.mean(np.abs(residuals))) if len(residuals) else None
        diag['mape_%'] = float(np.mean(np.abs(residuals / (y + 1e-9)))*100) if len(residuals) else None
        diag['season_len'] = season_len
        diag['freq'] = freq
        return out, diag