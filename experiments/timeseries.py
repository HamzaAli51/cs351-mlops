import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

from src.data.ingest import load_data

warnings.simplefilter("ignore")
FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)
HOLDOUT = 24

ratings, _, _ = load_data()
ratings["date"] = pd.to_datetime(ratings["timestamp"], unit="s")
series = ratings.set_index("date").resample("ME")["rating"].count().astype(float)
print("months:", len(series), "| zero-rating months:", int((series == 0).sum()))

decomp = seasonal_decompose(series, model="additive", period=12)
decomp.plot().savefig(FIG_DIR / "ts_decomposition.png", bbox_inches="tight")
plt.close("all")

adf_p = adfuller(series)[1]
print(f"ADF p-value: {adf_p:.4f} ({'stationary' if adf_p < 0.05 else 'non-stationary'})")

train, test = series.iloc[:-HOLDOUT], series.iloc[-HOLDOUT:]

forecasts = {
    "naive_last_value": np.full(HOLDOUT, train.iloc[-1]),
    "mean_last_12_months": np.full(HOLDOUT, train.iloc[-12:].mean()),
    "seasonal_naive": np.tile(train.iloc[-12:].to_numpy(), HOLDOUT // 12),
    "holt_winters": ExponentialSmoothing(
        train, trend="add", damped_trend=True,
        seasonal="add", seasonal_periods=12).fit().forecast(HOLDOUT).to_numpy(),
    "arima_1_1_1": ARIMA(train, order=(1, 1, 1)).fit().forecast(HOLDOUT).to_numpy(),
}
clipped = {name: np.clip(pred, 0, None) for name, pred in forecasts.items()}

results = pd.DataFrame([
    {"model": name,
     "rmse": float(np.sqrt(mean_squared_error(test, pred))),
     "mae": float(mean_absolute_error(test, pred))}
    for name, pred in clipped.items()
])
results.to_csv("reports/results_timeseries.csv", index=False)
print(results.round(1).to_string(index=False))

plt.figure(figsize=(11, 4))
plt.plot(series.index[-60:], series.iloc[-60:], label="actual", color="black")
for name, pred in clipped.items():
    plt.plot(test.index, pred, label=name)
plt.legend()
plt.title("Monthly rating volume: last 60 months and 24-month forecasts")
plt.savefig(FIG_DIR / "ts_forecasts.png", bbox_inches="tight")
plt.close()