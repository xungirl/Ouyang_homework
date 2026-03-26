import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ====================================================================
# DATASET 2: Daily Stock Returns
# ====================================================================
print("\n" + "=" * 70)
print("DATASET 2: Daily Stock Returns")
print("=" * 70)

n_days = 500
dates_stock = pd.date_range('2023-01-01', periods=n_days, freq='D')

returns = np.zeros(n_days)
returns[0] = np.random.normal(0, 0.01)
phi = 0.05
for i in range(1, n_days):
    returns[i] = phi * returns[i-1] + np.random.normal(0, 0.015)

price = 100 * np.exp(np.cumsum(returns))
df_stock = pd.DataFrame({
    'Date': dates_stock, 'Price': price, 'Returns': returns * 100
})
df_stock.set_index('Date', inplace=True)

print(f"Date range: {df_stock.index[0].date()} to {df_stock.index[-1].date()}")
print(f"Number of days: {len(df_stock)}")
print(f"\nPrice Statistics:")
print(df_stock['Price'].describe())


# ====================================================================
# B1: Stationarity Tests
# ====================================================================
print("\n" + "-" * 70)
print("B1: Stationarity Tests")
print("-" * 70)

# Plot 1: Visual stationarity check
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
axes[0].plot(df_stock.index, df_stock['Price'], 'b-')
axes[0].set_title('Stock Price')
axes[0].set_ylabel('Price')

roll_mean = df_stock['Price'].rolling(30).mean()
roll_std = df_stock['Price'].rolling(30).std()
axes[1].plot(df_stock.index, df_stock['Price'], 'b-', alpha=0.4, label='Price')
axes[1].plot(df_stock.index, roll_mean, 'r-', linewidth=2, label='30-day Rolling Mean')
axes[1].set_title('Price with Rolling Mean')
axes[1].legend()
axes[1].set_ylabel('Price')

axes[2].plot(df_stock.index, roll_std, 'g-', linewidth=2)
axes[2].set_title('30-day Rolling Std Dev')
axes[2].set_ylabel('Std Dev')

for ax in axes:
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('B1_stationarity.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: B1_stationarity.png")

print("Visual inspection: Rolling mean changes over time → NOT stationary")

# ADF test
adf_price = adfuller(df_stock['Price'], autolag='AIC')
print(f"\nADF Test on Prices:")
print(f"  Test Statistic = {adf_price[0]:.4f}")
print(f"  p-value        = {adf_price[1]:.4f}")
print(f"  Lags used      = {adf_price[2]}")
print(f"  Decision (α=0.05): {'Reject H0 → Stationary' if adf_price[1] < 0.05 else 'Fail to reject H0 → Non-stationary'}")

# KPSS test
kpss_price = kpss(df_stock['Price'], regression='c', nlags='auto')
print(f"\nKPSS Test on Prices:")
print(f"  Test Statistic = {kpss_price[0]:.4f}")
print(f"  p-value        = {kpss_price[1]:.4f}")
print(f"  Decision (α=0.05): {'Reject H0 → Non-stationary' if kpss_price[1] < 0.05 else 'Fail to reject H0 → Stationary'}")

print("\nInterpretation:")
print("  ADF: Fails to reject → suggests non-stationary (has unit root)")
print("  KPSS: Rejects H0 → confirms non-stationary")
print("  Both tests agree: the PRICE series is NON-STATIONARY.")


# ====================================================================
# B2: Differencing for Stationarity
# ====================================================================
print("\n" + "-" * 70)
print("B2: Differencing for Stationarity")
print("-" * 70)

diff_price = df_stock['Price'].diff().dropna()

# Plot 2: Differenced series
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(diff_price.index, diff_price.values, 'b-', alpha=0.7)
ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
ax.set_title('First Differenced Prices')
ax.set_xlabel('Date')
ax.set_ylabel('ΔPrice')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('B2_1_differenced.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: B2_1_differenced.png")

adf_diff = adfuller(diff_price, autolag='AIC')
print("ADF Test on Differenced Prices:")
print(f"  Test Statistic = {adf_diff[0]:.4f}")
print(f"  p-value        = {adf_diff[1]:.6f}")
print(f"  Decision: {'Reject H0 → Stationary' if adf_diff[1] < 0.05 else 'Fail to reject H0 → Non-stationary'}")

returns_pct = df_stock['Returns'].iloc[1:]
corr = np.corrcoef(diff_price.values, returns_pct.values)[0, 1]
print(f"\nCorrelation between differenced prices and returns: {corr:.6f}")
print("Differenced prices ≈ returns × price_level (approximately proportional)")

# Plot 3: ACF of differenced prices
fig, ax = plt.subplots(figsize=(10, 4))
plot_acf(diff_price, lags=30, ax=ax, title='ACF of Differenced Prices')
plt.tight_layout()
plt.savefig('B2_2_acf_diff.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: B2_2_acf_diff.png")

print("Most ACF lags within confidence bands → differenced series is approximately white noise.")


# ====================================================================
# B3: ACF and PACF
# ====================================================================
print("\n" + "-" * 70)
print("B3: ACF and PACF")
print("-" * 70)

ret = df_stock['Returns'].dropna()

# Plot 4: ACF & PACF
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(ret, lags=40, ax=axes[0], title='ACF of Returns')
plot_pacf(ret, lags=40, ax=axes[1], title='PACF of Returns', method='ywm')
plt.tight_layout()
plt.savefig('B3_acf_pacf.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: B3_acf_pacf.png")

print("ACF Interpretation:")
print("  - Most lags within confidence bands")
print("  - Returns show very weak autocorrelation (near white noise)")
print("  - ACF does not show clear exponential decay or sharp cutoff")
print("\nPACF Interpretation:")
print("  - Similar pattern; most lags insignificant")
print("  - Possible small spike at lag 1")
print("\nSuggested orders: AR(0) or AR(1) for p; MA(0) or MA(1) for q")

rho1 = ret.autocorr(lag=1)
rho5 = ret.autocorr(lag=5)
print(f"\nρ(1) = Corr(r_t, r_{{t-1}}) = {rho1:.4f}")
print(f"ρ(5) = Corr(r_t, r_{{t-5}}) = {rho5:.4f}")


# ====================================================================
# B4: AR(p) Model
# ====================================================================
print("\n" + "-" * 70)
print("B4: AR(p) Model")
print("-" * 70)

ar1 = ARIMA(ret, order=(1, 0, 0)).fit()
print("=== AR(1) Summary ===")
print(ar1.summary())

phi1 = ar1.params.iloc[1] if len(ar1.params) > 1 else ar1.params.iloc[0]
print(f"\nϕ₁ = {phi1:.4f}")
print(f"Significant? p-value = {ar1.pvalues.iloc[1]:.4f} → {'Yes' if ar1.pvalues.iloc[1] < 0.05 else 'No'}")
print(f"Sign: {'Positive (momentum)' if phi1 > 0 else 'Negative (mean-reversion)'}")
print(f"|ϕ₁| = {abs(phi1):.4f} < 1 → {'Stationary' if abs(phi1) < 1 else 'Non-stationary'}")

print("\n--- AR Model Comparison ---")
print(f"{'Model':<10} {'AIC':>10} {'BIC':>10}")
print("-" * 30)
for p in [1, 2, 3]:
    m = ARIMA(ret, order=(p, 0, 0)).fit()
    print(f"AR({p}){'':<5} {m.aic:>10.2f} {m.bic:>10.2f}")

# Plot 5: AR(1) residuals
fig, axes = plt.subplots(2, 1, figsize=(12, 6))
axes[0].plot(ar1.resid, 'b-', alpha=0.6)
axes[0].axhline(y=0, color='k', linestyle='--')
axes[0].set_title('AR(1) Residuals')
axes[0].set_ylabel('Residual')
axes[0].grid(True, alpha=0.3)

plot_acf(ar1.resid, lags=30, ax=axes[1], title='ACF of AR(1) Residuals')
plt.tight_layout()
plt.savefig('B4_ar1_residuals.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: B4_ar1_residuals.png")

print("Residuals appear to be white noise (no significant ACF spikes).")


# ====================================================================
# B5: MA(q) Model
# ====================================================================
print("\n" + "-" * 70)
print("B5: MA(q) Model")
print("-" * 70)

ma1 = ARIMA(ret, order=(0, 0, 1)).fit()
print("=== MA(1) Summary ===")
print(ma1.summary())

theta1 = ma1.params.iloc[1] if len(ma1.params) > 1 else ma1.params.iloc[0]
print(f"\nθ₁ = {theta1:.4f}")
print(f"Significant? p-value = {ma1.pvalues.iloc[1]:.4f}")
print(f"|θ₁| = {abs(theta1):.4f} < 1 → {'Invertible' if abs(theta1) < 1 else 'Not invertible'}")

print("\n--- AR(1) vs MA(1) ---")
print(f"{'Metric':<15} {'AR(1)':>12} {'MA(1)':>12}")
print("-" * 40)
print(f"{'AIC':<15} {ar1.aic:>12.2f} {ma1.aic:>12.2f}")
print(f"{'BIC':<15} {ar1.bic:>12.2f} {ma1.bic:>12.2f}")
print(f"{'Log-Lik':<15} {ar1.llf:>12.2f} {ma1.llf:>12.2f}")
better = "AR(1)" if ar1.aic < ma1.aic else "MA(1)"
print(f"\nBetter model by AIC: {better}")

# Plot 6: MA(1) residuals
fig, axes = plt.subplots(2, 1, figsize=(12, 6))
axes[0].plot(ma1.resid, 'b-', alpha=0.6)
axes[0].axhline(y=0, color='k', linestyle='--')
axes[0].set_title('MA(1) Residuals')
axes[0].grid(True, alpha=0.3)
plot_acf(ma1.resid, lags=30, ax=axes[1], title='ACF of MA(1) Residuals')
plt.tight_layout()
plt.savefig('B5_ma1_residuals.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: B5_ma1_residuals.png")


# ====================================================================
# B6: ARMA(p,q) Model
# ====================================================================
print("\n" + "-" * 70)
print("B6: ARMA(p,q) Model")
print("-" * 70)

arma11 = ARIMA(ret, order=(1, 0, 1)).fit()
print("=== ARMA(1,1) Summary ===")
print(arma11.summary())

print("\n--- Overparameterization Check ---")
print(f"ϕ₁ p-value: {arma11.pvalues.iloc[1]:.4f} → {'Significant' if arma11.pvalues.iloc[1] < 0.05 else 'Not significant'}")
print(f"θ₁ p-value: {arma11.pvalues.iloc[2]:.4f} → {'Significant' if arma11.pvalues.iloc[2] < 0.05 else 'Not significant'}")

print("\n--- Full Model Comparison ---")
print(f"{'Model':<12} {'AIC':>10} {'BIC':>10} {'Log-Lik':>10} {'Params':>8}")
print("-" * 52)
for name, m in [('AR(1)', ar1), ('MA(1)', ma1), ('ARMA(1,1)', arma11)]:
    print(f"{name:<12} {m.aic:>10.2f} {m.bic:>10.2f} {m.llf:>10.2f} {m.df_model:>8d}")


# ====================================================================
# B7: Automatic ARIMA
# ====================================================================
print("\n" + "-" * 70)
print("B7: Automatic ARIMA")
print("-" * 70)

best_aic = np.inf
best_order = None
results_table = []

for p in range(6):
    for q in range(6):
        try:
            model = ARIMA(ret, order=(p, 0, q)).fit()
            results_table.append({'p': p, 'q': q, 'AIC': model.aic, 'BIC': model.bic})
            if model.aic < best_aic:
                best_aic = model.aic
                best_order = (p, 0, q)
        except:
            continue

results_df = pd.DataFrame(results_table).sort_values('AIC').head(10)
print("Top 10 models by AIC:")
print(results_df.to_string(index=False))
print(f"\nBest model: ARIMA{best_order} with AIC = {best_aic:.2f}")

best_model = ARIMA(ret, order=best_order).fit()
print(f"\n=== Best Model ARIMA{best_order} Summary ===")
print(best_model.summary())

print(f"\nDoes auto-selection match manual? The best model is ARIMA{best_order}.")
print("Given the weak autocorrelation in returns, a simple model (like AR(1) or even")
print("ARIMA(0,0,0)) is expected to perform similarly.")


# ====================================================================
# B8: ARIMA Forecasting
# ====================================================================
print("\n" + "-" * 70)
print("B8: ARIMA Forecasting")
print("-" * 70)

forecast_result = best_model.get_forecast(steps=20)
forecast_mean = forecast_result.predicted_mean
forecast_ci = forecast_result.conf_int(alpha=0.05)

# Plot 7: Return forecast
fig, ax = plt.subplots(figsize=(12, 5))
last100 = ret.iloc[-100:]
ax.plot(last100.index, last100.values, 'b-', alpha=0.7, label='Actual Returns')

last_date = ret.index[-1]
forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=20, freq='D')

ax.plot(forecast_dates, forecast_mean.values, 'r-o', markersize=4, label='Forecast')
ax.fill_between(forecast_dates, forecast_ci.iloc[:, 0].values, forecast_ci.iloc[:, 1].values,
                color='red', alpha=0.15, label='95% PI')
ax.axvline(x=last_date, color='gray', linestyle=':', label='Forecast Start')
ax.set_title(f'ARIMA{best_order}: 20-Day Return Forecast')
ax.set_xlabel('Date')
ax.set_ylabel('Return (%)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('B8_1_return_forecast.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: B8_1_return_forecast.png")

# Plot 8: Price forecast
last_price = df_stock['Price'].iloc[-1]
cum_returns = np.cumsum(forecast_mean.values / 100)
price_forecast = last_price * np.exp(cum_returns)

cum_ret_lower = np.cumsum(forecast_ci.iloc[:, 0].values / 100)
cum_ret_upper = np.cumsum(forecast_ci.iloc[:, 1].values / 100)
price_lower = last_price * np.exp(cum_ret_lower)
price_upper = last_price * np.exp(cum_ret_upper)

fig, ax = plt.subplots(figsize=(12, 5))
last50_price = df_stock['Price'].iloc[-50:]
ax.plot(last50_price.index, last50_price.values, 'b-', label='Actual Price')
ax.plot(forecast_dates, price_forecast, 'r-o', markersize=4, label='Price Forecast')
ax.fill_between(forecast_dates, price_lower, price_upper,
                color='red', alpha=0.15, label='95% PI')
ax.axvline(x=last_date, color='gray', linestyle=':')
ax.set_title(f'ARIMA{best_order}: 20-Day Price Forecast')
ax.set_xlabel('Date')
ax.set_ylabel('Price ($)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('B8_2_price_forecast.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: B8_2_price_forecast.png")

print("\nForecast Behavior Observations:")
print(f"  - Return forecasts converge to the unconditional mean ({forecast_mean.iloc[-1]:.4f}%)")
print(f"    as horizon increases, since AR effects decay exponentially.")
print(f"  - Prediction interval width at h=1:  {(forecast_ci.iloc[0,1]-forecast_ci.iloc[0,0]):.4f}")
print(f"  - Prediction interval width at h=20: {(forecast_ci.iloc[19,1]-forecast_ci.iloc[19,0]):.4f}")
print(f"  - Intervals widen because forecast uncertainty accumulates with horizon.")
print(f"  - For near-white-noise returns, intervals widen quickly and forecasts")
print(f"    offer little improvement over the unconditional mean.")

print("\n" + "=" * 70)
print("PART B COMPLETE - All 8 plots saved!")
print("=" * 70)