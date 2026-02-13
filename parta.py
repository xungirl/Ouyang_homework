import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ====================================================================
# DATASET 1: Monthly Retail Sales
# ====================================================================
print("\n" + "=" * 70)
print("DATASET 1: Monthly Retail Sales")
print("=" * 70)

dates = pd.date_range('2021-01-01', '2024-12-31', freq='MS')
n_months = len(dates)
t = np.arange(n_months)

trend = 1000 + 15 * t
yearly_seasonal = 300 * np.sin(2 * np.pi * t / 12) + 200 * np.cos(2 * np.pi * t / 12)

holiday_effect = np.zeros(n_months)
for year in range(4):
    nov_idx = year * 12 + 10
    dec_idx = year * 12 + 11
    if nov_idx < n_months:
        holiday_effect[nov_idx] = 400
    if dec_idx < n_months:
        holiday_effect[dec_idx] = 600

noise = np.random.normal(0, 80, n_months)
sales = trend + yearly_seasonal + holiday_effect + noise
sales = np.maximum(sales, 0)

df_sales = pd.DataFrame({
    'Date': dates, 'Sales': sales,
    'Month': dates.month, 'Year': dates.year, 'Time': t
})
df_sales.set_index('Date', inplace=True)

print(f"Date range: {df_sales.index[0].date()} to {df_sales.index[-1].date()}")
print(f"Number of months: {len(df_sales)}")
print(f"\nSales Statistics:")
print(df_sales['Sales'].describe())


# ====================================================================
# A1: Exploratory Visualization
# ====================================================================
print("\n" + "-" * 70)
print("A1: Exploratory Visualization")
print("-" * 70)

# Plot 1: Time series
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df_sales.index, df_sales['Sales'], 'b-o', markersize=3, label='Monthly Sales')
ax.set_title('Monthly Retail Sales (2021-2024)', fontsize=14)
ax.set_xlabel('Date')
ax.set_ylabel('Sales ($)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('A1_1_timeseries.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: A1_1_timeseries.png")

# Plot 2: Box plot by month
fig, ax = plt.subplots(figsize=(10, 5))
month_labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
df_sales.boxplot(column='Sales', by='Month', ax=ax)
ax.set_xticklabels(month_labels)
ax.set_title('Sales Distribution by Month', fontsize=14)
ax.set_xlabel('Month')
ax.set_ylabel('Sales ($)')
plt.suptitle('')
plt.tight_layout()
plt.savefig('A1_2_boxplot.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: A1_2_boxplot.png")

monthly_avg = df_sales.groupby('Month')['Sales'].mean()
print("\nAverage Sales by Month:")
print(monthly_avg.round(1))
print(f"\nHighest sales months: Nov ({monthly_avg[11]:.1f}) and Dec ({monthly_avg[12]:.1f})")

# Plot 3: Decomposition
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

axes[0].plot(df_sales.index, df_sales['Sales'], 'b-o', markersize=3)
axes[0].set_title('Original Series')
axes[0].set_ylabel('Sales')
axes[0].grid(True, alpha=0.3)

ma12 = df_sales['Sales'].rolling(window=12, center=True).mean()
axes[1].plot(df_sales.index, ma12, 'r-', linewidth=2)
axes[1].set_title('12-Month Moving Average (Trend)')
axes[1].set_ylabel('Sales')
axes[1].grid(True, alpha=0.3)

detrended = df_sales['Sales'] - ma12
axes[2].plot(df_sales.index, detrended, 'g-o', markersize=3)
axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.5)
axes[2].set_title('Detrended Series (Original - Trend)')
axes[2].set_ylabel('Sales')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('A1_3_decomposition.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: A1_3_decomposition.png")


# ====================================================================
# A2: Regression Forecasting
# ====================================================================
print("\n" + "-" * 70)
print("A2: Regression Forecasting")
print("-" * 70)

month_dummies = pd.get_dummies(df_sales['Month'], prefix='Month', drop_first=True, dtype=int)

X_reg = pd.concat([df_sales[['Time']], month_dummies], axis=1)
X_reg = sm.add_constant(X_reg)
y_reg = df_sales['Sales']

reg_model = sm.OLS(y_reg, X_reg).fit()
print(reg_model.summary())

print("\n--- Interpretation ---")
print(f"Monthly trend (β_Time): {reg_model.params['Time']:.2f} (sales increase ~${reg_model.params['Time']:.2f}/month)")

seasonal_params = reg_model.params[[c for c in reg_model.params.index if c.startswith('Month_')]]
max_month = seasonal_params.idxmax()
print(f"Largest seasonal effect: {max_month} = {seasonal_params[max_month]:.2f}")

print(f"\nSignificant coefficients (p < 0.05):")
sig = reg_model.pvalues[reg_model.pvalues < 0.05]
for name, pval in sig.items():
    print(f"  {name}: coef={reg_model.params[name]:.2f}, p={pval:.4f}")

fitted_reg = reg_model.fittedvalues
r2 = reg_model.rsquared
rmse_reg = np.sqrt(mean_squared_error(y_reg, fitted_reg))
mae_reg = mean_absolute_error(y_reg, fitted_reg)
print(f"\nR² = {r2:.4f}")
print(f"RMSE = {rmse_reg:.2f}")
print(f"MAE = {mae_reg:.2f}")


# ====================================================================
# A3: Fourier Seasonality
# ====================================================================
print("\n" + "-" * 70)
print("A3: Fourier Seasonality")
print("-" * 70)

X_fourier = pd.DataFrame({'Time': df_sales['Time']})
for k in [1, 2]:
    X_fourier[f'sin_{k}'] = np.sin(2 * np.pi * k * t / 12)
    X_fourier[f'cos_{k}'] = np.cos(2 * np.pi * k * t / 12)
X_fourier = sm.add_constant(X_fourier)
X_fourier.index = df_sales.index

fourier_model = sm.OLS(y_reg, X_fourier).fit()
print(fourier_model.summary())

rmse_fourier = np.sqrt(mean_squared_error(y_reg, fourier_model.fittedvalues))
print("\n--- Model Comparison ---")
print(f"{'Metric':<20} {'Dummy Model':>15} {'Fourier Model':>15}")
print("-" * 50)
print(f"{'R²':<20} {reg_model.rsquared:>15.4f} {fourier_model.rsquared:>15.4f}")
print(f"{'AIC':<20} {reg_model.aic:>15.2f} {fourier_model.aic:>15.2f}")
print(f"{'# Parameters':<20} {reg_model.df_model+1:>15.0f} {fourier_model.df_model+1:>15.0f}")
print(f"{'RMSE':<20} {rmse_reg:>15.2f} {rmse_fourier:>15.2f}")

# Plot 4: Dummy vs Fourier
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df_sales.index, df_sales['Sales'], 'k-o', markersize=3, label='Actual', alpha=0.7)
ax.plot(df_sales.index, fitted_reg, 'b--', linewidth=2, label=f'Dummy Reg (R²={reg_model.rsquared:.3f})')
ax.plot(df_sales.index, fourier_model.fittedvalues, 'r--', linewidth=2, label=f'Fourier Reg (R²={fourier_model.rsquared:.3f})')
ax.set_title('Regression Models: Dummy vs Fourier Seasonality')
ax.set_xlabel('Date')
ax.set_ylabel('Sales')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('A3_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: A3_comparison.png")

print("\nPreference: The Dummy variable model has higher R² and better captures")
print("the holiday spikes (Nov/Dec), since Fourier terms produce smooth seasonal")
print("patterns. However, Fourier uses fewer parameters (6 vs 13), which is more")
print("parsimonious. For data with sharp seasonal spikes, dummies are preferred.")


# ====================================================================
# A4: Simple Exponential Smoothing (SES)
# ====================================================================
print("\n" + "-" * 70)
print("A4: Simple Exponential Smoothing")
print("-" * 70)

train_sales = df_sales.iloc[:36]
test_sales = df_sales.iloc[36:]

seasonal_avg = train_sales.groupby('Month')['Sales'].mean()
grand_mean = train_sales['Sales'].mean()
seasonal_indices = seasonal_avg - grand_mean

df_sales['Seasonal_Index'] = df_sales['Month'].map(seasonal_indices)
df_sales['Deseasonalized'] = df_sales['Sales'] - df_sales['Seasonal_Index']

train_deseas = df_sales['Deseasonalized'].iloc[:36]
test_deseas = df_sales['Deseasonalized'].iloc[36:]

ses_model = SimpleExpSmoothing(train_deseas).fit(optimized=True)

alpha_val = ses_model.params['smoothing_level']
print(f"Optimized α = {alpha_val:.4f}")
if alpha_val > 0.5:
    print("α > 0.5: Series reacts quickly to recent changes; recent obs have more weight.")
else:
    print("α < 0.5: Series is relatively smooth; historical values carry significant weight.")

ses_forecast_deseas = ses_model.forecast(12)
test_seasonal = df_sales['Seasonal_Index'].iloc[36:].values
ses_forecast = ses_forecast_deseas.values + test_seasonal

rmse_ses = np.sqrt(mean_squared_error(test_sales['Sales'], ses_forecast))
mae_ses = mean_absolute_error(test_sales['Sales'], ses_forecast)
print(f"\nSES Forecast Accuracy (holdout):")
print(f"  RMSE = {rmse_ses:.2f}")
print(f"  MAE  = {mae_ses:.2f}")


# ====================================================================
# A5: Holt's Linear Trend Method
# ====================================================================
print("\n" + "-" * 70)
print("A5: Holt's Method")
print("-" * 70)

holt_model = ExponentialSmoothing(
    train_deseas, trend='add', seasonal=None
).fit(optimized=True)

print(f"Optimized α (level)  = {holt_model.params['smoothing_level']:.4f}")
print(f"Optimized β (trend)  = {holt_model.params['smoothing_trend']:.4f}")

print(f"\nFinal level (ℓ_T) = {holt_model.level.iloc[-1]:.2f}")
print(f"Final trend (b_T) = {holt_model.trend.iloc[-1]:.2f}")

holt_forecast_deseas = holt_model.forecast(12)
holt_forecast = holt_forecast_deseas.values + test_seasonal

rmse_holt = np.sqrt(mean_squared_error(test_sales['Sales'], holt_forecast))
mae_holt = mean_absolute_error(test_sales['Sales'], holt_forecast)
print(f"\nHolt's Forecast Accuracy (holdout):")
print(f"  RMSE = {rmse_holt:.2f}")
print(f"  MAE  = {mae_holt:.2f}")

# Plot 5: SES vs Holt
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(test_sales.index, test_sales['Sales'], 'k-o', markersize=5, label='Actual')
ax.plot(test_sales.index, ses_forecast, 'b--s', markersize=5, label=f'SES (RMSE={rmse_ses:.1f})')
ax.plot(test_sales.index, holt_forecast, 'r--^', markersize=5, label=f"Holt's (RMSE={rmse_holt:.1f})")
ax.set_title("SES vs Holt's Method: 12-Month Forecast")
ax.set_xlabel('Date')
ax.set_ylabel('Sales')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('A5_ses_vs_holt.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: A5_ses_vs_holt.png")

print("\nHolt's method captures the upward trend, so its forecasts increase over")
print("time, while SES produces a flat forecast. Holt's is better for trending data.")


# ====================================================================
# A6: Holt-Winters Seasonal Method
# ====================================================================
print("\n" + "-" * 70)
print("A6: Holt-Winters Method")
print("-" * 70)

hw_model = ExponentialSmoothing(
    train_sales['Sales'],
    trend='add', seasonal='add', seasonal_periods=12
).fit(optimized=True)

print(f"Optimized α (level)    = {hw_model.params['smoothing_level']:.4f}")
print(f"Optimized β (trend)    = {hw_model.params['smoothing_trend']:.4f}")
print(f"Optimized γ (seasonal) = {hw_model.params['smoothing_seasonal']:.4f}")

print(f"\nFinal level  = {hw_model.level.iloc[-1]:.2f}")
print(f"Final trend  = {hw_model.trend.iloc[-1]:.2f}")
print("\nSeasonal Indices:")
for i, s in enumerate(hw_model.season.iloc[-12:]):
    print(f"  Month {i+1:2d}: {s:+.2f}")

# Plot 6: HW decomposition
fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
axes[0].plot(train_sales.index, train_sales['Sales'], 'b-o', markersize=3)
axes[0].set_title('Original')
axes[0].set_ylabel('Sales')

axes[1].plot(train_sales.index, hw_model.level, 'r-')
axes[1].set_title('Level')
axes[1].set_ylabel('Level')

axes[2].plot(train_sales.index, hw_model.trend, 'g-')
axes[2].set_title('Trend')
axes[2].set_ylabel('Trend')

axes[3].plot(train_sales.index, hw_model.season.iloc[:36], 'm-')
axes[3].set_title('Seasonal')
axes[3].set_ylabel('Seasonal')

for ax in axes:
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('A6_1_hw_decomposition.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: A6_1_hw_decomposition.png")

# Forecast & compare all models
hw_forecast = hw_model.forecast(12)
rmse_hw = np.sqrt(mean_squared_error(test_sales['Sales'], hw_forecast))
mae_hw = mean_absolute_error(test_sales['Sales'], hw_forecast)

X_test = pd.concat([test_sales[['Time']],
                     pd.get_dummies(test_sales['Month'], prefix='Month', drop_first=True, dtype=int)], axis=1)
X_test = sm.add_constant(X_test)
for col in X_reg.columns:
    if col not in X_test.columns:
        X_test[col] = 0
X_test = X_test[X_reg.columns]
reg_forecast = reg_model.predict(X_test)
rmse_regf = np.sqrt(mean_squared_error(test_sales['Sales'], reg_forecast))
mae_regf = mean_absolute_error(test_sales['Sales'], reg_forecast)

print("\n--- Forecast Accuracy Comparison (Holdout Period) ---")
print(f"{'Model':<25} {'RMSE':>10} {'MAE':>10}")
print("-" * 45)
print(f"{'Regression (Dummy)':<25} {rmse_regf:>10.2f} {mae_regf:>10.2f}")
print(f"{'SES + Reseasonalize':<25} {rmse_ses:>10.2f} {mae_ses:>10.2f}")
print(f"{'Holt + Reseasonalize':<25} {rmse_holt:>10.2f} {mae_holt:>10.2f}")
print(f"{'Holt-Winters (Additive)':<25} {rmse_hw:>10.2f} {mae_hw:>10.2f}")

# Plot 7: All models comparison
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df_sales.index, df_sales['Sales'], 'k-o', markersize=3, label='Actual', alpha=0.6)
ax.plot(test_sales.index, reg_forecast, 'b--', linewidth=2, label=f'Regression (RMSE={rmse_regf:.1f})')
ax.plot(test_sales.index, ses_forecast, 'g--', linewidth=2, label=f'SES (RMSE={rmse_ses:.1f})')
ax.plot(test_sales.index, holt_forecast, 'orange', linestyle='--', linewidth=2, label=f"Holt (RMSE={rmse_holt:.1f})")
ax.plot(test_sales.index, hw_forecast, 'r-', linewidth=2.5, label=f'Holt-Winters (RMSE={rmse_hw:.1f})')
ax.axvline(x=train_sales.index[-1], color='gray', linestyle=':', label='Train/Test Split')
ax.set_title('All Models: 12-Month Forecast Comparison')
ax.set_xlabel('Date')
ax.set_ylabel('Sales')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('A6_2_all_models.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: A6_2_all_models.png")

print("\n" + "=" * 70)
print("PART A COMPLETE - All 7 plots saved!")
print("=" * 70)