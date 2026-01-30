import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit, Probit, Poisson
from statsmodels.tsa.seasonal import STL, seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("=" * 70)
print("ECON 5140 - HOMEWORK 1")
print("Part A: Generalized Linear Models")
print("Part B: Time Series Decomposition")
print("=" * 70)

# ====================================================================
# DATASET 1: CUSTOMER PURCHASE DATA (for GLM analysis)
# ====================================================================
print("\n" + "=" * 70)
print("DATASET 1: Customer Purchase Behavior")
print("=" * 70)

n_customers = 1000

age = np.random.normal(35, 10, n_customers)
income = np.random.normal(50, 15, n_customers)
time_on_site = np.random.gamma(2, 3, n_customers)

z = -3 + 0.05*age + 0.04*income + 0.15*time_on_site + np.random.normal(0, 1, n_customers)
purchase = (z > 0).astype(int)

df_customers = pd.DataFrame({
    'Age': age,
    'Income': income,
    'TimeOnSite': time_on_site,
    'Purchase': purchase
})

print(f"Number of customers: {len(df_customers)}")
print(f"Purchase rate: {df_customers['Purchase'].mean():.2%}")
print(f"\nFirst 5 rows:")
print(df_customers.head())

# ====================================================================
# DATASET 2: E-COMMERCE SALES TIME SERIES
# ====================================================================
print("\n" + "=" * 70)
print("DATASET 2: E-commerce Daily Sales")
print("=" * 70)

dates = pd.date_range('2024-01-01', '2025-12-31', freq='D')
n_days = len(dates)
t = np.arange(n_days)

trend = 1000 + 2*t + 0.01*t**2
yearly_seasonal = 200 * np.sin(2*np.pi*t/365) + 150 * np.cos(2*np.pi*t/365)
weekly_seasonal = 100 * np.sin(2*np.pi*t/7)

special_events = np.zeros(n_days)
for year in [2024, 2025]:
    bf_date = pd.Timestamp(f'{year}-11-24')
    bf_idx = (dates == bf_date)
    special_events[bf_idx] = 800
    xmas_idx = (dates >= f'{year}-12-20') & (dates <= f'{year}-12-25')
    special_events[xmas_idx] = 400

noise = np.random.normal(0, 50, n_days)
sales = trend + yearly_seasonal + weekly_seasonal + special_events + noise
sales = np.maximum(sales, 0)

df_sales = pd.DataFrame({
    'Date': dates,
    'Sales': sales,
    'DayOfWeek': dates.dayofweek,
    'Month': dates.month,
    'IsWeekend': dates.dayofweek >= 5
})
df_sales.set_index('Date', inplace=True)

print(f"Date range: {df_sales.index[0].date()} to {df_sales.index[-1].date()}")
print(f"Number of days: {len(df_sales)}")
print(f"\nSales Statistics:")
print(df_sales['Sales'].describe())

# ====================================================================
# PART A: GENERALIZED LINEAR MODELS
# ====================================================================
print("\n" + "=" * 70)
print("PART A: GENERALIZED LINEAR MODELS")
print("=" * 70)

# --------------------------------------------------------------------
# A1: Exploratory Data Analysis
# --------------------------------------------------------------------
print("\n" + "-" * 70)
print("A1: Exploratory Data Analysis")
print("-" * 70)

# 1. Box plots
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

df_customers.boxplot(column='Age', by='Purchase', ax=axes[0])
axes[0].set_title('Age by Purchase')
axes[0].set_xlabel('Purchase (0=No, 1=Yes)')
axes[0].set_ylabel('Age')

df_customers.boxplot(column='Income', by='Purchase', ax=axes[1])
axes[1].set_title('Income by Purchase')
axes[1].set_xlabel('Purchase (0=No, 1=Yes)')
axes[1].set_ylabel('Income (thousands)')

df_customers.boxplot(column='TimeOnSite', by='Purchase', ax=axes[2])
axes[2].set_title('TimeOnSite by Purchase')
axes[2].set_xlabel('Purchase (0=No, 1=Yes)')
axes[2].set_ylabel('Time (minutes)')

plt.suptitle("")
plt.tight_layout()
plt.show()

# 2. Mean values by group
means = df_customers.groupby('Purchase')[['Age', 'Income', 'TimeOnSite']].mean()
print("\nMean values by group (0=Non-purchasers, 1=Purchasers):")
print(means)

# 3. Correlation matrix heatmap
corr = df_customers[['Age', 'Income', 'TimeOnSite']].corr()
plt.figure(figsize=(6, 5))
plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
plt.yticks(range(len(corr.index)), corr.index)
plt.colorbar()
for i in range(len(corr.index)):
    for j in range(len(corr.columns)):
        plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha='center', va='center', fontsize=12)
plt.title("Correlation Matrix (Features)")
plt.tight_layout()
plt.show()

# --------------------------------------------------------------------
# A2: Linear Probability Model (LPM)
# --------------------------------------------------------------------
print("\n" + "-" * 70)
print("A2: Linear Probability Model")
print("-" * 70)

# 1. Fit OLS model
X = sm.add_constant(df_customers[['Age', 'Income', 'TimeOnSite']])
y = df_customers['Purchase']
lpm_model = sm.OLS(y, X).fit()

# 2. Print summary
print(lpm_model.summary())

# 3. Predicted probabilities
lpm_pred = lpm_model.predict(X)
below_0 = (lpm_pred < 0).sum()
above_1 = (lpm_pred > 1).sum()
invalid = below_0 + above_1

print(f"\n--- Prediction Analysis ---")
print(f"Predictions < 0: {below_0} ({below_0/len(lpm_pred)*100:.2f}%)")
print(f"Predictions > 1: {above_1} ({above_1/len(lpm_pred)*100:.2f}%)")
print(f"Total invalid:   {invalid} ({invalid/len(lpm_pred)*100:.2f}%)")

# 4. Histogram
plt.figure(figsize=(10, 5))
plt.hist(lpm_pred, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
plt.axvline(0, color='red', linestyle='--', linewidth=2, label='Lower bound (0)')
plt.axvline(1, color='red', linestyle='--', linewidth=2, label='Upper bound (1)')
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.title('LPM Predicted Probabilities')
plt.legend()
plt.tight_layout()
plt.show()

# --------------------------------------------------------------------
# A3: Logistic Regression
# --------------------------------------------------------------------
print("\n" + "-" * 70)
print("A3: Logistic Regression")
print("-" * 70)

# 1. Fit logistic regression
logit_model = sm.Logit(y, X).fit()

# 2. Print summary
print(logit_model.summary())

# Coefficients and odds ratios
print("\n--- Coefficients and Odds Ratios ---")
coef_table = pd.DataFrame({
    'Coefficient': logit_model.params,
    'Std Error': logit_model.bse,
    'Odds Ratio': np.exp(logit_model.params),
    'p-value': logit_model.pvalues
})
print(coef_table.round(4))

# 3. Interpretation
print("\n--- Interpretation ---")
print(f"Age: A 1-year increase in age increases log-odds by {logit_model.params['Age']:.4f}")
print(f"     Odds ratio: {np.exp(logit_model.params['Age']):.4f}")
print(f"Income: A $1k increase in income increases log-odds by {logit_model.params['Income']:.4f}")
print(f"     Odds ratio: {np.exp(logit_model.params['Income']):.4f}")
print(f"TimeOnSite: A 1-min increase increases log-odds by {logit_model.params['TimeOnSite']:.4f}")
print(f"     Odds ratio: {np.exp(logit_model.params['TimeOnSite']):.4f}")

# 4. Predicted probabilities
logit_pred = logit_model.predict(X)
print(f"\nAll predictions in [0, 1]: {((logit_pred >= 0) & (logit_pred <= 1)).all()}")
print(f"Min prediction: {logit_pred.min():.4f}")
print(f"Max prediction: {logit_pred.max():.4f}")

plt.figure(figsize=(10, 5))
plt.hist(logit_pred, bins=30, edgecolor='black', alpha=0.7, color='forestgreen')
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.title('Logistic Regression Predicted Probabilities')
plt.tight_layout()
plt.show()

# --------------------------------------------------------------------
# A4: Prediction for New Customers
# --------------------------------------------------------------------
print("\n" + "-" * 70)
print("A4: Predictions for New Customers")
print("-" * 70)

new_customers = pd.DataFrame({
    'Age': [25, 35, 45, 55],
    'Income': [30, 50, 70, 90],
    'TimeOnSite': [2, 5, 8, 10]
})

# 1. Predict
X_new = sm.add_constant(new_customers)
new_pred = logit_model.predict(X_new)

# 2. Formatted table
result_df = new_customers.copy()
result_df['Pred_Probability'] = new_pred.round(4)
result_df['Classification'] = (new_pred > 0.5).astype(int)
result_df['Label'] = result_df['Classification'].map({0: 'No Purchase', 1: 'Purchase'})

print("\n--- Prediction Results ---")
print(result_df.to_string(index=False))

# 3. Most likely customer
max_idx = new_pred.argmax()
print(f"\nMost likely to purchase: Customer {max_idx + 1}")
print(f"  Age: {new_customers.iloc[max_idx]['Age']}")
print(f"  Income: ${new_customers.iloc[max_idx]['Income']}k")
print(f"  TimeOnSite: {new_customers.iloc[max_idx]['TimeOnSite']} min")
print(f"  Probability: {new_pred[max_idx]:.2%}")
print("  Reason: Highest income and longest time on site - both strong positive predictors")

# ====================================================================
# PART B: TIME SERIES ANALYSIS
# ====================================================================
print("\n" + "=" * 70)
print("PART B: TIME SERIES ANALYSIS")
print("=" * 70)

# --------------------------------------------------------------------
# B1: Time Series Visualization
# --------------------------------------------------------------------
print("\n" + "-" * 70)
print("B1: Time Series Visualization")
print("-" * 70)

# 1. Time series plot
plt.figure(figsize=(14, 5))
plt.plot(df_sales.index, df_sales['Sales'], linewidth=0.8)
plt.xlabel('Date')
plt.ylabel('Sales ($)')
plt.title('Daily E-commerce Sales (2024-2025)')
plt.tight_layout()
plt.show()

# 2. Seasonal subseries plots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
day_data = [df_sales[df_sales['DayOfWeek'] == i]['Sales'] for i in range(7)]
axes[0].boxplot(day_data, labels=day_names)
axes[0].set_title('Sales by Day of Week')
axes[0].set_xlabel('Day')
axes[0].set_ylabel('Sales')

month_data = [df_sales[df_sales['Month'] == i]['Sales'] for i in range(1, 13)]
axes[1].boxplot(month_data, labels=range(1, 13))
axes[1].set_title('Sales by Month')
axes[1].set_xlabel('Month')
axes[1].set_ylabel('Sales')

plt.tight_layout()
plt.show()

# 3. Mean by day and month
print("\nMean Sales by Day of Week:")
for i, day in enumerate(day_names):
    mean_val = df_sales[df_sales['DayOfWeek'] == i]['Sales'].mean()
    print(f"  {day}: ${mean_val:.2f}")

print("\nMean Sales by Month:")
for m in range(1, 13):
    mean_val = df_sales[df_sales['Month'] == m]['Sales'].mean()
    print(f"  Month {m:2d}: ${mean_val:.2f}")

# 4. Patterns
print("\n--- Observed Patterns ---")
print("1. Clear upward TREND over the 2-year period")
print("2. WEEKLY seasonality: variation across weekdays")
print("3. YEARLY seasonality: Nov-Dec shows higher sales (holidays)")
print("4. Spikes visible around Black Friday and Christmas")

# --------------------------------------------------------------------
# B2: Stationarity Assessment
# --------------------------------------------------------------------
print("\n" + "-" * 70)
print("B2: Stationarity Check")
print("-" * 70)

# 1. Plot with rolling statistics
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

axes[0].plot(df_sales.index, df_sales['Sales'], linewidth=0.8)
axes[0].set_title('Original Series')
axes[0].set_ylabel('Sales')

rolling_mean = df_sales['Sales'].rolling(window=30).mean()
axes[1].plot(df_sales.index, rolling_mean, color='orange')
axes[1].set_title('30-day Rolling Mean')
axes[1].set_ylabel('Mean')

rolling_std = df_sales['Sales'].rolling(window=30).std()
axes[2].plot(df_sales.index, rolling_std, color='green')
axes[2].set_title('30-day Rolling Standard Deviation')
axes[2].set_ylabel('Std Dev')

plt.tight_layout()
plt.show()

# 2 & 3. Compare periods
first_6m = df_sales['Sales'][:180]
last_6m = df_sales['Sales'][-180:]

print("\n--- Stationarity Assessment ---")
print(f"First 6 months: Mean = ${first_6m.mean():.2f}, Std = ${first_6m.std():.2f}")
print(f"Last 6 months:  Mean = ${last_6m.mean():.2f}, Std = ${last_6m.std():.2f}")
print(f"\nMean change: ${last_6m.mean() - first_6m.mean():.2f}")
print("\nConclusion: The series is NOT STATIONARY")
print("  - Mean increases significantly over time (upward trend)")
print("  - Rolling mean shows clear upward movement")

# --------------------------------------------------------------------
# B3: Autocorrelation Analysis
# --------------------------------------------------------------------
print("\n" + "-" * 70)
print("B3: Autocorrelation Function")
print("-" * 70)

# 1. ACF plot
fig, ax = plt.subplots(figsize=(12, 5))
plot_acf(df_sales['Sales'], lags=60, ax=ax)
plt.title('Autocorrelation Function (ACF)')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.tight_layout()
plt.show()

# 2. Manual autocorrelation
sales_arr = df_sales['Sales'].values
acf_1 = np.corrcoef(sales_arr[:-1], sales_arr[1:])[0, 1]
acf_7 = np.corrcoef(sales_arr[:-7], sales_arr[7:])[0, 1]
acf_30 = np.corrcoef(sales_arr[:-30], sales_arr[30:])[0, 1]

print("\n--- Manual Autocorrelation Calculations ---")
print(f"Lag 1  (yesterday):  {acf_1:.4f}")
print(f"Lag 7  (last week):  {acf_7:.4f}")
print(f"Lag 30 (last month): {acf_30:.4f}")

# 3. Interpretation
print("\n--- Interpretation ---")
print("1. Very high autocorrelation at all lags (due to strong trend)")
print("2. ACF decays slowly - characteristic of non-stationary data")
print("3. Slight bumps at lag 7, 14, 21... indicate weekly pattern")
print("4. Persistence: Today's sales highly correlated with recent past")

# --------------------------------------------------------------------
# B4: STL Decomposition
# --------------------------------------------------------------------
print("\n" + "-" * 70)
print("B4: STL Decomposition")
print("-" * 70)

# 1. Apply STL
stl = STL(df_sales['Sales'], seasonal=7, robust=True)
result = stl.fit()

# 2. Plot components
fig, axes = plt.subplots(4, 1, figsize=(14, 12))

axes[0].plot(df_sales.index, result.observed, linewidth=0.8)
axes[0].set_title('Observed')
axes[0].set_ylabel('Sales')

axes[1].plot(df_sales.index, result.trend, color='orange', linewidth=1.5)
axes[1].set_title('Trend')
axes[1].set_ylabel('Sales')

axes[2].plot(df_sales.index, result.seasonal, color='green', linewidth=0.8)
axes[2].set_title('Seasonal (Weekly)')
axes[2].set_ylabel('Sales')

axes[3].plot(df_sales.index, result.resid, color='red', linewidth=0.8)
axes[3].set_title('Remainder')
axes[3].set_ylabel('Sales')
axes[3].axhline(0, color='black', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.show()

# 3. Analysis
print("\n--- Component Analysis ---")
print("TREND:")
print(f"  Start: ${result.trend.dropna().iloc[0]:.2f}")
print(f"  End:   ${result.trend.dropna().iloc[-1]:.2f}")
print("  Pattern: Quadratic upward trend (accelerating growth)")

print("\nSEASONAL (Weekly):")
print(f"  Range: {result.seasonal.min():.2f} to {result.seasonal.max():.2f}")
print("  Pattern: 7-day repeating cycle")

print("\nREMAINDER:")
print(f"  Mean: {result.resid.mean():.2f}")
print(f"  Std:  {result.resid.std():.2f}")
print("  Contains: Random noise + special events (Black Friday, Christmas)")

# --------------------------------------------------------------------
# B5: Remainder Diagnostics
# --------------------------------------------------------------------
print("\n" + "-" * 70)
print("B5: Remainder Analysis")
print("-" * 70)

# 1. Extract remainder
remainder = result.resid.dropna()

# 2. Diagnostic plots
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(remainder.index, remainder, linewidth=0.8, color='red')
axes[0].axhline(0, color='black', linestyle='--')
axes[0].set_title('Remainder Time Series')
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Remainder')

axes[1].hist(remainder, bins=40, edgecolor='black', alpha=0.7, color='steelblue')
axes[1].axvline(0, color='red', linestyle='--', linewidth=2)
axes[1].set_title('Remainder Histogram')
axes[1].set_xlabel('Remainder')
axes[1].set_ylabel('Frequency')

plot_acf(remainder, lags=30, ax=axes[2])
axes[2].set_title('Remainder ACF')

plt.tight_layout()
plt.show()

# 3. Statistical tests
print("\n--- Remainder Statistics ---")
print(f"Mean: {remainder.mean():.4f} (should be ≈ 0)")
print(f"Std:  {remainder.std():.4f}")

# Normality test
stat, p_val = stats.shapiro(remainder[:500])
print(f"\nShapiro-Wilk Normality Test:")
print(f"  Statistic: {stat:.4f}")
print(f"  p-value:   {p_val:.4f}")
print(f"  Normal:    {'Yes' if p_val > 0.05 else 'No'}")

# 4. Identify outliers
std_rem = remainder.std()
threshold = 3 * std_rem
outliers = remainder[np.abs(remainder) > threshold]

print(f"\n--- Outliers (|remainder| > 3×std = {threshold:.2f}) ---")
print(f"Number of outliers: {len(outliers)}")
print("\nOutlier dates:")
for date, val in outliers.items():
    print(f"  {date.date()}: {val:.2f}")

print("\n--- Investigation ---")
print("These outliers correspond to special events:")
print("  - Black Friday (Nov 24): Large positive spike")
print("  - Christmas period (Dec 20-25): Elevated sales")
print("The STL decomposition successfully isolated these events in the remainder.")

print("\n" + "=" * 70)
print("HOMEWORK COMPLETE")
print("=" * 70)