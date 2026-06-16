"""04_linear_models.py - Simple Linear, Multiple Linear Regression + Model Assumptions"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

sns.set_style("whitegrid")

PROCESSED_DIR = "data/processed"
MODELS_DIR = "outputs/models"
FIGURES_DIR = "outputs/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

print("=" * 80)
print("1. VERİ YÜKLENİYOR")
print("=" * 80)

X_train = np.load(f"{PROCESSED_DIR}/X_train.npy")
X_val = np.load(f"{PROCESSED_DIR}/X_val.npy")
X_test = np.load(f"{PROCESSED_DIR}/X_test.npy")
y_train = np.load(f"{PROCESSED_DIR}/y_train.npy")
y_val = np.load(f"{PROCESSED_DIR}/y_val.npy")
y_test = np.load(f"{PROCESSED_DIR}/y_test.npy")

X_train_raw = pd.read_csv(f"{PROCESSED_DIR}/X_train_raw.csv")
X_test_raw = pd.read_csv(f"{PROCESSED_DIR}/X_test_raw.csv")

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

print("\n" + "=" * 80)
print("2. SIMPLE LINEAR REGRESSION (experience_years -> salary)")
print("=" * 80)

X_train_simple = X_train_raw[['experience_years']].values
X_test_simple = X_test_raw[['experience_years']].values

simple_lr = LinearRegression()
simple_lr.fit(X_train_simple, y_train)

y_pred_simple_test = simple_lr.predict(X_test_simple)

simple_test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_simple_test))
simple_test_r2 = r2_score(y_test, y_pred_simple_test)

print(f"Intercept (β₀): {simple_lr.intercept_:,.2f}")
print(f"Coefficient (β₁): {simple_lr.coef_[0]:,.2f}")
print(f"Denklem: salary = {simple_lr.intercept_:,.2f} + {simple_lr.coef_[0]:,.2f} * experience_years")
print(f"\nTest RMSE: {simple_test_rmse:,.2f}")
print(f"Test R²: {simple_test_r2:.4f}")

plt.figure(figsize=(10, 6))
sample_idx = np.random.choice(len(X_test_simple), 3000, replace=False)
plt.scatter(X_test_simple[sample_idx], y_test[sample_idx],
            alpha=0.3, s=10, color='steelblue', label='Gerçek')
x_line = np.linspace(X_test_simple.min(), X_test_simple.max(), 100).reshape(-1, 1)
y_line = simple_lr.predict(x_line)
plt.plot(x_line, y_line, color='red', linewidth=2,
         label=f'Tahmin doğrusu (R²={simple_test_r2:.3f})')
plt.xlabel('experience_years')
plt.ylabel('salary')
plt.title('Simple Linear Regression: experience_years -> salary', fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/07_simple_linear.png", dpi=100, bbox_inches='tight')
plt.close()
print(f"Kaydedildi: {FIGURES_DIR}/07_simple_linear.png")

print("\n" + "=" * 80)
print("3. MULTIPLE LINEAR REGRESSION (tüm özellikler)")
print("=" * 80)

multi_lr = LinearRegression()
multi_lr.fit(X_train, y_train)

y_pred_train = multi_lr.predict(X_train)
y_pred_val = multi_lr.predict(X_val)
y_pred_test = multi_lr.predict(X_test)

train_mae = mean_absolute_error(y_train, y_pred_train)
val_mae = mean_absolute_error(y_val, y_pred_val)
test_mae = mean_absolute_error(y_test, y_pred_test)

train_mse = mean_squared_error(y_train, y_pred_train)
val_mse = mean_squared_error(y_val, y_pred_val)
test_mse = mean_squared_error(y_test, y_pred_test)

train_rmse = np.sqrt(train_mse)
val_rmse = np.sqrt(val_mse)
test_rmse = np.sqrt(test_mse)

train_r2 = r2_score(y_train, y_pred_train)
val_r2 = r2_score(y_val, y_pred_val)
test_r2 = r2_score(y_test, y_pred_test)

print(f"Intercept: {multi_lr.intercept_:,.2f}")
print(f"\n{'Metric':<10}{'Train':>15}{'Val':>15}{'Test':>15}")
print(f"{'MAE':<10}{train_mae:>15,.2f}{val_mae:>15,.2f}{test_mae:>15,.2f}")
print(f"{'MSE':<10}{train_mse:>15,.2f}{val_mse:>15,.2f}{test_mse:>15,.2f}")
print(f"{'RMSE':<10}{train_rmse:>15,.2f}{val_rmse:>15,.2f}{test_rmse:>15,.2f}")
print(f"{'R²':<10}{train_r2:>15.4f}{val_r2:>15.4f}{test_r2:>15.4f}")

joblib.dump(simple_lr, f"{MODELS_DIR}/simple_linear.pkl")
joblib.dump(multi_lr, f"{MODELS_DIR}/multiple_linear.pkl")

print("\n" + "=" * 80)
print("4. MODEL ASSUMPTIONS - RESIDUAL ANALİZİ")
print("=" * 80)

residuals = y_test - y_pred_test

print(f"Residual ortalama: {residuals.mean():,.2f}")
print(f"Residual std: {residuals.std():,.2f}")
print(f"Residual skewness: {pd.Series(residuals).skew():.4f}")
print(f"Residual kurtosis: {pd.Series(residuals).kurt():.4f}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

idx = np.random.choice(len(y_test), 5000, replace=False)
axes[0, 0].scatter(y_pred_test[idx], residuals[idx], alpha=0.3, s=10, color='steelblue')
axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[0, 0].set_xlabel('Tahmin (Fitted)')
axes[0, 0].set_ylabel('Residual')
axes[0, 0].set_title('Residual vs Fitted\n(Homoskedastisite kontrolü)', fontweight='bold')

axes[0, 1].hist(residuals, bins=50, color='steelblue', edgecolor='black')
axes[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[0, 1].axvline(x=residuals.mean(), color='green', linestyle='--',
                   linewidth=2, label=f'Mean: {residuals.mean():,.0f}')
axes[0, 1].set_xlabel('Residual')
axes[0, 1].set_ylabel('Frekans')
axes[0, 1].set_title('Residual Dağılımı\n(Normallik kontrolü)', fontweight='bold')
axes[0, 1].legend()

axes[1, 0].scatter(y_test[idx], y_pred_test[idx], alpha=0.3, s=10, color='steelblue')
min_val = min(y_test.min(), y_pred_test.min())
max_val = max(y_test.max(), y_pred_test.max())
axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
                label='Mükemmel tahmin')
axes[1, 0].set_xlabel('Gerçek')
axes[1, 0].set_ylabel('Tahmin')
axes[1, 0].set_title(f'Tahmin vs Gerçek (R²={test_r2:.4f})', fontweight='bold')
axes[1, 0].legend()

exp_years = X_test_raw['experience_years'].values
axes[1, 1].scatter(exp_years[idx], residuals[idx], alpha=0.3, s=10, color='steelblue')
axes[1, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[1, 1].set_xlabel('experience_years')
axes[1, 1].set_ylabel('Residual')
axes[1, 1].set_title('Residual vs experience_years\n(Linearity kontrolü)', fontweight='bold')

plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/08_linear_residuals.png", dpi=100, bbox_inches='tight')
plt.close()
print(f"Kaydedildi: {FIGURES_DIR}/08_linear_residuals.png")

print("\n" + "=" * 80)
print("5. SONUÇLAR KAYDEDİLİYOR")
print("=" * 80)

results = pd.DataFrame({
    'Model': ['Simple Linear', 'Multiple Linear'],
    'Train MAE': [np.nan, train_mae],
    'Val MAE': [np.nan, val_mae],
    'Test MAE': [mean_absolute_error(y_test, y_pred_simple_test), test_mae],
    'Train RMSE': [np.nan, train_rmse],
    'Val RMSE': [np.nan, val_rmse],
    'Test RMSE': [simple_test_rmse, test_rmse],
    'Train R²': [np.nan, train_r2],
    'Val R²': [np.nan, val_r2],
    'Test R²': [simple_test_r2, test_r2]
})
print(results.to_string(index=False))
results.to_csv(f"{MODELS_DIR}/results_linear.csv", index=False)

print("\nLinear models tamamlandı. Sıradaki: 05_regularization.py")
