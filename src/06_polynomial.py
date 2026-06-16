"""06_polynomial.py - Polynomial Features + Bias-Variance Trade-off"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
import time

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

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

print("\n" + "=" * 80)
print("2. POLYNOMIAL FEATURES - DEGREE TARAMASI")
print("=" * 80)
print("NOT: Yüksek degree çok fazla feature üretir, bu yüzden 50K subsample kullanılıyor.")

np.random.seed(42)
n_sub = 5000
sub_idx_train = np.random.choice(len(X_train), n_sub, replace=False)
X_train_sub = X_train[sub_idx_train]
y_train_sub = y_train[sub_idx_train]

n_val_sub = 5000
val_sub_idx = np.random.choice(len(X_val), n_val_sub, replace=False)
X_val_sub = X_val[val_sub_idx]
y_val_sub = y_val[val_sub_idx]

print(f"Train subsample shape: {X_train_sub.shape}")
print(f"Val subsample shape: {X_val_sub.shape}")
print("NOT: Polynomial degree=3 çok fazla feature ürettiği için küçük subsample (5K)")
print("kullanılıyor. Bias-variance trend'i bu boyutta da net görülebilir.")

degrees_config = [
    {'degree': 1, 'interaction_only': False, 'label': 'Degree 1'},
    {'degree': 2, 'interaction_only': False, 'label': 'Degree 2'},
    {'degree': 3, 'interaction_only': True, 'label': 'Degree 3 (interaction_only)'}
]
results = []

for cfg in degrees_config:
    d = cfg['degree']
    io = cfg['interaction_only']
    label = cfg['label']
    print(f"\n--- {label} ---")
    start = time.time()

    pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=d, include_bias=False, interaction_only=io)),
        ('linreg', LinearRegression())
    ])

    pipeline.fit(X_train_sub, y_train_sub)

    pred_train = pipeline.predict(X_train_sub)
    pred_val = pipeline.predict(X_val_sub)

    train_rmse = np.sqrt(mean_squared_error(y_train_sub, pred_train))
    val_rmse = np.sqrt(mean_squared_error(y_val_sub, pred_val))
    train_r2 = r2_score(y_train_sub, pred_train)
    val_r2 = r2_score(y_val_sub, pred_val)

    n_features = pipeline.named_steps['poly'].n_output_features_
    elapsed = time.time() - start

    print(f"  Feature sayısı: {n_features}")
    print(f"  Süre: {elapsed:.1f}s")
    print(f"  Train RMSE: {train_rmse:,.2f}  |  R²: {train_r2:.4f}")
    print(f"  Val RMSE:   {val_rmse:,.2f}  |  R²: {val_r2:.4f}")

    results.append({
        'degree': label,
        'n_features': n_features,
        'train_rmse': train_rmse,
        'val_rmse': val_rmse,
        'train_r2': train_r2,
        'val_r2': val_r2
    })

df_poly = pd.DataFrame(results)
df_poly.to_csv(f"{MODELS_DIR}/polynomial_results.csv", index=False)

print("\n" + "=" * 80)
print("3. BIAS-VARIANCE TRADE-OFF GRAFİĞİ")
print("=" * 80)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

x_pos = range(len(df_poly))

axes[0].plot(x_pos, df_poly['train_rmse'], 'o-',
             label='Train RMSE', color='steelblue', linewidth=2, markersize=10)
axes[0].plot(x_pos, df_poly['val_rmse'], 's-',
             label='Validation RMSE', color='red', linewidth=2, markersize=10)
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(df_poly['degree'], rotation=15, ha='right')
axes[0].set_xlabel('Polynomial Degree')
axes[0].set_ylabel('RMSE')
axes[0].set_title('Bias-Variance Trade-off (RMSE)', fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(x_pos, df_poly['train_r2'], 'o-',
             label='Train R²', color='steelblue', linewidth=2, markersize=10)
axes[1].plot(x_pos, df_poly['val_r2'], 's-',
             label='Validation R²', color='red', linewidth=2, markersize=10)
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(df_poly['degree'], rotation=15, ha='right')
axes[1].set_xlabel('Polynomial Degree')
axes[1].set_ylabel('R²')
axes[1].set_title('Bias-Variance Trade-off (R²)', fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/11_bias_variance.png", dpi=100, bbox_inches='tight')
plt.close()
print(f"Kaydedildi: {FIGURES_DIR}/11_bias_variance.png")

print("\n" + "=" * 80)
print("4. POLYNOMIAL + RIDGE (REGULARIZATION İLE OVERFITTING KONTROLÜ)")
print("=" * 80)

best_idx = df_poly['val_rmse'].idxmin()
best_label = df_poly.loc[best_idx, 'degree']
best_degree = degrees_config[best_idx]['degree']
best_io = degrees_config[best_idx]['interaction_only']
print(f"En iyi degree (val RMSE'ye göre): {best_label}")

ridge_alphas = [0.1, 1.0, 10.0, 100.0]
poly_ridge_results = []

print(f"\n{best_label} ile Polynomial + Ridge taraması:")
print(f"{'Alpha':>10}{'Train RMSE':>15}{'Val RMSE':>15}{'Val R²':>10}")

for a in ridge_alphas:
    pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=best_degree, include_bias=False,
                                     interaction_only=best_io)),
        ('ridge', Ridge(alpha=a, random_state=42))
    ])
    pipeline.fit(X_train_sub, y_train_sub)
    tr_rmse = np.sqrt(mean_squared_error(y_train_sub, pipeline.predict(X_train_sub)))
    va_rmse = np.sqrt(mean_squared_error(y_val_sub, pipeline.predict(X_val_sub)))
    va_r2 = r2_score(y_val_sub, pipeline.predict(X_val_sub))
    poly_ridge_results.append({
        'alpha': a, 'train_rmse': tr_rmse, 'val_rmse': va_rmse, 'val_r2': va_r2
    })
    print(f"{a:>10}{tr_rmse:>15,.2f}{va_rmse:>15,.2f}{va_r2:>10.4f}")

df_poly_ridge = pd.DataFrame(poly_ridge_results)
df_poly_ridge.to_csv(f"{MODELS_DIR}/poly_ridge_results.csv", index=False)

print("\n" + "=" * 80)
print("5. EN İYİ POLYNOMIAL MODEL - TEST PERFORMANSI")
print("=" * 80)

best_alpha = df_poly_ridge.loc[df_poly_ridge['val_rmse'].idxmin(), 'alpha']
print(f"En iyi alpha: {best_alpha}")

final_pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=best_degree, include_bias=False,
                                 interaction_only=best_io)),
    ('ridge', Ridge(alpha=best_alpha, random_state=42))
])
final_pipeline.fit(X_train_sub, y_train_sub)
pred_test = final_pipeline.predict(X_test)

test_mae = mean_absolute_error(y_test, pred_test)
test_mse = mean_squared_error(y_test, pred_test)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(y_test, pred_test)

print(f"\nFinal Polynomial ({best_label}) + Ridge (alpha={best_alpha}):")
print(f"  Test MAE:  {test_mae:,.2f}")
print(f"  Test MSE:  {test_mse:,.2f}")
print(f"  Test RMSE: {test_rmse:,.2f}")
print(f"  Test R²:   {test_r2:.4f}")

joblib.dump(final_pipeline, f"{MODELS_DIR}/polynomial_ridge.pkl")

final_results = pd.DataFrame([{
    'Model': f'Polynomial({best_label}) + Ridge(α={best_alpha})',
    'Test MAE': test_mae,
    'Test MSE': test_mse,
    'Test RMSE': test_rmse,
    'Test R²': test_r2
}])
final_results.to_csv(f"{MODELS_DIR}/results_polynomial.csv", index=False)

print("\nPolynomial tamamlandı. Sıradaki: 07_knn_tree.py")
