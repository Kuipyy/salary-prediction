"""05_regularization.py - Ridge (L2), Lasso (L1), ElasticNet + alpha taraması"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
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

with open(f"{PROCESSED_DIR}/feature_names.txt") as f:
    feature_names = [line.strip() for line in f]

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

print("\n" + "=" * 80)
print("2. ALPHA TARAMASI: RIDGE, LASSO, ELASTICNET")
print("=" * 80)

alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

results_ridge = []
results_lasso = []
results_elastic = []

print("\n--- RIDGE (L2) ---")
print(f"{'Alpha':>10}{'Train RMSE':>15}{'Val RMSE':>15}{'Val R²':>10}")
for a in alphas:
    m = Ridge(alpha=a, random_state=42)
    m.fit(X_train, y_train)
    tr_rmse = np.sqrt(mean_squared_error(y_train, m.predict(X_train)))
    va_rmse = np.sqrt(mean_squared_error(y_val, m.predict(X_val)))
    va_r2 = r2_score(y_val, m.predict(X_val))
    results_ridge.append({'alpha': a, 'train_rmse': tr_rmse, 'val_rmse': va_rmse, 'val_r2': va_r2})
    print(f"{a:>10}{tr_rmse:>15,.2f}{va_rmse:>15,.2f}{va_r2:>10.4f}")

print("\n--- LASSO (L1) ---")
print(f"{'Alpha':>10}{'Train RMSE':>15}{'Val RMSE':>15}{'Val R²':>10}{'Sıfır katsayı':>18}")
for a in alphas:
    m = Lasso(alpha=a, random_state=42, max_iter=10000)
    m.fit(X_train, y_train)
    tr_rmse = np.sqrt(mean_squared_error(y_train, m.predict(X_train)))
    va_rmse = np.sqrt(mean_squared_error(y_val, m.predict(X_val)))
    va_r2 = r2_score(y_val, m.predict(X_val))
    zero_coefs = int(np.sum(m.coef_ == 0))
    results_lasso.append({
        'alpha': a, 'train_rmse': tr_rmse, 'val_rmse': va_rmse,
        'val_r2': va_r2, 'zero_coefs': zero_coefs
    })
    print(f"{a:>10}{tr_rmse:>15,.2f}{va_rmse:>15,.2f}{va_r2:>10.4f}{zero_coefs:>18}")

print("\n--- ELASTICNET (L1 + L2, l1_ratio=0.5) ---")
print(f"{'Alpha':>10}{'Train RMSE':>15}{'Val RMSE':>15}{'Val R²':>10}")
for a in alphas:
    m = ElasticNet(alpha=a, l1_ratio=0.5, random_state=42, max_iter=10000)
    m.fit(X_train, y_train)
    tr_rmse = np.sqrt(mean_squared_error(y_train, m.predict(X_train)))
    va_rmse = np.sqrt(mean_squared_error(y_val, m.predict(X_val)))
    va_r2 = r2_score(y_val, m.predict(X_val))
    results_elastic.append({'alpha': a, 'train_rmse': tr_rmse, 'val_rmse': va_rmse, 'val_r2': va_r2})
    print(f"{a:>10}{tr_rmse:>15,.2f}{va_rmse:>15,.2f}{va_r2:>10.4f}")

df_ridge = pd.DataFrame(results_ridge)
df_lasso = pd.DataFrame(results_lasso)
df_elastic = pd.DataFrame(results_elastic)

df_ridge.to_csv(f"{MODELS_DIR}/ridge_alpha_sweep.csv", index=False)
df_lasso.to_csv(f"{MODELS_DIR}/lasso_alpha_sweep.csv", index=False)
df_elastic.to_csv(f"{MODELS_DIR}/elastic_alpha_sweep.csv", index=False)

print("\n" + "=" * 80)
print("3. REGULARIZATION STRENGTH GRAFİĞİ")
print("=" * 80)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, df_m, name in zip(axes,
                          [df_ridge, df_lasso, df_elastic],
                          ['Ridge (L2)', 'Lasso (L1)', 'ElasticNet']):
    ax.plot(df_m['alpha'], df_m['train_rmse'], 'o-', label='Train RMSE', color='steelblue')
    ax.plot(df_m['alpha'], df_m['val_rmse'], 's-', label='Val RMSE', color='red')
    ax.set_xscale('log')
    ax.set_xlabel('Alpha (log scale)')
    ax.set_ylabel('RMSE')
    ax.set_title(f'{name}', fontweight='bold')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)

plt.suptitle('Regularization Strength: Alpha vs RMSE', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/09_alpha_sweep.png", dpi=100, bbox_inches='tight')
plt.close()
print(f"Kaydedildi: {FIGURES_DIR}/09_alpha_sweep.png")

print("\n" + "=" * 80)
print("4. EN İYİ ALPHA'LARLA FINAL MODELLER")
print("=" * 80)

best_alpha_ridge = df_ridge.loc[df_ridge['val_rmse'].idxmin(), 'alpha']
best_alpha_lasso = df_lasso.loc[df_lasso['val_rmse'].idxmin(), 'alpha']
best_alpha_elastic = df_elastic.loc[df_elastic['val_rmse'].idxmin(), 'alpha']

print(f"En iyi alpha - Ridge: {best_alpha_ridge}")
print(f"En iyi alpha - Lasso: {best_alpha_lasso}")
print(f"En iyi alpha - ElasticNet: {best_alpha_elastic}")

ridge_best = Ridge(alpha=best_alpha_ridge, random_state=42).fit(X_train, y_train)
lasso_best = Lasso(alpha=best_alpha_lasso, random_state=42, max_iter=10000).fit(X_train, y_train)
elastic_best = ElasticNet(alpha=best_alpha_elastic, l1_ratio=0.5,
                          random_state=42, max_iter=10000).fit(X_train, y_train)

joblib.dump(ridge_best, f"{MODELS_DIR}/ridge.pkl")
joblib.dump(lasso_best, f"{MODELS_DIR}/lasso.pkl")
joblib.dump(elastic_best, f"{MODELS_DIR}/elasticnet.pkl")

print("\n" + "=" * 80)
print("5. LASSO FEATURE SEÇİM ÇIKTISI")
print("=" * 80)

lasso_coefs = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': lasso_best.coef_,
    'Is_Zero': lasso_best.coef_ == 0
}).sort_values('Coefficient', key=abs, ascending=False)

n_zero = lasso_coefs['Is_Zero'].sum()
n_nonzero = (~lasso_coefs['Is_Zero']).sum()

print(f"\nToplam özellik: {len(lasso_coefs)}")
print(f"Sıfıra çekilen (eleme): {n_zero}")
print(f"Korunan: {n_nonzero}")

if n_zero > 0:
    print("\n--- Lasso'nun ELEDİĞİ özellikler ---")
    print(lasso_coefs[lasso_coefs['Is_Zero']]['Feature'].tolist())
else:
    print("\nLasso hiçbir özelliği elemiyor (alpha çok düşük olduğu için).")

lasso_coefs.to_csv(f"{MODELS_DIR}/lasso_coefficients.csv", index=False)

plt.figure(figsize=(10, 12))
top_coefs = lasso_coefs[lasso_coefs['Coefficient'] != 0].head(20).iloc[::-1]
colors = ['#d62728' if c < 0 else '#1f77b4' for c in top_coefs['Coefficient']]
plt.barh(top_coefs['Feature'], top_coefs['Coefficient'], color=colors, edgecolor='black')
plt.axvline(x=0, color='black', linewidth=0.8)
plt.xlabel('Katsayı')
plt.title(f'Lasso - Sıfır Olmayan En Büyük 20 Katsayı (alpha={best_alpha_lasso})',
          fontweight='bold')
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/10_lasso_coefs.png", dpi=100, bbox_inches='tight')
plt.close()
print(f"Kaydedildi: {FIGURES_DIR}/10_lasso_coefs.png")

print("\n" + "=" * 80)
print("6. FINAL TEST PERFORMANSI")
print("=" * 80)

baseline = LinearRegression().fit(X_train, y_train)
models = {
    'Linear (baseline)': baseline,
    f'Ridge (α={best_alpha_ridge})': ridge_best,
    f'Lasso (α={best_alpha_lasso})': lasso_best,
    f'ElasticNet (α={best_alpha_elastic})': elastic_best
}

final_results = []
for name, m in models.items():
    pred = m.predict(X_test)
    final_results.append({
        'Model': name,
        'Test MAE': mean_absolute_error(y_test, pred),
        'Test MSE': mean_squared_error(y_test, pred),
        'Test RMSE': np.sqrt(mean_squared_error(y_test, pred)),
        'Test R²': r2_score(y_test, pred)
    })

df_final = pd.DataFrame(final_results)
print(df_final.to_string(index=False))
df_final.to_csv(f"{MODELS_DIR}/results_regularization.csv", index=False)

print("\nRegularization tamamlandı. Sıradaki: 06_polynomial.py")
