"""08_evaluation.py - Tüm modeller karşılaştırma + Cross-Validation + Hata analizi"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os

sns.set_style("whitegrid")

PROCESSED_DIR = "data/processed"
MODELS_DIR = "outputs/models"
FIGURES_DIR = "outputs/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

print("=" * 80)
print("1. VERİ ve MODELLER YÜKLENİYOR")
print("=" * 80)

X_train = np.load(f"{PROCESSED_DIR}/X_train.npy")
X_test = np.load(f"{PROCESSED_DIR}/X_test.npy")
y_train = np.load(f"{PROCESSED_DIR}/y_train.npy")
y_test = np.load(f"{PROCESSED_DIR}/y_test.npy")
X_test_raw = pd.read_csv(f"{PROCESSED_DIR}/X_test_raw.csv")

models = {
    'Multiple Linear': joblib.load(f"{MODELS_DIR}/multiple_linear.pkl"),
    'Ridge': joblib.load(f"{MODELS_DIR}/ridge.pkl"),
    'Lasso': joblib.load(f"{MODELS_DIR}/lasso.pkl"),
    'ElasticNet': joblib.load(f"{MODELS_DIR}/elasticnet.pkl"),
    'Polynomial+Ridge': joblib.load(f"{MODELS_DIR}/polynomial_ridge.pkl"),
    'KNN': joblib.load(f"{MODELS_DIR}/knn.pkl"),
    'Decision Tree': joblib.load(f"{MODELS_DIR}/decision_tree.pkl")
}

print(f"Test: {X_test.shape}")
print(f"Yüklenen modeller: {list(models.keys())}")

print("\n" + "=" * 80)
print("2. TÜM MODELLER - TEST METRİKLERİ")
print("=" * 80)

results = []
predictions = {}

print("\nModeller değerlendiriliyor...")
for name, m in models.items():
    print(f"  - {name}...")

    if name == 'KNN':
        np.random.seed(42)
        test_idx = np.random.choice(len(X_test), 10000, replace=False)
        X_test_eval = X_test[test_idx]
        y_test_eval = y_test[test_idx]
        train_idx = np.random.choice(len(X_train), 10000, replace=False)
        X_train_eval = X_train[train_idx]
        y_train_eval = y_train[train_idx]
        pred_train = m.predict(X_train_eval)
        pred_test = m.predict(X_test_eval)
        full_pred_test = np.full(len(X_test), np.nan)
        full_pred_test[test_idx] = pred_test
        predictions[name] = full_pred_test
        train_r2 = r2_score(y_train_eval, pred_train)
        test_mae = mean_absolute_error(y_test_eval, pred_test)
        test_mse = mean_squared_error(y_test_eval, pred_test)
        test_rmse = np.sqrt(test_mse)
        test_r2 = r2_score(y_test_eval, pred_test)
        note = ' (10K subsample)'
    else:
        pred_train = m.predict(X_train)
        pred_test = m.predict(X_test)
        predictions[name] = pred_test
        train_r2 = r2_score(y_train, pred_train)
        test_mae = mean_absolute_error(y_test, pred_test)
        test_mse = mean_squared_error(y_test, pred_test)
        test_rmse = np.sqrt(test_mse)
        test_r2 = r2_score(y_test, pred_test)
        note = ''

    results.append({
        'Model': name + note,
        'Test MAE': test_mae,
        'Test MSE': test_mse,
        'Test RMSE': test_rmse,
        'Train R²': train_r2,
        'Test R²': test_r2,
        'Overfit (Train-Test R²)': train_r2 - test_r2
    })

df_results = pd.DataFrame(results)
df_results = df_results.sort_values('Test R²', ascending=False).reset_index(drop=True)
print(df_results.to_string(index=False))
df_results.to_csv(f"{MODELS_DIR}/results_all_models.csv", index=False)

print("\n" + "=" * 80)
print("3. 5-FOLD CROSS-VALIDATION (lineer modeller için)")
print("=" * 80)
print("NOT: KNN, Decision Tree ve Polynomial CV'de çok yavaş, atlanıyor.")
print("Lineer modeller için CV yapılıyor (hızlı + ana karşılaştırma için yeterli).")

fast_models = {
    'Multiple Linear': models['Multiple Linear'],
    'Ridge': models['Ridge'],
    'Lasso': models['Lasso'],
    'ElasticNet': models['ElasticNet'],
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = []

print(f"\n{'Model':<20}{'CV RMSE (mean)':>18}{'CV RMSE (std)':>18}{'CV R² (mean)':>15}")
for name, m in fast_models.items():
    cv_rmse = -cross_val_score(m, X_train, y_train, cv=kf,
                                scoring='neg_root_mean_squared_error', n_jobs=-1)
    cv_r2 = cross_val_score(m, X_train, y_train, cv=kf, scoring='r2', n_jobs=-1)
    cv_results.append({
        'Model': name,
        'CV RMSE mean': cv_rmse.mean(),
        'CV RMSE std': cv_rmse.std(),
        'CV R² mean': cv_r2.mean(),
        'CV R² std': cv_r2.std()
    })
    print(f"{name:<20}{cv_rmse.mean():>18,.2f}{cv_rmse.std():>18,.2f}{cv_r2.mean():>15.4f}")

df_cv = pd.DataFrame(cv_results)
df_cv.to_csv(f"{MODELS_DIR}/cv_results.csv", index=False)

print("\n" + "=" * 80)
print("4. MODEL KARŞILAŞTIRMA GRAFİĞİ")
print("=" * 80)

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

x_pos = range(len(df_results))
colors = sns.color_palette('viridis', len(df_results))

for ax, metric, title, fmt in zip(
    axes,
    ['Test MAE', 'Test RMSE', 'Test R²'],
    ['Test MAE (Düşük = İyi)', 'Test RMSE (Düşük = İyi)', 'Test R² (Yüksek = İyi)'],
    [',.0f', ',.0f', '.4f']
):
    bars = ax.bar(x_pos, df_results[metric], color=colors, edgecolor='black')
    for i, (bar, val) in enumerate(zip(bars, df_results[metric])):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                format(val, fmt), ha='center', va='bottom',
                fontsize=9, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df_results['Model'], rotation=30, ha='right')
    ax.set_title(title, fontweight='bold')
    ax.set_ylabel(metric)

plt.suptitle('Tüm Modellerin Test Performansı', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/14_all_models_comparison.png", dpi=100, bbox_inches='tight')
plt.close()
print(f"Kaydedildi: {FIGURES_DIR}/14_all_models_comparison.png")

print("\n" + "=" * 80)
print("5. OVERFITTING ANALİZİ (Train R² vs Test R²)")
print("=" * 80)

print(df_results[['Model', 'Train R²', 'Test R²', 'Overfit (Train-Test R²)']].to_string(index=False))

plt.figure(figsize=(12, 6))
x = np.arange(len(df_results))
width = 0.35
plt.bar(x - width/2, df_results['Train R²'], width,
        label='Train R²', color='steelblue', edgecolor='black')
plt.bar(x + width/2, df_results['Test R²'], width,
        label='Test R²', color='orange', edgecolor='black')
plt.xticks(x, df_results['Model'], rotation=30, ha='right')
plt.ylabel('R²')
plt.title('Train vs Test R² (Overfitting Kontrolü)', fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/15_overfit_check.png", dpi=100, bbox_inches='tight')
plt.close()
print(f"Kaydedildi: {FIGURES_DIR}/15_overfit_check.png")

print("\n" + "=" * 80)
print("6. EN İYİ MODEL")
print("=" * 80)

best_model_name = df_results.iloc[0]['Model']
print(f"En iyi model: {best_model_name}")
print(f"  Test MAE:  {df_results.iloc[0]['Test MAE']:,.2f}")
print(f"  Test RMSE: {df_results.iloc[0]['Test RMSE']:,.2f}")
print(f"  Test R²:   {df_results.iloc[0]['Test R²']:.4f}")

with open(f"{MODELS_DIR}/best_model.txt", "w") as f:
    f.write(best_model_name)

print("\n" + "=" * 80)
print("7. EN İYİ MODEL İÇİN HATA ANALİZİ")
print("=" * 80)

best_pred = predictions[best_model_name]

if np.any(np.isnan(best_pred)):
    valid_mask = ~np.isnan(best_pred)
    best_pred_clean = best_pred[valid_mask]
    y_test_clean = y_test[valid_mask]
    print(f"NOT: {best_model_name} subsample değerlendirildi, hata analizi {valid_mask.sum()} örnek üzerinde.")
else:
    best_pred_clean = best_pred
    y_test_clean = y_test

residuals = y_test_clean - best_pred_clean

print(f"Residual mean: {residuals.mean():,.2f}")
print(f"Residual std:  {residuals.std():,.2f}")
print(f"Residual min:  {residuals.min():,.2f}")
print(f"Residual max:  {residuals.max():,.2f}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

idx = np.random.choice(len(y_test_clean), min(5000, len(y_test_clean)), replace=False)

axes[0, 0].scatter(y_test_clean[idx], best_pred_clean[idx], alpha=0.3, s=10, color='steelblue')
min_v = min(y_test_clean.min(), best_pred_clean.min())
max_v = max(y_test_clean.max(), best_pred_clean.max())
axes[0, 0].plot([min_v, max_v], [min_v, max_v], 'r--', linewidth=2)
axes[0, 0].set_xlabel('Gerçek')
axes[0, 0].set_ylabel('Tahmin')
axes[0, 0].set_title(f'{best_model_name} - Tahmin vs Gerçek', fontweight='bold')

axes[0, 1].scatter(best_pred_clean[idx], residuals[idx], alpha=0.3, s=10, color='steelblue')
axes[0, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[0, 1].set_xlabel('Tahmin')
axes[0, 1].set_ylabel('Residual')
axes[0, 1].set_title('Residual vs Tahmin', fontweight='bold')

axes[1, 0].hist(residuals, bins=50, color='steelblue', edgecolor='black')
axes[1, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[1, 0].axvline(x=residuals.mean(), color='green', linestyle='--',
                   linewidth=2, label=f'Mean: {residuals.mean():,.0f}')
axes[1, 0].set_xlabel('Residual')
axes[1, 0].set_ylabel('Frekans')
axes[1, 0].set_title('Residual Dağılımı', fontweight='bold')
axes[1, 0].legend()

abs_residuals = np.abs(residuals)
axes[1, 1].scatter(y_test_clean[idx], abs_residuals[idx], alpha=0.3, s=10, color='steelblue')
axes[1, 1].set_xlabel('Gerçek Maaş')
axes[1, 1].set_ylabel('|Residual|')
axes[1, 1].set_title('Mutlak Hata vs Gerçek Maaş', fontweight='bold')

plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/16_error_analysis.png", dpi=100, bbox_inches='tight')
plt.close()
print(f"Kaydedildi: {FIGURES_DIR}/16_error_analysis.png")

print("\n" + "=" * 80)
print("8. SEGMENT BAZLI HATA ANALİZİ (Maaş çeyreklikleri)")
print("=" * 80)

y_quartiles = pd.qcut(y_test_clean, q=4, labels=['Q1 (Düşük)', 'Q2', 'Q3', 'Q4 (Yüksek)'])

segment_results = []
for q in ['Q1 (Düşük)', 'Q2', 'Q3', 'Q4 (Yüksek)']:
    mask = y_quartiles == q
    segment_results.append({
        'Segment': q,
        'N': int(mask.sum()),
        'MAE': mean_absolute_error(y_test_clean[mask], best_pred_clean[mask]),
        'RMSE': np.sqrt(mean_squared_error(y_test_clean[mask], best_pred_clean[mask])),
        'R²': r2_score(y_test_clean[mask], best_pred_clean[mask])
    })

df_seg = pd.DataFrame(segment_results)
print(df_seg.to_string(index=False))
df_seg.to_csv(f"{MODELS_DIR}/segment_analysis.csv", index=False)

print("\nDeğerlendirme tamamlandı. Sıradaki: 09_interpretability.py")
