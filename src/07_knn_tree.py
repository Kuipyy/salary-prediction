"""07_knn_tree.py - KNN Regressor + Decision Tree Regressor"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
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

with open(f"{PROCESSED_DIR}/feature_names.txt") as f:
    feature_names = [line.strip() for line in f]

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

print("\n" + "=" * 80)
print("2. KNN - K TARAMASI (Euclidean Distance)")
print("=" * 80)
print("NOT: KNN yavaş çalışır, 50K train subsample kullanılıyor.")

np.random.seed(42)
n_sub = 50000
sub_idx = np.random.choice(len(X_train), n_sub, replace=False)
X_train_sub = X_train[sub_idx]
y_train_sub = y_train[sub_idx]

n_val_sub = 10000
val_sub_idx = np.random.choice(len(X_val), n_val_sub, replace=False)
X_val_sub = X_val[val_sub_idx]
y_val_sub = y_val[val_sub_idx]

k_values = [3, 5, 10, 20, 50]
knn_results = []

print(f"\n{'k':>5}{'Train RMSE':>15}{'Val RMSE':>15}{'Val R²':>10}{'Süre (s)':>10}")
for k in k_values:
    start = time.time()
    m = KNeighborsRegressor(n_neighbors=k, metric='euclidean', n_jobs=-1)
    m.fit(X_train_sub, y_train_sub)
    pred_train = m.predict(X_train_sub)
    pred_val = m.predict(X_val_sub)
    tr_rmse = np.sqrt(mean_squared_error(y_train_sub, pred_train))
    va_rmse = np.sqrt(mean_squared_error(y_val_sub, pred_val))
    va_r2 = r2_score(y_val_sub, pred_val)
    elapsed = time.time() - start
    knn_results.append({
        'k': k, 'train_rmse': tr_rmse, 'val_rmse': va_rmse,
        'val_r2': va_r2, 'time_s': elapsed
    })
    print(f"{k:>5}{tr_rmse:>15,.2f}{va_rmse:>15,.2f}{va_r2:>10.4f}{elapsed:>10.1f}")

df_knn = pd.DataFrame(knn_results)
df_knn.to_csv(f"{MODELS_DIR}/knn_k_sweep.csv", index=False)

best_k = int(df_knn.loc[df_knn['val_rmse'].idxmin(), 'k'])
print(f"\nEn iyi k: {best_k}")

print("\n" + "=" * 80)
print("3. KNN - DISTANCE METRIC KARŞILAŞTIRMASI")
print("=" * 80)
print(f"k={best_k} ile farklı distance metric'ler deneniyor.")

metrics = ['euclidean', 'manhattan']
dist_results = []

print(f"\n{'Metric':>12}{'Val RMSE':>15}{'Val R²':>10}{'Süre (s)':>10}")
for metric in metrics:
    start = time.time()
    m = KNeighborsRegressor(n_neighbors=best_k, metric=metric, n_jobs=-1)
    m.fit(X_train_sub, y_train_sub)
    pred_val = m.predict(X_val_sub)
    va_rmse = np.sqrt(mean_squared_error(y_val_sub, pred_val))
    va_r2 = r2_score(y_val_sub, pred_val)
    elapsed = time.time() - start
    dist_results.append({
        'metric': metric, 'val_rmse': va_rmse, 'val_r2': va_r2, 'time_s': elapsed
    })
    print(f"{metric:>12}{va_rmse:>15,.2f}{va_r2:>10.4f}{elapsed:>10.1f}")

df_dist = pd.DataFrame(dist_results)
df_dist.to_csv(f"{MODELS_DIR}/knn_distance_metrics.csv", index=False)

best_metric = df_dist.loc[df_dist['val_rmse'].idxmin(), 'metric']
print(f"\nEn iyi distance metric: {best_metric}")

print("\n" + "=" * 80)
print("4. KNN GRAFİKLERİ")
print("=" * 80)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(df_knn['k'], df_knn['train_rmse'], 'o-', label='Train RMSE',
             color='steelblue', linewidth=2, markersize=10)
axes[0].plot(df_knn['k'], df_knn['val_rmse'], 's-', label='Val RMSE',
             color='red', linewidth=2, markersize=10)
axes[0].set_xlabel('k (Komşu sayısı)')
axes[0].set_ylabel('RMSE')
axes[0].set_title('KNN - k Taraması', fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].bar(df_dist['metric'], df_dist['val_rmse'],
            color=['steelblue', 'orange'], edgecolor='black')
for i, (m, r) in enumerate(zip(df_dist['metric'], df_dist['val_rmse'])):
    axes[1].text(i, r, f'{r:,.0f}', ha='center', va='bottom', fontweight='bold')
axes[1].set_xlabel('Distance Metric')
axes[1].set_ylabel('Val RMSE')
axes[1].set_title(f'Distance Metric Karşılaştırması (k={best_k})', fontweight='bold')

plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/12_knn_results.png", dpi=100, bbox_inches='tight')
plt.close()
print(f"Kaydedildi: {FIGURES_DIR}/12_knn_results.png")

print("\n" + "=" * 80)
print("5. DECISION TREE - MAX_DEPTH TARAMASI")
print("=" * 80)

depths = [3, 5, 10, 15, 20, None]
dt_results = []

print(f"\n{'Max Depth':>12}{'Train RMSE':>15}{'Val RMSE':>15}{'Val R²':>10}{'Train R²':>12}")
for d in depths:
    m = DecisionTreeRegressor(max_depth=d, random_state=42)
    m.fit(X_train, y_train)
    pred_train = m.predict(X_train)
    pred_val = m.predict(X_val)
    tr_rmse = np.sqrt(mean_squared_error(y_train, pred_train))
    va_rmse = np.sqrt(mean_squared_error(y_val, pred_val))
    tr_r2 = r2_score(y_train, pred_train)
    va_r2 = r2_score(y_val, pred_val)
    depth_label = 'None' if d is None else str(d)
    dt_results.append({
        'max_depth': depth_label, 'train_rmse': tr_rmse, 'val_rmse': va_rmse,
        'train_r2': tr_r2, 'val_r2': va_r2
    })
    print(f"{depth_label:>12}{tr_rmse:>15,.2f}{va_rmse:>15,.2f}{va_r2:>10.4f}{tr_r2:>12.4f}")

df_dt = pd.DataFrame(dt_results)
df_dt.to_csv(f"{MODELS_DIR}/dt_depth_sweep.csv", index=False)

best_depth_label = df_dt.loc[df_dt['val_rmse'].idxmin(), 'max_depth']
best_depth = None if best_depth_label == 'None' else int(best_depth_label)
print(f"\nEn iyi max_depth: {best_depth_label}")

print("\n" + "=" * 80)
print("6. DECISION TREE - FEATURE IMPORTANCE (Information Gain)")
print("=" * 80)

dt_best = DecisionTreeRegressor(max_depth=best_depth, random_state=42)
dt_best.fit(X_train, y_train)

importances = pd.DataFrame({
    'Feature': feature_names,
    'Importance': dt_best.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n--- En önemli 15 özellik ---")
print(importances.head(15).to_string(index=False))

importances.to_csv(f"{MODELS_DIR}/dt_feature_importance.csv", index=False)

print("\n" + "=" * 80)
print("7. DECISION TREE GRAFİKLERİ")
print("=" * 80)

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

dt_plot = df_dt.copy()
x_labels = dt_plot['max_depth'].astype(str)
x_pos = range(len(x_labels))

axes[0].plot(x_pos, dt_plot['train_rmse'], 'o-', label='Train RMSE',
             color='steelblue', linewidth=2, markersize=10)
axes[0].plot(x_pos, dt_plot['val_rmse'], 's-', label='Val RMSE',
             color='red', linewidth=2, markersize=10)
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(x_labels)
axes[0].set_xlabel('Max Depth')
axes[0].set_ylabel('RMSE')
axes[0].set_title('Decision Tree - Depth Taraması\n(Overfitting kontrolü)', fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

top_15 = importances.head(15).iloc[::-1]
axes[1].barh(top_15['Feature'], top_15['Importance'],
             color='steelblue', edgecolor='black')
axes[1].set_xlabel('Importance (Information Gain bazlı)')
axes[1].set_title(f'Decision Tree - En Önemli 15 Özellik\n(max_depth={best_depth_label})',
                  fontweight='bold')

plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/13_decision_tree.png", dpi=100, bbox_inches='tight')
plt.close()
print(f"Kaydedildi: {FIGURES_DIR}/13_decision_tree.png")

print("\n" + "=" * 80)
print("8. FINAL TEST PERFORMANSI")
print("=" * 80)

knn_final = KNeighborsRegressor(n_neighbors=best_k, metric=best_metric, n_jobs=-1)
knn_final.fit(X_train_sub, y_train_sub)
pred_knn = knn_final.predict(X_test)

pred_dt = dt_best.predict(X_test)

joblib.dump(knn_final, f"{MODELS_DIR}/knn.pkl")
joblib.dump(dt_best, f"{MODELS_DIR}/decision_tree.pkl")

results = pd.DataFrame([
    {
        'Model': f'KNN (k={best_k}, {best_metric})',
        'Test MAE': mean_absolute_error(y_test, pred_knn),
        'Test MSE': mean_squared_error(y_test, pred_knn),
        'Test RMSE': np.sqrt(mean_squared_error(y_test, pred_knn)),
        'Test R²': r2_score(y_test, pred_knn)
    },
    {
        'Model': f'Decision Tree (depth={best_depth_label})',
        'Test MAE': mean_absolute_error(y_test, pred_dt),
        'Test MSE': mean_squared_error(y_test, pred_dt),
        'Test RMSE': np.sqrt(mean_squared_error(y_test, pred_dt)),
        'Test R²': r2_score(y_test, pred_dt)
    }
])
print(results.to_string(index=False))
results.to_csv(f"{MODELS_DIR}/results_knn_tree.csv", index=False)

print("\nKNN & Decision Tree tamamlandı. Sıradaki: 08_evaluation.py")
