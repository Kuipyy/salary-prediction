"""09_interpretability.py - Linear katsayılar, Lasso eleme, Decision Tree feature importance"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

sns.set_style("whitegrid")

PROCESSED_DIR = "data/processed"
MODELS_DIR = "outputs/models"
FIGURES_DIR = "outputs/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

print("=" * 80)
print("1. MODELLER YÜKLENİYOR")
print("=" * 80)

linear = joblib.load(f"{MODELS_DIR}/multiple_linear.pkl")
ridge = joblib.load(f"{MODELS_DIR}/ridge.pkl")
lasso = joblib.load(f"{MODELS_DIR}/lasso.pkl")
dt = joblib.load(f"{MODELS_DIR}/decision_tree.pkl")

with open(f"{PROCESSED_DIR}/feature_names.txt") as f:
    feature_names = [line.strip() for line in f]

print(f"Feature sayısı: {len(feature_names)}")

print("\n" + "=" * 80)
print("2. LINEAR REGRESSION KATSAYILARI")
print("=" * 80)

linear_coefs = pd.DataFrame({
    'Feature': feature_names,
    'Linear': linear.coef_,
    'Ridge': ridge.coef_,
    'Lasso': lasso.coef_
})

linear_coefs['Abs_Linear'] = np.abs(linear_coefs['Linear'])
linear_coefs = linear_coefs.sort_values('Abs_Linear', ascending=False)
linear_coefs = linear_coefs.drop(columns=['Abs_Linear'])

print("\n--- En etkili 15 özellik (Linear katsayısına göre) ---")
print(linear_coefs.head(15).to_string(index=False))

linear_coefs.to_csv(f"{MODELS_DIR}/all_linear_coefficients.csv", index=False)

print("\n" + "=" * 80)
print("3. LASSO FEATURE ELİMİNASYONU")
print("=" * 80)

n_eliminated = (lasso.coef_ == 0).sum()
n_kept = (lasso.coef_ != 0).sum()

print(f"Toplam özellik: {len(feature_names)}")
print(f"Lasso'nun ELEDİĞİ: {n_eliminated}")
print(f"Lasso'nun KORUDUĞU: {n_kept}")

eliminated = [f for f, c in zip(feature_names, lasso.coef_) if c == 0]
if eliminated:
    print("\n--- Lasso'nun sıfıra çektiği özellikler ---")
    for f in eliminated:
        print(f"  - {f}")
else:
    print("\nLasso hiçbir özelliği elemiyor (alpha düşük).")

print("\n" + "=" * 80)
print("4. DECISION TREE FEATURE IMPORTANCE")
print("=" * 80)

dt_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': dt.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n--- En önemli 15 özellik (Decision Tree) ---")
print(dt_importance.head(15).to_string(index=False))

dt_importance.to_csv(f"{MODELS_DIR}/decision_tree_importance.csv", index=False)

print("\n" + "=" * 80)
print("5. KARŞILAŞTIRMALI GÖRSEL")
print("=" * 80)

fig, axes = plt.subplots(1, 3, figsize=(22, 10))

top_linear = linear_coefs.head(15).iloc[::-1]
colors_l = ['#d62728' if c < 0 else '#1f77b4' for c in top_linear['Linear']]
axes[0].barh(top_linear['Feature'], top_linear['Linear'],
             color=colors_l, edgecolor='black')
axes[0].axvline(x=0, color='black', linewidth=0.8)
axes[0].set_xlabel('Linear Katsayı')
axes[0].set_title('Linear Regression\n(en büyük 15 |katsayı|)', fontweight='bold')

lasso_nonzero = linear_coefs[linear_coefs['Lasso'] != 0].copy()
lasso_nonzero['Abs_Lasso'] = np.abs(lasso_nonzero['Lasso'])
lasso_nonzero = lasso_nonzero.sort_values('Abs_Lasso', ascending=False).head(15).iloc[::-1]
colors_la = ['#d62728' if c < 0 else '#1f77b4' for c in lasso_nonzero['Lasso']]
axes[1].barh(lasso_nonzero['Feature'], lasso_nonzero['Lasso'],
             color=colors_la, edgecolor='black')
axes[1].axvline(x=0, color='black', linewidth=0.8)
axes[1].set_xlabel('Lasso Katsayı')
axes[1].set_title(f'Lasso (L1)\n{n_kept} korundu / {n_eliminated} eleme',
                  fontweight='bold')

top_dt = dt_importance.head(15).iloc[::-1]
axes[2].barh(top_dt['Feature'], top_dt['Importance'],
             color='steelblue', edgecolor='black')
axes[2].set_xlabel('Importance')
axes[2].set_title('Decision Tree\n(Information Gain bazlı)', fontweight='bold')

plt.suptitle('Feature Önem Karşılaştırması (3 model)', fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/17_feature_importance_comparison.png", dpi=100, bbox_inches='tight')
plt.close()
print(f"Kaydedildi: {FIGURES_DIR}/17_feature_importance_comparison.png")

print("\n" + "=" * 80)
print("6. ORTAK EN ÖNEMLİ ÖZELLİKLER")
print("=" * 80)

top_linear_set = set(linear_coefs.head(10)['Feature'])
top_dt_set = set(dt_importance.head(10)['Feature'])
common = top_linear_set & top_dt_set

print(f"\nLinear'a göre en önemli 10: {sorted(top_linear_set)}")
print(f"\nDT'ye göre en önemli 10: {sorted(top_dt_set)}")
print(f"\nORTAK olanlar ({len(common)} adet): {sorted(common)}")

print("\n" + "=" * 80)
print("PROJE TAMAMLANDI")
print("=" * 80)
print("""
Tüm aşamalar tamamlandı:
  01_eda.py                  ✓
  02_feature_analysis.py     ✓
  03_preprocessing.py        ✓
  04_linear_models.py        ✓
  05_regularization.py       ✓
  06_polynomial.py           ✓
  07_knn_tree.py             ✓
  08_evaluation.py           ✓
  09_interpretability.py     ✓
""")
