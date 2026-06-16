"""02_feature_analysis.py - Korelasyon analizi ve feature seçim kararı"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_style("whitegrid")

DATA_PATH = "data/raw/job_salary_prediction_dataset.csv"
FIGURES_DIR = "outputs/figures"
MODELS_DIR = "outputs/models"
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

print("=" * 80)
print("1. VERİ YÜKLENİYOR")
print("=" * 80)

df = pd.read_csv(DATA_PATH)
print(f"Shape: {df.shape}")

print("\n" + "=" * 80)
print("2. SAYISAL DEĞİŞKENLER - PEARSON KORELASYON")
print("=" * 80)

numerical_cols = ['experience_years', 'skills_count', 'certifications', 'salary']
corr_numerical = df[numerical_cols].corr()
print(corr_numerical)

plt.figure(figsize=(8, 6))
sns.heatmap(corr_numerical, annot=True, cmap='coolwarm', center=0,
            fmt='.4f', square=True, linewidths=1, vmin=-1, vmax=1)
plt.title('Sayısal Değişkenler Korelasyon Matrisi', fontweight='bold')
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/04_corr_numerical.png", dpi=100, bbox_inches='tight')
plt.close()
print(f"\nKaydedildi: {FIGURES_DIR}/04_corr_numerical.png")

print("\n" + "=" * 80)
print("3. TÜM DEĞİŞKENLER - ONE-HOT SONRASI KORELASYON")
print("=" * 80)

df_encoded = pd.get_dummies(df, drop_first=True, dtype=float)
print(f"One-hot sonrası shape: {df_encoded.shape}")

corr_full = df_encoded.corr()

target_corr = corr_full['salary'].drop('salary').sort_values(key=abs, ascending=False)
print("\n--- Salary ile korelasyonu (mutlak değere göre sıralı) ---")
print(target_corr)

target_corr.to_csv(f"{MODELS_DIR}/feature_target_correlation.csv")

plt.figure(figsize=(10, 12))
colors = ['#d62728' if x < 0 else '#1f77b4' for x in target_corr.values]
plt.barh(range(len(target_corr)), target_corr.values, color=colors, edgecolor='black')
plt.yticks(range(len(target_corr)), target_corr.index)
plt.axvline(x=0, color='black', linewidth=0.8)
plt.xlabel('Pearson Korelasyon (salary ile)')
plt.title('Her Özelliğin Salary ile Korelasyonu', fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/05_target_correlation.png", dpi=100, bbox_inches='tight')
plt.close()
print(f"\nKaydedildi: {FIGURES_DIR}/05_target_correlation.png")

print("\n" + "=" * 80)
print("4. ÖZELLİKLER ARASI MULTİCOLLİNEARİTY KONTROLÜ")
print("=" * 80)

features_only = corr_full.drop(columns=['salary']).drop(index=['salary'])

mask = np.triu(np.ones_like(features_only, dtype=bool), k=1)
high_corr_pairs = []
for i in range(len(features_only.columns)):
    for j in range(i+1, len(features_only.columns)):
        corr_val = features_only.iloc[i, j]
        if abs(corr_val) > 0.7:
            high_corr_pairs.append({
                'Feature 1': features_only.columns[i],
                'Feature 2': features_only.columns[j],
                'Correlation': corr_val
            })

if high_corr_pairs:
    high_corr_df = pd.DataFrame(high_corr_pairs)
    print("Yüksek korelasyonlu (|r| > 0.7) çiftler:")
    print(high_corr_df)
else:
    print("Özellikler arasında |r| > 0.7 olan çift YOK.")
    print("→ Multicollinearity sorunu görülmedi.")

plt.figure(figsize=(20, 16))
sns.heatmap(corr_full, annot=False, cmap='coolwarm', center=0,
            square=True, linewidths=0.3, vmin=-1, vmax=1,
            cbar_kws={'shrink': 0.8})
plt.title('Tüm Değişkenler Korelasyon Matrisi (One-Hot sonrası)', fontweight='bold')
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/06_corr_full_heatmap.png", dpi=100, bbox_inches='tight')
plt.close()
print(f"\nKaydedildi: {FIGURES_DIR}/06_corr_full_heatmap.png")

print("\n" + "=" * 80)
print("5. FEATURE SEÇİM KARARI")
print("=" * 80)

WEAK_THRESHOLD = 0.02
weak_features = target_corr[target_corr.abs() < WEAK_THRESHOLD]

print(f"\nSalary ile mutlak korelasyonu < {WEAK_THRESHOLD} olan özellikler:")
if len(weak_features) > 0:
    print(weak_features)
else:
    print(f"Yok. Tüm özelliklerin salary ile en az {WEAK_THRESHOLD} korelasyonu var.")

print("\n--- KARAR ---")
print(f"""
Feature seçim analizinin sonuçları:

1. Özellikler arası multicollinearity: {'VAR' if high_corr_pairs else 'YOK'} (|r| > 0.7 eşiği)
2. Salary ile zayıf korelasyonlu özellik (|r| < {WEAK_THRESHOLD}): {len(weak_features)} adet
3. Toplam özellik sayısı (one-hot sonrası): {df_encoded.shape[1] - 1}

Karar gerekçesi:
- Multicollinearity sorunu yok, hiçbir özellik diğeriyle yüksek korele değil.
- {'Bazı özelliklerin salary ile korelasyonu çok zayıf' if len(weak_features) > 0 else 'Tüm özelliklerin salary ile makul korelasyonu var'}.
- Ders kapsamında Lasso (L1) regularization otomatik feature seçimi yapacak,
  yani manuel eksiltmeye gerek yok. Lasso, gereksiz özelliklerin katsayısını
  sıfıra çekerek bunu kendisi öğrenecek.
- Bu nedenle MANUEL FEATURE ELEMİNASYONU YAPILMIYOR, tüm özelliklerle devam.
- Lasso sonuçları (05_regularization.py) hangi özelliklerin gereksiz olduğunu
  retrospektif olarak gösterecek.
""")

print("Feature analizi tamamlandı. Sıradaki: 03_preprocessing.py")
