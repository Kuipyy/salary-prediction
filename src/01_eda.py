"""01_eda.py - Keşifsel Veri Analizi"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

DATA_PATH = "data/raw/job_salary_prediction_dataset.csv"
FIGURES_DIR = "outputs/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

print("=" * 80)
print("1. VERİ YÜKLENİYOR")
print("=" * 80)

df = pd.read_csv(DATA_PATH)
print(f"Shape: {df.shape}")
print("\n--- İlk 5 satır ---")
print(df.head())
print("\n--- Veri tipleri ---")
print(df.dtypes)
print("\n--- Eksik değerler ---")
print(df.isnull().sum())
print(f"\nTekrar eden satır: {df.duplicated().sum()}")

print("\n" + "=" * 80)
print("2. SAYISAL DEĞİŞKENLER")
print("=" * 80)
print(df.describe())

print("\n" + "=" * 80)
print("3. KATEGORİK DEĞİŞKENLER")
print("=" * 80)

categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    print(f"\n--- {col} ({df[col].nunique()} benzersiz değer) ---")
    print(df[col].value_counts())

print("\n" + "=" * 80)
print("4. HEDEF DEĞİŞKEN: SALARY")
print("=" * 80)
print(f"Min: {df['salary'].min():,}")
print(f"Max: {df['salary'].max():,}")
print(f"Mean: {df['salary'].mean():,.0f}")
print(f"Median: {df['salary'].median():,.0f}")
print(f"Std: {df['salary'].std():,.0f}")
print(f"Skewness: {df['salary'].skew():.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df['salary'], bins=50, color='steelblue', edgecolor='black')
axes[0].set_title('Salary Distribution (Histogram)', fontweight='bold')
axes[0].set_xlabel('Annual Salary')
axes[0].set_ylabel('Frequency')
axes[0].axvline(df['salary'].mean(), color='red', linestyle='--',
                label=f'Mean: {df["salary"].mean():,.0f}')
axes[0].axvline(df['salary'].median(), color='green', linestyle='--',
                label=f'Median: {df["salary"].median():,.0f}')
axes[0].legend()

axes[1].boxplot(df['salary'], vert=True)
axes[1].set_title('Salary Boxplot', fontweight='bold')
axes[1].set_ylabel('Annual Salary')

plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/01_salary_distribution.png", dpi=100, bbox_inches='tight')
plt.close()
print(f"\nKaydedildi: {FIGURES_DIR}/01_salary_distribution.png")

print("\n" + "=" * 80)
print("5. SAYISAL DEĞİŞKENLER vs SALARY (SCATTER)")
print("=" * 80)

numerical_cols = ['experience_years', 'skills_count', 'certifications']
sample = df.sample(5000, random_state=42)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, col in enumerate(numerical_cols):
    axes[i].scatter(sample[col], sample['salary'], alpha=0.3, s=10, color='steelblue')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('salary')
    axes[i].set_title(f'{col} vs salary', fontweight='bold')

plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/02_numerical_vs_salary.png", dpi=100, bbox_inches='tight')
plt.close()
print(f"Kaydedildi: {FIGURES_DIR}/02_numerical_vs_salary.png")

print("\n" + "=" * 80)
print("6. KATEGORİK DEĞİŞKENLER vs SALARY")
print("=" * 80)

for col in categorical_cols:
    group_means = df.groupby(col)['salary'].mean().sort_values(ascending=False)
    print(f"\n--- {col} ortalama maaş ---")
    print(group_means)

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.flatten()

for i, col in enumerate(categorical_cols):
    order = df.groupby(col)['salary'].median().sort_values().index
    sns.boxplot(data=df, x=col, y='salary', ax=axes[i], order=order,
                hue=col, palette='viridis', legend=False)
    axes[i].set_title(f'{col} vs salary', fontweight='bold')
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/03_categorical_vs_salary.png", dpi=100, bbox_inches='tight')
plt.close()
print(f"\nKaydedildi: {FIGURES_DIR}/03_categorical_vs_salary.png")

print("\nEDA tamamlandı. Sıradaki: 02_feature_analysis.py")
