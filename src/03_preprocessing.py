"""03_preprocessing.py - Train/Val/Test split, One-Hot Encoding, StandardScaler"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
import os

DATA_PATH = "data/raw/job_salary_prediction_dataset.csv"
PROCESSED_DIR = "data/processed"
MODELS_DIR = "outputs/models"
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

print("=" * 80)
print("1. VERİ YÜKLENİYOR")
print("=" * 80)

df = pd.read_csv(DATA_PATH)
print(f"Shape: {df.shape}")
print(f"Eksik değer: {df.isnull().sum().sum()}")

print("\n" + "=" * 80)
print("2. X / y AYRIMI")
print("=" * 80)

y = df['salary']
X = df.drop(columns=['salary'])
print(f"X: {X.shape}, y: {y.shape}")

print("\n" + "=" * 80)
print("3. TRAIN / VALIDATION / TEST AYRIMI (60/20/20)")
print("=" * 80)

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42
)

print(f"Train: {X_train.shape[0]} ({X_train.shape[0]/len(X)*100:.0f}%)")
print(f"Val:   {X_val.shape[0]} ({X_val.shape[0]/len(X)*100:.0f}%)")
print(f"Test:  {X_test.shape[0]} ({X_test.shape[0]/len(X)*100:.0f}%)")

print("\n" + "=" * 80)
print("4. PREPROCESSING PIPELINE")
print("=" * 80)

numerical_cols = ['experience_years', 'skills_count', 'certifications']
categorical_cols = ['job_title', 'education_level', 'industry',
                    'company_size', 'location', 'remote_work']

print(f"Sayısal: {numerical_cols}")
print(f"Kategorik: {categorical_cols}")

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'),
         categorical_cols)
    ],
    remainder='drop'
)

print("\nTrain üzerinde fit ediliyor (data leakage önlemi)...")
X_train_processed = preprocessor.fit_transform(X_train)
print("Val ve test üzerinde transform uygulanıyor...")
X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)

print(f"\nTrain processed: {X_train_processed.shape}")
print(f"Val processed: {X_val_processed.shape}")
print(f"Test processed: {X_test_processed.shape}")

feature_names = preprocessor.get_feature_names_out()
feature_names = [name.replace('num__', '').replace('cat__', '') for name in feature_names]
print(f"Toplam özellik sayısı: {len(feature_names)}")

print("\n" + "=" * 80)
print("5. KAYDETME")
print("=" * 80)

np.save(f"{PROCESSED_DIR}/X_train.npy", X_train_processed)
np.save(f"{PROCESSED_DIR}/X_val.npy", X_val_processed)
np.save(f"{PROCESSED_DIR}/X_test.npy", X_test_processed)
np.save(f"{PROCESSED_DIR}/y_train.npy", y_train.values)
np.save(f"{PROCESSED_DIR}/y_val.npy", y_val.values)
np.save(f"{PROCESSED_DIR}/y_test.npy", y_test.values)

X_train.to_csv(f"{PROCESSED_DIR}/X_train_raw.csv", index=False)
X_val.to_csv(f"{PROCESSED_DIR}/X_val_raw.csv", index=False)
X_test.to_csv(f"{PROCESSED_DIR}/X_test_raw.csv", index=False)

joblib.dump(preprocessor, f"{MODELS_DIR}/preprocessor.pkl")

with open(f"{PROCESSED_DIR}/feature_names.txt", "w") as f:
    for name in feature_names:
        f.write(name + "\n")

print(f"Kaydedildi:")
print(f"  {PROCESSED_DIR}/X_train.npy, X_val.npy, X_test.npy")
print(f"  {PROCESSED_DIR}/y_train.npy, y_val.npy, y_test.npy")
print(f"  {PROCESSED_DIR}/feature_names.txt")
print(f"  {MODELS_DIR}/preprocessor.pkl")

print("\nPreprocessing tamamlandı. Sıradaki: 04_linear_models.py")
