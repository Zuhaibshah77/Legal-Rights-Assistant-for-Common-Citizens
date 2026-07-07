import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pickle

# ── 1. Load Data ──────────────────────────────────────────
print("Loading data...")
train_df = pd.read_csv('data/raw/train.csv')
test_df  = pd.read_csv('data/raw/test.csv')

# ── 2. Merge Family Law into Civil Law ────────────────────
train_df['label'] = train_df['label'].replace('Family Law', 'Civil Law')
test_df['label']  = test_df['label'].replace('Family Law', 'Civil Law')

print("\n--- LABEL DISTRIBUTION AFTER MERGE ---")
print(train_df['label'].value_counts())

# ── 3. Prepare X and y ────────────────────────────────────
X_train = train_df['input_text']
y_train = train_df['label']
X_test  = test_df['input_text']
y_test  = test_df['label']

# ── 4. TF-IDF Vectorization ───────────────────────────────
# Converts text into numbers the ML model can understand
print("\nVectorizing text with TF-IDF...")
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

print(f"Feature matrix shape: {X_train_vec.shape}")

# ── 5. Compute Class Weights ──────────────────────────────
# This fixes the imbalance problem
classes = np.unique(y_train)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, weights))
print(f"\nClass weights: {class_weight_dict}")

# ── 6. Train 3 Models ─────────────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
    'Random Forest'      : RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
    'SVM'                : SVC(kernel='linear', class_weight='balanced', random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

# ── 7. Pick Best Model ────────────────────────────────────
best_model_name = max(results, key=results.get)
best_model      = models[best_model_name]
print(f"\n✅ Best Model: {best_model_name} with accuracy {results[best_model_name]:.4f}")

# ── 8. Save Model and Vectorizer ──────────────────────────
with open('models/classifier.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('models/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("✅ Model saved to models/classifier.pkl")
print("✅ Vectorizer saved to models/vectorizer.pkl")