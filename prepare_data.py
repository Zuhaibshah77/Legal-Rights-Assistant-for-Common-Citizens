import json
import pandas as pd
from datasets import load_dataset
from sklearn.utils import resample

# ── 1. Load ILC dataset ───────────────────────────────────
print("Loading ILC dataset...")
dataset = load_dataset("d0r1h/ILC")
train_df = pd.DataFrame(dataset['train'])
test_df  = pd.DataFrame(dataset['test'])

train_df['input_text'] = train_df['Title'] + ' ' + train_df['Summary']
test_df['input_text']  = test_df['Title'] + ' ' + test_df['Summary']

def assign_label(text):
    text = text.lower()
    if any(w in text for w in ['murder', 'criminal', 'fir', 'accused', 'bail', 'arrest', 'offence', 'crime', 'conviction', 'sentence']):
        return 'Criminal Law'
    elif any(w in text for w in ['tax', 'income tax', 'gst', 'customs', 'excise', 'revenue']):
        return 'Tax Law'
    elif any(w in text for w in ['labour', 'worker', 'employment', 'workman', 'industrial dispute', 'wage', 'retrenchment']):
        return 'Labour Law'
    elif any(w in text for w in ['constitution', 'fundamental right', 'article 14', 'article 21', 'writ', 'high court', 'petition']):
        return 'Constitutional Law'
    elif any(w in text for w in ['divorce', 'marriage', 'custody', 'matrimonial', 'husband', 'wife', 'adoption']):
        return 'Family Law'
    else:
        return 'Civil Law'

train_df['label'] = train_df['input_text'].apply(assign_label)
test_df['label']  = test_df['input_text'].apply(assign_label)
train_df['label'] = train_df['label'].replace('Family Law', 'Civil Law')
test_df['label']  = test_df['label'].replace('Family Law', 'Civil Law')

ilc_train = train_df[['input_text', 'label']]
ilc_test  = test_df[['input_text', 'label']]

# ── 2. Load Kaggle JSON files ──────────────────────────────
print("Loading Kaggle JSON files...")
with open('data/raw/constitution_qa.json', encoding='utf-8') as f:
    constitution = json.load(f)
with open('data/raw/crpc_qa.json', encoding='utf-8') as f:
    crpc = json.load(f)
with open('data/raw/ipc_qa.json', encoding='utf-8') as f:
    ipc = json.load(f)

const_df = pd.DataFrame(constitution)
const_df['input_text'] = const_df['question'] + ' ' + const_df['answer']
const_df['label'] = 'Constitutional Law'

crpc_df = pd.DataFrame(crpc)
crpc_df['input_text'] = crpc_df['question'] + ' ' + crpc_df['answer']
crpc_df['label'] = 'Criminal Law'

ipc_df = pd.DataFrame(ipc)
ipc_df['input_text'] = ipc_df['question'] + ' ' + ipc_df['answer']
ipc_df['label'] = 'Criminal Law'

kaggle_df = pd.concat([
    const_df[['input_text', 'label']],
    crpc_df[['input_text', 'label']],
    ipc_df[['input_text', 'label']]
], ignore_index=True)

# ── 3. Load PDF extracted data ────────────────────────────
print("Loading PDF extracted data...")
pdf_df = pd.read_csv('data/raw/pdf_extracted_data.csv')
print(pdf_df['label'].value_counts())

# ── 4. Combine all datasets ────────────────────────────────
print("\nCombining all datasets...")
full_df = pd.concat([ilc_train, kaggle_df, pdf_df], ignore_index=True)

print("\n--- BEFORE BALANCING ---")
print(full_df['label'].value_counts())

# ── 5. Balance to 2000 per class ──────────────────────────
TARGET = 2000
balanced_dfs = []

for label in full_df['label'].unique():
    class_df = full_df[full_df['label'] == label]
    if len(class_df) >= TARGET:
        sampled = resample(class_df, replace=False, n_samples=TARGET, random_state=42)
    else:
        sampled = resample(class_df, replace=True, n_samples=TARGET, random_state=42)
    balanced_dfs.append(sampled)

balanced_df = pd.concat(balanced_dfs, ignore_index=True)
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

print("\n--- AFTER BALANCING ---")
print(balanced_df['label'].value_counts())
print(f"\nTotal training samples : {len(balanced_df)}")
print(f"Total testing samples  : {len(ilc_test)}")

# ── 6. Save ────────────────────────────────────────────────
balanced_df.to_csv('data/raw/train.csv', index=False)
ilc_test.to_csv('data/raw/test.csv', index=False)
print("\n✅ Saved final balanced dataset!")