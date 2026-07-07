from datasets import load_dataset
import pandas as pd

# Load dataset
print("Loading dataset...")
dataset = load_dataset("d0r1h/ILC")

# Convert to dataframe
train_df = pd.DataFrame(dataset['train'])
test_df = pd.DataFrame(dataset['test'])

# Combine title + summary as our input text
train_df['input_text'] = train_df['Title'] + ' ' + train_df['Summary']
test_df['input_text'] = test_df['Title'] + ' ' + test_df['Summary']

# Rule-based label generation function
def assign_label(text):
    text = text.lower()
    if any(word in text for word in ['murder', 'criminal', 'fir', 'accused', 'bail', 'arrest', 'offence', 'crime', 'conviction', 'sentence']):
        return 'Criminal Law'
    elif any(word in text for word in ['tax', 'income tax', 'gst', 'customs', 'excise', 'revenue']):
        return 'Tax Law'
    elif any(word in text for word in ['labour', 'worker', 'employment', 'workman', 'industrial dispute', 'wage', 'retrenchment']):
        return 'Labour Law'
    elif any(word in text for word in ['constitution', 'fundamental right', 'article 14', 'article 21', 'writ', 'high court', 'petition']):
        return 'Constitutional Law'
    elif any(word in text for word in ['divorce', 'marriage', 'custody', 'matrimonial', 'husband', 'wife', 'adoption']):
        return 'Family Law'
    else:
        return 'Civil Law'

# Apply labels
print("Assigning labels...")
train_df['label'] = train_df['input_text'].apply(assign_label)
test_df['label'] = test_df['input_text'].apply(assign_label)

# Check label distribution
print("\n--- LABEL DISTRIBUTION (Training) ---")
print(train_df['label'].value_counts())

# Save to CSV
train_df[['input_text', 'label']].to_csv('data/raw/train.csv', index=False)
test_df[['input_text', 'label']].to_csv('data/raw/test.csv', index=False)

print("\n✅ Saved to data/raw/train.csv and data/raw/test.csv")