import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
import json

# Download NLTK resources if not already present
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    """Lemmatize, remove stopwords and punctuation using NLTK."""
    tokens = nltk.word_tokenize(text)
    filtered = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.lower() not in stop_words and token not in string.punctuation]
    return ' '.join(filtered)

# Load dataset
with open('news_dataset.json', 'r') as f:
    df = pd.read_json(f)

# Visualize class distribution before balancing
plt.figure(figsize=(6,4))
sns.countplot(y=df['category'])
plt.title('Original Class Distribution')
plt.tight_layout()
plt.savefig('original_class_distribution.png')
plt.close()

# Balance the dataset (4 classes, equal samples)
min_samples = min(df['category'].value_counts()[cat] for cat in ['BUSINESS','SPORTS','CRIME','SCIENCE'])
business = df[df.category == 'BUSINESS'].sample(min_samples, random_state=0)
sports = df[df.category == 'SPORTS'].sample(min_samples, random_state=0)
crime = df[df.category == 'CRIME'].sample(min_samples, random_state=0)
science = df[df.category == 'SCIENCE'].sample(min_samples, random_state=0)
dataset = pd.concat([business, sports, crime, science], axis=0)

# Map categories to numbers
target = {'BUSINESS': 0, 'SPORTS': 1, 'CRIME': 2, 'SCIENCE': 3}
dataset['category'] = dataset['category'].map(target)

# Preprocess text
print('Preprocessing text...')
dataset['text'] = dataset['text'].apply(preprocess)

# Split data
x_train, x_test, y_train, y_test = train_test_split(
    dataset.text, dataset.category, test_size=0.2, random_state=0, stratify=dataset.category
)

# Vectorizer for unigrams and bigrams
vectorizer = CountVectorizer(ngram_range=(1, 2))

# Models to compare
models = {
    'LinearSVC': LinearSVC(max_iter=2000, random_state=0),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=0),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=0)
}

results = {}
metrics_summary = []
for name, model in models.items():
    print(f'Training {name}...')
    pipeline = Pipeline([
        ('vect', vectorizer),
        ('clf', model)
    ])
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)
    # Save classification report
    report = classification_report(y_test, y_pred, target_names=target.keys(), output_dict=True)
    with open(f'{name}_classification_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target.keys(), yticklabels=target.keys())
    plt.title(f'{name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(f'{name}_confusion_matrix.png')
    plt.close()
    results[name] = report
    # Collect metrics for comparison
    metrics_summary.append({
        'Model': name,
        'Accuracy': report['accuracy'],
        'F1-score (macro)': report['macro avg']['f1-score']
    })

# Plot comparison of models
metrics_df = pd.DataFrame(metrics_summary)
metrics_df.set_index('Model', inplace=True)
metrics_df[['Accuracy', 'F1-score (macro)']].plot(kind='bar', figsize=(8,5))
plt.title('Model Comparison: Accuracy and F1-score (macro)')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('model_comparison.png')
plt.close()

print('Done! Classification reports, confusion matrices, and model comparison saved.')
