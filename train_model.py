import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.corpus import movie_reviews

print("Loading NLTK Movie Reviews Dataset...")
print("=" * 60)

# Download if not already present
try:
    nltk.data.find('corpora/movie_reviews')
except LookupError:
    print("Downloading movie reviews corpus (first time only)...")
    nltk.download('movie_reviews')

# Load actual NLTK movie reviews dataset
# This is the real Pang & Lee (2005) dataset with 2000 actual reviews
texts = []
labels = []

# Extract reviews from both categories
for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        texts.append(movie_reviews.raw(fileid))
        # 0 = negative, 1 = positive
        labels.append(0 if category == 'neg' else 1)

labels = np.array(labels)

# Print dataset statistics
print(f"\n✓ Successfully loaded NLTK Movie Reviews Corpus")
print(f"\nDataset Statistics:")
print(f"Total samples: {len(texts)}")
print(f"Classes: 2 (Binary Classification)")
for label_val in [0, 1]:
    count = sum(1 for l in labels if l == label_val)
    sentiment = "Negative" if label_val == 0 else "Positive"
    print(f"  {sentiment}: {count} samples")

# Print sample texts (first few lines)
print(f"\nSample Review (Negative):")
print(f"  {texts[0][:150]}...")
print(f"\nSample Review (Positive):")
print(f"  {texts[1000][:150]}...")

# Train vectorizer
print(f"\n{'=' * 60}")
print("Training TF-IDF Vectorizer...")
vectorizer = TfidfVectorizer(
    max_features=1000,
    lowercase=True,
    stop_words='english',
    ngram_range=(1, 2),  # Unigrams and bigrams
    min_df=2,
    max_df=0.8,
    sublinear_tf=True
)
X = vectorizer.fit_transform(texts)

print(f"✓ Vectorization complete")
print(f"  Feature matrix shape: {X.shape}")
print(f"  Vocabulary size: {len(vectorizer.get_feature_names_out())}")

# Train model
print(f"\n{'=' * 60}")
print("Training Logistic Regression Model...")
model = LogisticRegression(
    random_state=42,
    max_iter=1000,
    solver='lbfgs'
)
model.fit(X, labels)

# Evaluate on training data
train_score = model.score(X, labels)
print(f"✓ Training complete")
print(f"  Training accuracy: {train_score:.2%}")

# Save model and vectorizer
print(f"\n{'=' * 60}")
print("Saving Model and Vectorizer...")
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print(f"✓ Files saved successfully!")
print(f"  - model.pkl: Trained Logistic Regression classifier")
print(f"  - vectorizer.pkl: TF-IDF vectorizer (learned from 2000 reviews)")
print(f"\n{'=' * 60}")