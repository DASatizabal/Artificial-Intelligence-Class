"""Intent classifiers.

  kind="bow"   : the starter baseline, bag-of-words counts + logistic regression
  kind="tfidf" : TF-IDF with word unigrams and bigrams (so "in network" and "no cardiologist" are seen as phrases)
Both return a confidence (the top class probability) so the rule base can ask for clarification when unsure.
"""
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class IntentClassifier:
    def __init__(self, kind="bow", seed=42):
        self.kind = kind
        if kind == "bow":
            self.vec = CountVectorizer(lowercase=True)
        else:
            self.vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True, min_df=1)
        self.clf = LogisticRegression(max_iter=1000, random_state=seed)

    def fit(self, texts, labels):
        self.clf.fit(self.vec.fit_transform(texts), labels)
        return self

    def predict(self, texts):
        return self.clf.predict(self.vec.transform(texts))

    def predict_proba(self, texts):
        return self.clf.predict_proba(self.vec.transform(texts))

    @property
    def classes_(self):
        return self.clf.classes_

    def top_terms(self, n=6):
        """Highest-weight words per intent (a simple look inside the model)."""
        vocab = self.vec.get_feature_names_out()
        return {c: [vocab[j] for j in self.clf.coef_[i].argsort()[::-1][:n]] for i, c in enumerate(self.clf.classes_)}
