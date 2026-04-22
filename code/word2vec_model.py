import numpy as np
import gensim
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
import joblib

class Word2VecModel:
    def __init__(self, vector_size=300, window=5, min_count=10, workers=4, 
                 sg=1, epochs=10, classifier='lr'):
        self.w2v_params = {
            'vector_size': vector_size,
            'window': window,
            'min_count': min_count,
            'workers': workers,
            'sg': sg,
            'epochs': epochs
        }
        self.w2v_model = None
        self.classifier = self._init_classifier(classifier)
    
    def _init_classifier(self, clf_type):
        if clf_type == 'lr':
            return LogisticRegression(C=1.0, max_iter=500, random_state=42)
        elif clf_type == 'rf':
            return RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        return LogisticRegression(C=1.0, max_iter=500, random_state=42)
    
    def train_word2vec(self, sentences):
        self.w2v_model = Word2Vec(sentences=sentences, **self.w2v_params)
        return self.w2v_model
    
    def get_sentence_vector(self, words):
        embedding_dim = self.w2v_params['vector_size']
        feature_vector = np.zeros(embedding_dim, dtype=np.float32)
        valid_words = 0
        
        vocab = set(self.w2v_model.wv.index_to_key)
        for word in words:
            if word in vocab:
                feature_vector += self.w2v_model.wv[word]
                valid_words += 1
        
        return feature_vector / valid_words if valid_words > 0 else feature_vector
    
    def transform(self, tokenized_texts):
        embedding_dim = self.w2v_params['vector_size']
        features = np.zeros((len(tokenized_texts), embedding_dim), dtype=np.float32)
        for i, text in enumerate(tokenized_texts):
            features[i] = self.get_sentence_vector(text)
        return features
    
    def fit(self, tokenized_texts, y, train_word2vec=True):
        if train_word2vec:
            self.train_word2vec(tokenized_texts)
        
        X = self.transform(tokenized_texts)
        self.classifier.fit(X, y)
        return self
    
    def predict_proba(self, tokenized_texts):
        X = self.transform(tokenized_texts)
        return self.classifier.predict_proba(X)[:, 1]
    
    def predict(self, tokenized_texts):
        X = self.transform(tokenized_texts)
        return self.classifier.predict(X)
    
    def evaluate(self, tokenized_texts, y):
        y_pred_proba = self.predict_proba(tokenized_texts)
        return roc_auc_score(y, y_pred_proba)
    
    def save(self, word2vec_path='word2vec.model', classifier_path='word2vec_classifier.pkl'):
        if self.w2v_model:
            self.w2v_model.save(word2vec_path)
        joblib.dump(self.classifier, classifier_path)
    
    def load(self, word2vec_path='word2vec.model', classifier_path='word2vec_classifier.pkl'):
        self.w2v_model = Word2Vec.load(word2vec_path)
        self.classifier = joblib.load(classifier_path)
