import re
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup

# 手动定义常用英文停用词
ENGLISH_STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
    'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
    'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
    'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having',
    'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
    'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for',
    'with', 'about', 'against', 'between', 'into', 'through', 'during',
    'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',
    'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
    'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
    'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
    'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'don',
    'should', 'now'
}

class TextPreprocessor:
    def __init__(self, method='stem'):
        self.stop_words = ENGLISH_STOPWORDS
        self.stemmer = PorterStemmer() if method == 'stem' else None
        self.use_stemming = (method == 'stem')
    
    def _remove_html(self, text):
        return BeautifulSoup(text, 'html.parser').get_text()
    
    def _normalize(self, text):
        text = re.sub(r'[^a-zA-Z]', ' ', text)
        return text.lower()
    
    def clean_text(self, text):
        text = self._remove_html(text)
        text = self._normalize(text)
        words = text.split()
        words = [w for w in words if w not in self.stop_words]
        
        if self.use_stemming and self.stemmer:
            words = [self.stemmer.stem(word) for word in words]
        
        return ' '.join(words)
    
    def tokenize(self, text):
        cleaned = self.clean_text(text)
        return cleaned.split()
