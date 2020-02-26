from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from joblib import load

import re


class Pipeline:
    def __init__(self, textlines):
        self.textlines = textlines

    def transforming(self):
        count_vect = load('core/pretrained/count_vec.joblib')
        tfidf_transformer = load('core/pretrained/tfidf_transformer.joblib')
        counts = count_vect.transform(self.textlines)
        tfidfs = tfidf_transformer.transform(counts)
        return tfidfs

    def predicted(self):
        result = list()

        clf = load('core/pretrained/classifier.joblib')
        encoder = load('core/pretrained/encoder.joblib')

        transformed = self.transforming()
        predicts = clf.predict(transformed)
        group = dict(zip(encoder.transform(
            encoder.classes_), encoder.classes_))

        for doc, category in zip(self.textlines, predicts):
            if re.sub(r'\s+', '',doc).isnumeric():
                category = 'card'
            else:
                category = group[category]
            
            result.append('Entity : {} => Predict type : {}'.format(doc, category))

        return result