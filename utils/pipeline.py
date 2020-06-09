from joblib import load
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from utils.service import convert_to_dict


import numpy as np
import re

MULTINOMIAL_NB = 'core/pretrained/mnb'
GRADIENT_BOOST = 'core/pretrained/gbc'
LSTM = 'core/pretrained/lstm'
CNN = 'core/pretrained/cnn'


class Pipeline:
    def __init__(self, textlines):
        self.textlines = textlines

    def word_to_vector(self, model='mnb'):
        if model == 'gbc':
            count_vect = load(f'{GRADIENT_BOOST}/count_vec.joblib')
            tfidf_transformer = load(
                f'{GRADIENT_BOOST}/tfidf_transformer.joblib')
            counts = count_vect.transform(self.textlines)
            tfidfs = tfidf_transformer.transform(counts)
            return tfidfs
        else:
            count_vect = load(f'{MULTINOMIAL_NB}/count_vect.joblib')
            counts = count_vect.transform(self.textlines)
            return counts

    def gb_clf(self):
        clf = load(f'{GRADIENT_BOOST}/classifier.joblib')
        encoder = load(f'{GRADIENT_BOOST}/encoder.joblib')
        word_vect = self.word_to_vector(model='gbc')
        predicts = clf.predict(word_vect)

        group = dict(zip(encoder.transform(
            encoder.classes_), encoder.classes_))

        result = list()
        for text, category in zip(self.textlines, predicts):
            if re.sub(r'\s+', '', text).isnumeric():
                category = 'card'
            else:
                category = group[category]

            result.append((text, category))

        return result

    def mnb_clf(self):
        clf = load(f'{MULTINOMIAL_NB}/mnb.joblib')
        encoder = load(f'{MULTINOMIAL_NB}/encoder.joblib')
        word_vect = self.word_to_vector(model='mnb')
        predicts = clf.predict(word_vect)

        group = dict(zip(encoder.transform(
            encoder.classes_), encoder.classes_))

        result = list()
        for text, category in zip(self.textlines, predicts):
            if re.sub(r'\s+', '', text).isnumeric():
                category = 'nik'
            else:
                category = group[category]

            result.append((category, text))

        print(result)

        result = convert_to_dict(result)

        return result

    def lstm_clf(self):
        tokenizer = load(f'{LSTM}/tokenizer.joblib')
        model = load_model(f'{LSTM}/lstm_model.h5')
        seq = tokenizer.texts_to_sequences(self.textlines)
        padded = pad_sequences(seq, maxlen=5)
        predicts = model.predict(padded)

        labels = ['address', 'card', 'name', 'ttl']
        result = list()

        for text, category in zip(self.textlines, predicts):
            result.append((text, labels[np.argmax(category)]))

        return result

    def cnn_clf(self):
        tokenizer = load(f'{CNN}/tokenizer.joblib')
        model = load_model(f'{CNN}/cnn_model.h5')
        seq = tokenizer.texts_to_sequences(self.textlines)
        padded = pad_sequences(seq, maxlen=5)
        predicts = model.predict(padded)

        labels = ['address', 'name', 'ttl']
        result = list()

        for text, category in zip(self.textlines, predicts):
            result.append((text, labels[np.argmax(category)]))

        return result

    def classifier(self, model='mnb'):
        self.textlines = list(filter(None, self.textlines))

        if model == 'lstm':
            return self.lstm_clf()
        elif model == 'cnn':
            return self.cnn_clf()
        elif model == 'mnb':
            return self.mnb_clf()
        else:
            return self.gb_clf()
