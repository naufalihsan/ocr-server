from joblib import load
from talos import Restore
from keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from utils.service import convert_to_dict

import numpy as np
import pandas as pd
import re


COMMON = 'core/pretrained/common'
MULTINOMIAL_NB = 'core/pretrained/mnb/new'
GRADIENT_BOOST = 'core/pretrained/gbc/new'
LSTM = 'core/pretrained/lstm/new'


class Pipeline:
    def __init__(self, textlines):
        self.textlines = textlines
        self.encoder = load(f'{COMMON}/encoder.joblib')
        self.tokenizer = load(f'{COMMON}/tokenizer.joblib')

    def transform(self, texts):
        sequence = self.tokenizer.texts_to_sequences(texts)
        return pad_sequences(sequence, maxlen=12)

    def evaluate(self, texts, predict_words, model=None):
        evals = list()
        classes = dict(zip(
            self.encoder.transform(self.encoder.classes_),
            self.encoder.classes_
        ))

        for entity, predict in zip(texts, predict_words):
            if model:
                predict = np.argmax(predict)

            if re.sub(r'\s+', '', entity).isnumeric():
                category = 'nik'
            else:
                category = classes[predict]

            evals.append((entity, category))

        print('Detailed prediction report')
        print()
        print('The text is predicted using LSTM model')
        print()
        print('{:<26} | {:<10}'.format('entity', 'category'))
        print('{:<26} | {:<10}'.format('-'*26, '-'*10))
        for i in range(3, len(evals)):
            print('{:<26} | {:<10}'.format(evals[i][0], evals[i][1]))
        print()

        return convert_to_dict(evals)

    def mnb(self):
        mnb_cv = load(f'{MULTINOMIAL_NB}/mnb_cv.joblib')
        predict_words = mnb_cv.predict(pd.Series(np.array(self.textlines)))
        evals = self.evaluate(self.textlines, predict_words)
        return evals

    def gbc(self):
        gbc_cv = load(f'{GRADIENT_BOOST}/gbc_cv.joblib')
        predict_words = gbc_cv.predict(pd.Series(np.array(self.textlines)))
        evals = self.evaluate(self.textlines, predict_words)
        return evals

    def lstm(self):
        lstm_cv = Restore(f'{LSTM}/lstm_deploy_3.zip')
        text_sequence = self.transform(self.textlines)
        predict_words = lstm_cv.model.predict(text_sequence)
        evals = self.evaluate(self.textlines, predict_words, model='lstm')
        return evals

    def predicts(self, model='mnb'):
        self.textlines = list(filter(None, self.textlines))
        
        if model == 'lstm':
            return self.lstm()
        elif model == 'mnb':
            return self.mnb()
        else:
            return self.gbc()
