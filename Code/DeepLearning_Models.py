
import pickle
import pandas as pd
import numpy as np
from random import shuffle
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Embedding, Input, RepeatVector
from keras.utils import np_utils
from keras.preprocessing import text, sequence
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import pandas as pd
import numpy as np


from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer
import nltk
import re
from nltk.corpus import stopwords
import os


def lsa_feature(data):
	svd = TruncatedSVD(100)
	lsa = make_pipeline(svd, Normalizer(copy=False))
	vectorizer = TfidfVectorizer(max_features=1000,min_df=2, stop_words='english',use_idf=True)
	X_train_tfidf = vectorizer.fit_transform(data)
	X_train_lsa = lsa.fit_transform(X_train_tfidf)
	return X_train_lsa



def baseline_model():
    model = Sequential()
    model.add(Dense(512, input_dim=training_data.shape[1]+testing_data.shape[1], init='normal', activation='relu'))
    model.add(Dropout(0.15))
    model.add(Dense(512, init='normal', activation='relu'))
    model.add(Dropout(0.15))
    model.add(Dense(512, init='normal', activation='relu'))
    model.add(Dropout(0.15))
    model.add(Dense(512, init='normal', activation='relu'))
    model.add(Dense(256, init='normal', activation='relu'))
    model.add(Dense(64, init='normal', activation='relu'))
    model.add(Dense(9, init='normal', activation="softmax"))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_crossentropy'])
    return model

#Loading Data 
N = 10
data = pd.read_csv("../Data/Prcoessed_Data.csv",sep="|",nrows=N)
target = data.CODE
print("Data Loading Done")
X = data['PROCESSED_TEXT']
Y = data['PROCESSED_TEXT_TEST']


vectorizer = TfidfVectorizer().fit(X)

tfidf_vector_X = vectorizer.transform(X).toarray()  #//shape - (3,6)
tfidf_vector_Y = vectorizer.transform(data.CODE).toarray() #//shape - (3,6)
tfidf_vector_X = tfidf_vector_X[:, :, None] #//shape - (3,6,1) 
tfidf_vector_Y = tfidf_vector_Y[:, :, None] #//shape - (3,6,1)


from keras import Sequential
from keras.layers import LSTM

model = Sequential()
model.add(LSTM(units=6, input_shape = tfidf_vector_X.shape[1:], return_sequences = True))
model.add(LSTM(units=4, return_sequences=True))
model.add(LSTM(units=1, return_sequences=True, name='output'))
model.compile(loss='cosine_proximity', optimizer='sgd', metrics = ['accuracy'])
print(model.summary())
model.fit(tfidf_vector_X, data.CODE, epochs=10, verbose=1)
print(model.predict_proba(tfidf_vector_X))



