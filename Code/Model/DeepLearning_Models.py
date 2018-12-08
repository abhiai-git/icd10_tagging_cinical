
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
import os,keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.metrics import roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix

import pickle
from sklearn.decomposition import TruncatedSVD
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import log_loss,confusion_matrix,classification_report,roc_curve,auc
from scipy import sparse
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import timeit
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier  
from sklearn import decomposition, ensemble
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier 


def lsa_feature(data):
	svd = TruncatedSVD(100)
	lsa = make_pipeline(svd, Normalizer(copy=False))
	vectorizer = TfidfVectorizer(max_features=1000,min_df=2, stop_words='english',use_idf=True)
	X_train_tfidf = vectorizer.fit_transform(data)
	X_train_lsa = lsa.fit_transform(X_train_tfidf)
	return X_train_lsa


alg_result=[]

#Loading Data 
N = 100
data = pd.read_csv("../../Data/Prcoessed_Data.csv",sep="|",nrows=N)
target = data.CODE
print("Data Loading Done")
X = data['PROCESSED_TEXT']
X_TEST = data['PROCESSED_TEXT_TEST']
alg_name ="LSTM_"+str(N)
max_words = 15000
max_len = 100
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X)
sequences = tok.texts_to_sequences(X)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)
print(sequences_matrix)
def RNN():
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(max_words,50,input_length=max_len)(inputs)
    layer = LSTM(64)(layer)
    layer = Dense(256,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1,name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model

yt = []
pt = []
thr = 0.002
plt.figure(figsize=(14,10))
plt.plot([0,1],[0,1],color='r',linestyle='-.')
start = timeit.default_timer()
for i,col in enumerate(target[:10]): 
    model = RNN()
    model.summary()
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.fit(sequences_matrix,data[target[i]],batch_size=64,epochs=100,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)],verbose=0)
    test_sequences = tok.texts_to_sequences(X_TEST)
    test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)
    accr = model.evaluate(test_sequences_matrix,data[target[i]])
    label  = list(data[target[i]])
    predict_prob = [x[0] for x in model.predict(test_sequences_matrix)]
    thr = max(predict_prob)
    predict_label = [1 if x >= thr else 0 for x in predict_prob]
    yt += label
    pt += predict_label
    print(label)
    print(predict_prob)
    print(predict_label)
    # print(model.predict_classes(test_sequences_matrix))
    # print(max(predict_prob))
    print('Test set Loss: {:0.3f} Accuracy: {:0.3f}'.format(accr[0],accr[1]))
frp,trp,thres = roc_curve(yt,pt)
auc_val =auc(frp,trp)
plt.plot(frp,trp,label= alg_name+' Threshold ='+str(thr)+' AUC = %.4f'%auc_val)
F1 = f1_score(yt, pt, average="macro")
print(precision_recall_curve(yt,pt)[2])
PRESCISION = round(precision_score(yt, pt, average="macro"),2)
RECALL = round(recall_score(yt, pt, average="macro"),2)
ACCURACY  = round(accuracy_score(yt, pt),2)
stop = timeit.default_timer()
alg_result.append({"ALG_NAME":alg_name,"AUC":auc_val,"RUNTIME":round(stop - start,3),"THRESHOLD":thr,"Average F-Score":F1,"PRESCISION":PRESCISION,"RECALL":RECALL,"ACCURACY":ACCURACY})
plt.legend(loc='lower right')
plt.xlabel('True positive rate')
plt.ylabel('False positive rate')
plt.title(alg_name+' Reciever Operating Characteristic')
plt.savefig('../../Result/GRAPHS/'+alg_name+'_ROC.png')
plt.show()
pd.DataFrame(alg_result).to_csv("../../Result/CSV/"+alg_name+".csv",index=False)


