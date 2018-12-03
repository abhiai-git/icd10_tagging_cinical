
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
classifier = KNeighborsClassifier(n_neighbors=5)

def tfidf_vectorization(input_data):
	vect_word = TfidfVectorizer(max_features=1000, lowercase=True, analyzer='word',stop_words= 'english',ngram_range=(1,3),dtype=np.float32)
	vect_char = TfidfVectorizer(max_features=40000, lowercase=True, analyzer='char',stop_words= 'english',ngram_range=(3,6),dtype=np.float32)
	tr_vect = vect_word.fit_transform(input_data)
	tr_vect_char = vect_char.fit_transform(input_data)
	return sparse.hstack([tr_vect, tr_vect_char])

def lsa_feature(data):
	svd = TruncatedSVD(100)
	lsa = make_pipeline(svd, Normalizer(copy=False))
	vectorizer = TfidfVectorizer(max_features=1000,min_df=2, stop_words='english',use_idf=True)
	X_train_tfidf = vectorizer.fit_transform(data)
	X_train_lsa = lsa.fit_transform(X_train_tfidf)
	return X_train_lsa

def classifier(alg,alg_name,threshold,training_data,training_label,testing_data,testing_labels):
	X = training_data
	x_test = testing_data
	y=training_label
	y_test=testing_labels
	plt.figure(figsize=(14,10))
	plt.plot([0,1],[0,1],color='r',linestyle='-.')
	alg_result=[]
	for thr in threshold:
		print(thr)
		model_list=[]
		start = timeit.default_timer()
		yt,pt=[],[]
		for i,col in enumerate(target):
			# print(col)
			labels = list(y_test[col])
			# print('Building {} model for CODE:{''}'.format(i,col))
			model = alg.fit(X,y[col])
			model_list.append(model)
			yt += labels
			# print(labels)
			# print(list(model.predict_proba(x_test)[:,1]))
			pt += [1 if x > thr else 0 for x in list(model.predict_proba(x_test)[:,1])]
			# print("******")
		frp,trp,thres = roc_curve(yt,pt)
		auc_val =auc(frp,trp)
		plt.plot(frp,trp,label= alg_name+' Threshold ='+str(thr)+' AUC = %.4f'%auc_val)
		F1 = f1_score(yt, pt, average="macro")
		PRESCISION = round(precision_score(yt, pt, average="macro"),2)
		RECALL = round(recall_score(yt, pt, average="macro"),2)
		ACCURACY  = round(accuracy_score(yt, pt),2)
		stop = timeit.default_timer()
		alg_result.append({"ALG_NAME":alg_name,"AUC":auc_val,"RUNTIME":round(stop - start,3),"THRESHOLD":thr,"Average F-Score":F1,"PRESCISION":PRESCISION,"RECALL":RECALL,"ACCURACY":ACCURACY})
	plt.legend(loc='lower right')
	plt.xlabel('True positive rate')
	plt.ylabel('False positive rate')
	plt.title(alg_name+' Reciever Operating Characteristic')
	plt.savefig('../Result/GRAPHS/'+alg_name+'_ROC.png')
	# plt.show()
	return alg_result

#Loading Data 
N = 1000
data = pd.read_csv("../Data/Prcoessed_Data.csv",sep="|",nrows=N)
target = data.CODE
print("Data Loading Done")
# ## Training and Testing Data :
training_data = tfidf_vectorization(data['PROCESSED_TEXT'])
print("Traning Data vectorization Done")
testing_data =  tfidf_vectorization(data['PROCESSED_TEXT_TEST'])
print("Traning Data vectorization Done")
training_label,testing_labels = data[target],data[target]
print("Data Done")

alg = LogisticRegression()
threshold = [0.01,0.19,0.2,0.3,0.4]
alg_name = "Logistic_Regression_TFidf_1000"
pd.DataFrame(classifier(alg,alg_name,threshold,training_data,training_label,testing_data,testing_labels)).to_csv("../Result/CSV/"+alg_name+".csv",index=False)

alg = RandomForestClassifier()
threshold =  [0.02,0.1,0.14,0.22,0.25,0.3]
alg_name = "RandomForestClassifier_TFidf_1000"
pd.DataFrame(classifier(alg,alg_name,threshold,training_data,training_label,testing_data,testing_labels)).to_csv("../Result/CSV/"+alg_name+".csv",index=False)

alg = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=5)  
threshold = [0.07,0.1,0.25,0.5,0.7]
alg_name = "FeedForwardClassifier_TFidf_1000"
pd.DataFrame(classifier(alg,alg_name,threshold,training_data,training_label,testing_data,testing_labels)).to_csv("../Result/CSV/"+alg_name+".csv",index=False)

alg = svm.SVC(probability=True)
threshold = [0.07,0.1,0.25,0.5,0.7]
alg_name = "SVM_TFidf_1000"
pd.DataFrame(classifier(alg,alg_name,threshold,training_data,training_label,testing_data,testing_labels)).to_csv("../Result/CSV/"+alg_name+".csv",index=False)


## Testing Data :
training_data = lsa_feature(data['PROCESSED_TEXT'])
testing_data = lsa_feature(data['PROCESSED_TEXT_TEST'])
training_label,testing_labels = data[target],data[target]

alg = LogisticRegression()
threshold = [0.07,0.1,0.25,0.5,0.7]
alg_name = "Logistic_Regression_LSA_1000"
pd.DataFrame(classifier(alg,alg_name,threshold,training_data,training_label,testing_data,testing_labels)).to_csv("../Result/CSV/"+alg_name+".csv",index=False)

alg = RandomForestClassifier()
threshold = [0.07,0.1,0.25,0.5,0.7]
alg_name = "RandomForestClassifier_LSA_1000"
pd.DataFrame(classifier(alg,alg_name,threshold,training_data,training_label,testing_data,testing_labels)).to_csv("../Result/CSV/"+alg_name+".csv",index=False)

alg = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=5)  
threshold = [0.07,0.1,0.25,0.5,0.7]
alg_name = "FeedForwardClassifier_LSA_1000"
pd.DataFrame(classifier(alg,alg_name,threshold,training_data,training_label,testing_data,testing_labels)).to_csv("../Result/CSV/"+alg_name+".csv",index=False)

alg = svm.SVC(probability=True)
threshold = [0.07,0.1,0.25,0.5,0.7]
alg_name = "SVM_LSA_1000"
pd.DataFrame(classifier(alg,alg_name,threshold,training_data,training_label,testing_data,testing_labels)).to_csv("../Result/CSV/"+alg_name+".csv",index=False)


