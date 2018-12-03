
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


# Data Preprocessing 
# Cleaning of Text and Feature Engineering 
def preprocessing(text):
	# Convert to lower case
	# Remove punctuation from each word
	# Remove remaining tokens that are not alphabetic
	# Filter out stop wordsfrom nltk.corpus import stopwords
	stop_words = stopwords.words('english')
	tokens = word_tokenize(text)
	porter = PorterStemmer()
	tokens = [w.lower() for w in tokens]
	table = str.maketrans('', '', string.punctuation)
	stripped = [w.translate(table) for w in tokens]
	words = [word for word in stripped if word.isalpha()]
	stop_words = set(stopwords.words('english'))
	words = [w for w in words if not w in stop_words]
	stemmed  = words
	# stemmed = [porter.stem(word) for word in words]
	filtered = [w for w in stemmed if  len(w)>1]
	return " ".join(filtered).replace("nan","")

def tfidf_vectorization(input_data):
	vect_word = TfidfVectorizer(max_features=1000, lowercase=True, analyzer='word',stop_words= 'english',ngram_range=(1,3),dtype=np.float32)
	vect_char = TfidfVectorizer(max_features=40000, lowercase=True, analyzer='char',stop_words= 'english',ngram_range=(3,6),dtype=np.float32)
	tr_vect = vect_word.fit_transform(data['PROCESSED_TEXT'])
	tr_vect_char = vect_char.fit_transform(data['PROCESSED_TEXT'])
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
	plt.show()
	return alg_result

#Loading Data 
N = 1000
data = pd.read_csv("WIKIPEDIA_DATA_FINAL.csv",sep="|",nrows=N)
data = data[['NAME','CODE','TEXT']]
data.TEXT = data.TEXT.fillna('').apply(lambda x : x.replace("b'","'").replace("b\"","\""))
data.CODE = data.CODE.apply(lambda x : x.split(".")[0])
df1 = data.groupby(['CODE']).TEXT.unique().reset_index()
df2 = data.groupby(['CODE']).NAME.unique().reset_index()
data = pd.merge(df1,df2,on='CODE')
result=[]
for x in data.TEXT :
	text = ""
	for m in x:
		text +=str(m)
	result.append(str(text))
data.TEXT=result
data['PROCESSED_TEXT']=data.TEXT.apply(lambda x :  preprocessing(str(x)))

#Loading Test Data
data_test = pd.read_csv("Data.csv",sep="|",nrows=N)
data_test = data_test[['CODE','text']]
data_test.text = data_test.text.fillna('').apply(lambda x : x.replace("b'","'").replace("b\"","\""))
data_test.CODE = data_test.CODE.apply(lambda x : x.split(".")[0])
data_test = data_test.groupby(['CODE']).text.unique().reset_index()

result=[]
for x in data_test.text :
	text = ""
	for m in x:
		text +=str(m)
	result.append(str(text))
data_test.text=result
data_test['PROCESSED_TEXT_TEST']=data_test.text.apply(lambda x :  preprocessing(str(x)))

data = pd.merge(data,data_test,on='CODE',how="outer")
data = data.fillna('')
data = data[data.PROCESSED_TEXT != ""]
data = data[data.PROCESSED_TEXT_TEST != ""]

# # Feature Engineering
# # Get one hot encoding of Coding 
# # Join the encoded df
one_hot = pd.get_dummies(data['CODE'])
target = data.CODE
data = data.join(one_hot)


# ## Training and Testing Data :
training_data = tfidf_vectorization(data['PROCESSED_TEXT'])
testing_data =  tfidf_vectorization(data['PROCESSED_TEXT_TEST'])
training_label,testing_labels = data[target],data[target]

alg = LogisticRegression()
threshold = [0.01,0.09,0.1,0.2,0.4]
alg_name = "Logistic_Regression_TFidf"
pd.DataFrame(classifier(alg,alg_name,threshold,training_data,training_label,testing_data,testing_labels)).to_csv("../Result/CSV/"+alg_name+".csv",index=False)

# alg = RandomForestClassifier()
# threshold = [0.1,0.14,0.22,0.25,0.3]
# alg_name = "RandomForestClassifier"
# pd.DataFrame(classifier(alg,alg_name,threshold,training_data,training_label,testing_data,testing_labels)).to_csv("../Result/CSV/"+alg_name+".csv",index=False)

# alg = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=5)  
# threshold = [0.1,0.14,0.2,0.3,0.5]
# alg_name = "FeedForwardClassifier"
# pd.DataFrame(classifier(alg,alg_name,threshold,training_data,training_label,testing_data,testing_labels)).to_csv("../Result/CSV/"+alg_name+".csv",index=False)



# ## Testing Data :
# training_data = lsa_feature(data['PROCESSED_TEXT'])
# testing_data = lsa_feature(data['PROCESSED_TEXT_TEST'])
# training_label,testing_labels = data[target],data[target]

# alg = LogisticRegression()
# threshold = [0.1,0.25,0.28,0.4,0.8]
# alg_name = "Logistic_Regression_LSA"
# pd.DataFrame(classifier(alg,alg_name,threshold,training_data,training_label,testing_data,testing_labels)).to_csv("../Result/CSV/"+alg_name+".csv",index=False)

# alg = RandomForestClassifier()
# threshold = [0.1,0.14,0.22,0.25,0.3]
# alg_name = "RandomForestClassifier"
# pd.DataFrame(classifier(alg,alg_name,threshold,training_data,training_label,testing_data,testing_labels)).to_csv("../Result/CSV/"+alg_name+".csv",index=False)

# alg = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=5)  
# threshold = [0.1,0.14,0.2,0.3,0.5]
# alg_name = "FeedForwardClassifier"
# pd.DataFrame(classifier(alg,alg_name,threshold,training_data,training_label,testing_data,testing_labels)).to_csv("../Result/CSV/"+alg_name+".csv",index=False)


