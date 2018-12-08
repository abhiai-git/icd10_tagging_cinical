#Word2Vector.py
from sklearn.metrics import log_loss,confusion_matrix,classification_report,roc_curve,auc
from scipy import sparse
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import timeit
import pandas as pd
from gensim import corpora, models, similarities
import matplotlib.pyplot as plt

#Loading Data 
N = 100
data = pd.read_csv("../../Data/Prcoessed_Data.csv",sep="|")
target = data.CODE
X_train = data['PROCESSED_TEXT']
X_test = data['PROCESSED_TEXT_TEST']

texts = [[word for word in document.lower().split()] for document in X_train]
from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1
texts = [[token for token in text if frequency[token] > 1] for text in texts]
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=100)
index = similarities.MatrixSimilarity(lsi[corpus])
alg_name="Word2Vector_ALL"


plt.figure(figsize=(8,6))
plt.plot([0,1],[0,1],color='r',linestyle='-.')
threshold = [0.3,0.4,0.498989,0.50123,0.51,0.54,0.6,0.61,0.62,0.8]
alg_result=[]
for ther in threshold:
	i=0
	yt,pt=[],[]
	
	start = timeit.default_timer()
	for doc in X_test:
		vec_bow = dictionary.doc2bow(doc.lower().split())
		vec_lsi = lsi[vec_bow]
		labels = list(data[target[i]])
		print(labels)
		r = [1 if x>ther else 0 for x in list(index[vec_lsi])]
		print(r)
		print(list(index[vec_lsi]))
		print("****")
		yt += labels
		pt += r
		i+=1
	frp,trp,thres = roc_curve(yt,pt)
	auc_val =auc(frp,trp)
	plt.plot(frp,trp,label= alg_name+' Threshold ='+str(ther)+' AUC = %.4f'%auc_val)
	F1 = f1_score(yt, pt, average="macro")
	PRESCISION = round(precision_score(yt, pt, average="macro"),2)
	RECALL = round(recall_score(yt, pt, average="macro"),2)
	ACCURACY  = round(accuracy_score(yt, pt),2)
	stop = timeit.default_timer()
	alg_result.append({"ALG_NAME":alg_name,"AUC":auc_val,"RUNTIME":round(stop - start,3),"THRESHOLD":ther,"Average F-Score":F1,"PRESCISION":PRESCISION,"RECALL":RECALL,"ACCURACY":ACCURACY})


plt.legend(loc='lower right')
plt.xlabel('True positive rate')
plt.ylabel('False positive rate')
plt.title(alg_name+' Reciever Operating Characteristic')
pd.DataFrame(alg_result).to_csv("../../Result/CSV/"+alg_name+".csv",index=False)
plt.savefig('../../Result/GRAPHS/'+alg_name+'_ROC.png')
# plt.show()
