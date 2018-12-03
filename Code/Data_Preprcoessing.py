# Data_Preprcoessing
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
import numpy as np
import pandas as pd


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
	print("stopwords removec")
	words = [w for w in words if not w in stop_words]
	stemmed  = words
	# stemmed = [porter.stem(word) for word in words]
	filtered = [w for w in stemmed if  len(w)>1]
	return " ".join(filtered).replace("nan","")


#Loading Data 

data = pd.read_csv("WIKIPEDIA_DATA_FINAL.csv",sep="|")
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
data_test = pd.read_csv("Data.csv",sep="|")
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
data.to_csv("../Data/Prcoessed_Data.csv",index=False,sep="|")
print("Processed File Created")