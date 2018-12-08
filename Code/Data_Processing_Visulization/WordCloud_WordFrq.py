from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 
import pandas as pd 

data = pd.read_csv("../../Data/Prcoessed_Data.csv",sep="|")
target = data.CODE
X_train = data['PROCESSED_TEXT']
X_test = data['PROCESSED_TEXT_TEST']
comment_words = ' '
stopwords = set(STOPWORDS) 
for val in X_train:       
    # typecaste each val to string 
    val = str(val)   
    # split the value 
    tokens = val.split()      
    # Converts each token into lowercase 
    for i in range(len(tokens)): 
        tokens[i] = tokens[i].lower()     
    for words in tokens: 
    	comment_words += words + ' '  
wordcloud = WordCloud(width = 800, height = 800, background_color ='white', stopwords = stopwords, min_font_size = 10).generate(comment_words) 
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.savefig('../Result/GRAPHS/WordCloud_Wikipedia.png')
plt.show() 
for val in X_test:       
    # typecaste each val to string 
    val = str(val)   
    # split the value 
    tokens = val.split()      
    # Converts each token into lowercase 
    for i in range(len(tokens)): 
        tokens[i] = tokens[i].lower()     
    for words in tokens: 
        comment_words += words + ' '  
wordcloud = WordCloud(width = 800, height = 800, background_color ='white', stopwords = stopwords, min_font_size = 10).generate(comment_words) 
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.savefig('../Result/GRAPHS/WordCloud_ICD10_COM.png')
plt.show() 