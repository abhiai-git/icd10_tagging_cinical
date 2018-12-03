import requests,time
from bs4 import BeautifulSoup
import pandas as pd
import wikipedia

URL ="https://en.wikipedia.org/wiki/ICD-10_Chapter_I:_Certain_infectious_and_parasitic_diseases"
URLS = []
result =[]
URLS.append(URL)

soup = BeautifulSoup(requests.get(URL).content, "html.parser")
for URL_ICD in soup.find('table').findAll('span')[1:]:
	URLS.append("https://en.wikipedia.org"+str(URL_ICD.find('a')['href']))

count = 0
for URL in URLS:
	soup = BeautifulSoup(requests.get(URL).content, "html.parser")
	for LEVEL_1 in soup.findAll('a',{"class":"external text","rel":"nofollow"}):
		print(LEVEL_1)
		try:		
			print(LEVEL_1.text , LEVEL_1.find_next('a').text.encode("utf-8"))
			result.append({"CODE":str(LEVEL_1.text),"NAME":str(LEVEL_1.find_next('a').text.encode("utf-8")),"TEXT":text})
			text = str(wikipedia.page(str(LEVEL_1.find_next('a').text.encode("utf-8"))).content.encode("utf-8")).replace('\n', ' ').replace('\r', '').replace('\t',' ').replace(',',' ')
		except Exception:
			count +=1
			print(count)
			text = ""
			pass	
		

	pd.DataFrame(result).to_csv("WIKIPEDIA_DATA_FINAL.csv",sep="|",index=False)

