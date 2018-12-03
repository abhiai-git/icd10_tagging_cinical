import requests,time
from bs4 import BeautifulSoup
import pandas as pd
import wikipedia

data = []
URL ="https://www.icd10data.com/ICD10CM/Codes"
MAIN_URL = "https://www.icd10data.com"
soup = BeautifulSoup(requests.get(URL).content, "html.parser")
for x in soup.find('div',{'class':'body-content'}).find('ul').findAll('li'):
	URL_GROUP = MAIN_URL+str(x.find('a')['href'])
	soup_level_1 = BeautifulSoup(requests.get(URL_GROUP).content, "html.parser")
	for y in soup_level_1.find('ul',{'class':'i51'}).findAll('li'):
		URL_GROUP_level = MAIN_URL+y.find('a')['href']
		soup_level_2 = BeautifulSoup(requests.get(URL_GROUP_level).content, "html.parser")
		for z in soup_level_2.find('ul',{'class':'i51'}).findAll('li'):
			URL_GROUP_level_2 = MAIN_URL+z.find('a')['href']
			soup_level_3 = BeautifulSoup(requests.get(URL_GROUP_level_2).content, "html.parser")
			text = ""
			for r in soup_level_3.find('ul',{'class':'codeHierarchy'}).findAll('a'):
				PAGE_URL = MAIN_URL+r['href']
				PAGE_SOUP = BeautifulSoup(requests.get(PAGE_URL).content, "html.parser")
				for ci in PAGE_SOUP.findAll('span'):
					if ci.text == 'Clinical Information':
						text += ci.find_next('ul').text.replace("\n","")
			print(soup_level_3.find('ul',{'class':'codeHierarchy'}).findAll('a')[0].text,text)
			data.append({"CODE":soup_level_3.find('ul',{'class':'codeHierarchy'}).findAll('a')[0].text,"text":text})
			print("**********")

pd.DataFrame(data).to_csv("Data.csv",sep="|",index=False)