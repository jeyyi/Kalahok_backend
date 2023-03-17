from fastapi import FastAPI
from typing import List
import operator
import xlsxwriter, io, re
import sys
import nltk
import preprocessor as p
from gsdmm import MovieGroupProcess
from nltk.tokenize import word_tokenize
from pydantic import BaseModel
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import json
import pickle
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db:List[str]=[
]

@app.get("/")
def read_root():
    return {"Hello":"World"}

@app.post("/thematic-analysis")
async def add_txt(txt: str):
    db.append(txt)
    return {"Text addedd:":txt}


@app.get("/thematic-analysis")
async def thematic_analysis():
    nltk.download('stopwords')
    nltk.download('punkt')
    stop_words = set(stopwords.words('english'))
    stop_words_tl = []
    crimefile = open('tagalog_stopwords.txt', 'r')
    tl = crimefile.readlines()
    for textl in tl:
        textl = re.sub('\n','',textl)
        stop_words_tl.append(textl)
    #removing empty strings
    while("" in stop_words_tl) :
        stop_words_tl.remove("")
    stop_words_tl.extend(list(stopwords.words('english')))
    comment_list = []
    for txt in db:
        t = txt.replace('\n', '')
        t = t.strip().lower()
        t = p.clean(t)
        t = word_tokenize(t)
        t = " ".join(w for w in t if w not in stop_words_tl)
        t= word_tokenize(t)
        t = re.sub('[^a-zA-Z0-9 ]', '', str(t))
        t = re.sub('\b\w{1,3}\b', '', str(t))
        comment_list.append(t)
    comment_list = list(filter(None, comment_list))
    comment_list = [text.split() for text in comment_list]
    K = 6
    alpha_val = 0.05
    beta_val = 0.1
    iter_val = 100
    mgp = MovieGroupProcess(K=K, alpha=alpha_val , beta=beta_val , n_iters=iter_val)

    modelName = 'comment1'+ str(alpha_val) + '_' + str(beta_val) + '_' + str(iter_val) +'.model'

    vocab = set(x for doc in comment_list for x in doc)
    n_terms = len(vocab)
    n_docs = len(comment_list)
    # Fit the model on the data given the chosen seeds
    y = mgp.fit(comment_list, n_terms)
    # Save model, input model name

    with open(modelName, "wb") as f:
        pickle.dump(mgp, f)
        f.close()
    # Load the model used in the post
    filehandler = open(modelName, 'rb')
    mgp = pickle.load(filehandler)

    doc_count = np.array(mgp.cluster_doc_count)
    # Topics sorted by document inside
    top_index = doc_count.argsort()[-10:][::-1]
    # Show the top n words by cluster, it helps to make the topic_dict below
    clusterNum = 0
    data = []
    for i in mgp.cluster_word_distribution:
        i = sorted(i.items(), key=lambda item: item[1], reverse=True)
        counter = K
        for key in i:
            if counter > 0:
                content = []
                content.append(clusterNum)
                content.append(key[0])
                content.append(key[1])
                data.append(content)
            counter-= 1
        clusterNum += 1
    #writing data to Pandas dataframe
    df = pd.DataFrame(data, columns = ['cluster_num', 'topic_word', 'occurrence']) 
    #df.to_excel("GSDMM_output.xlsx", index=False)
    df.sort_values(['occurrence'])
    return json.loads(df.to_json(orient='records'))