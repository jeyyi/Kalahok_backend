from nltk import bigrams
from fastapi import FastAPI, HTTPException, Response
from wordcloud import WordCloud
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
import itertools
import collections
import networkx as nx
import io
import matplotlib.pyplot as plt
import tempfile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse
from io import BytesIO
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
    
def get_words():
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
    return (comment_list)


@app.get("/")
def read_root():
    return {"Hello":"World"}

@app.post("/thematic-analysis")
async def add_txt(txt: str):
    db.append(txt)
    return {"Text addedd:":txt}


@app.get("/thematic-analysis")
async def thematic_analysis():
    comment_list = get_words()
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

@app.get("/bigram")
async def bigram_analysis():
    from nltk import bigrams
    comment_list = get_words()
    # Create list of lists containing bigrams in the data
    terms_bigram = [list(bigrams(doc)) for doc in comment_list]

    # Flatten list of bigrams in clean tweets
    bigrams = list(itertools.chain(*terms_bigram))

    # Create counter of words in clean bigrams
    bigram_counts = collections.Counter(bigrams)

    bigram_counts.most_common(30)

    bigram_df = pd.DataFrame(bigram_counts.most_common(30),columns=['bigram', 'count'])
    #bigram_df.to_excel("bigram_output.xlsx", index=False)

    # Create dictionary of bigrams and their counts
    d = bigram_df.set_index('bigram').T.to_dict('records')

    # Create network plot 
    G = nx.Graph()

    # Create connections between nodes
    for k, v in d[0].items():
        G.add_edge(k[0], k[1], weight=(v * 10))

    fig, ax = plt.subplots(figsize=(12, 12))
    fig.suptitle('Word Bigram Co-occurence Network', fontsize=22, y=1.05) 
    fig.tight_layout()    

    ax.axis("off")

    pos = nx.spring_layout(G, k=1)

    # Plot networks
    nx.draw_networkx(G, pos,
                    font_size=11,
                    width=1,
                    edge_color='black',
                    node_color='tab:red',
                    with_labels = False,
                    ax=ax)

    # Create offset labels
    for key, value in pos.items():
        x, y = value[0], value[1]+.06
        ax.text(x, y,
                s=key,
                bbox=dict(facecolor='tab:red', alpha=0.1),
                horizontalalignment='center', fontsize=11)


    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        fig.savefig(tmp_file.name)

    return StreamingResponse(open(tmp_file.name, "rb"), media_type="image/png")


@app.get("/wordcloud")
async def get_wordcloud():
    if not db:
        raise HTTPException(status_code=400, detail="No words provided")
    
    # Join the list of words into a single string
    text = " ".join(db)
    
    # Generate the wordcloud image
    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    
    # Save the image to a byte buffer
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    
    # Return the image as a response with the appropriate media type
    return Response(content=buffer.getvalue(), media_type="image/png")