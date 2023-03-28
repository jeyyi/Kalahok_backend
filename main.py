import collections
import itertools
import json
import pickle
import re
import tempfile
from collections import Counter
from io import BytesIO
from typing import Dict, List

import matplotlib.pyplot as plt
import networkx as nx
import nltk
import numpy as np
import pandas as pd
import preprocessor as p
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from gsdmm import MovieGroupProcess
from nltk import bigrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pydantic import BaseModel
from wordcloud import WordCloud
from pymongo import MongoClient

# Connect to your MongoDB cluster
client = MongoClient("mongodb://localhost:27017")
db = client["Sales"]
collection = db["employees"]


nltk.download('stopwords')
nltk.download('punkt')
app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_text_list():
    # Query for all documents in the collection and project only the "text" field
    cursor = collection.find({}, {"What other issues affecting the youth do you think should also be prioritized?": 1})

    # Extract the "text" field from each document and add it to a list
    text_list = [doc["What other issues affecting the youth do you think should also be prioritized?"] for doc in cursor]

    # Return the list of text strings
    return text_list

db:List[str] = get_text_list()
    
def get_words():
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
    return {"Hello":"Mundo"}

@app.post("/thematic-analysis")
async def add_txt(txt: str):
    db.append(txt)
    return {"Text addedd:":txt}

@app.get("/get-all-text")
async def get_texts():
    return get_text_list()

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
    sample =  json.loads(df.to_json(orient='records'))

    clusters = {}
    for item in sample:
        if item['cluster_num'] not in clusters:
            clusters[item['cluster_num']] = {'cluster_num': item['cluster_num'], 'topic_words': item['topic_word']}
        else:
            clusters[item['cluster_num']]['topic_words'] += f",{item['topic_word']}"
    return list(clusters.values())

@app.get("/bigram")
async def bigram_analysis():
    from nltk import bigrams
    """
    This function returns a temporary streaming response png file of a bigram based on all the texts from the database.
    """
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

@app.get('/frequent')
#returns top 10 words
async def get_frequent() -> List:
    # Combine all strings into one long string
    result = [word for sublist in get_words() for word in sublist]
    combined_string = ' '.join(result)

    # Split the long string into individual words
    words = combined_string.split()

    # Count the occurrences of each word
    word_counts = Counter(words)

    # Get the top 10 most frequent words
    top_10 = word_counts.most_common(10)

    # Create a dictionary to store the results
    results = {}
    for word, count in top_10:
        results[word] = count

    # Return the dictionary as JSON
    lst=[]
    for key in results:
        temp = {"word":key, "number":results[key]}
        lst.append(temp)
    return(lst)
