import pandas as pd
import numpy as np
from konlpy.tag import Okt
import gensim
from gensim.models import Word2Vec, keyedvectors

def okt_exactR(title, category):
    #load data - 추후 json to dataFrame으로 변경해야함.
    df = pd.read_csv("./modelTrainData.csv", engine='python')
    df['Nouns'] = 0
    df['Hits'] = 0

    #category labeling
    label = {'language': 0, 'sports': 1, 'programming': 2, 'study': 3, 'art': 4, 'writing': 5, 'design': 6, 'music': 7, 'cooking': 8, 'etc': 9}
    df['category'] = df.category.map(label)

    #keyword extraction
    for i in range(len(df)):
        doc = df['title'][i]
        okt = Okt()

        tokenized_doc = okt.pos(doc)
        tokenized_nouns = ' '.join([word[0] for word in tokenized_doc if word[1] == 'Noun'])
        tokenized_nouns = tokenized_nouns.split(' ')
        df['Nouns'][i] = tokenized_nouns

    #현재 클래스 data
    doc2 = title

    tokenized_doc2 = okt.pos(doc2)
    tokenized_nouns2 = ' '.join([word[0] for word in tokenized_doc2 if word[1] == 'Noun'])
    tokenized_nouns2 = tokenized_nouns2.split(' ')

    #현재 클래스 title과 DB title 데이터 비교
    for i in range(len(df)):
        hit = 0
        for word in tokenized_nouns2:
            text = df['Nouns'][i]
            if word in text:
                hit += 1
            else: continue
        df['Hits'][i] = hit

    df_sorted = df.sort_values(by='Hits', ascending=False)
    df_sorted = df_sorted[["category", "title"]][:5]
    df2json = df_sorted.to_json(orient="records", force_ascii=False)
    return df2json

# okt_exactR("영어 과외 모집", 0)