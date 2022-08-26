import pandas as pd
from konlpy.tag import Okt
from gensim.models import Word2Vec

def word2vec_similarR(title, category):
    #load model
    model = Word2Vec.load('./Word2VecTrainedModel.model')
    doc2 = title

    okt = Okt()
    tokenized_doc2 = okt.pos(doc2)
    tokenized_nouns2 = ' '.join([word[0] for word in tokenized_doc2 if word[1] == 'Noun'])
    tokenized_nouns2 = tokenized_nouns2.split(' ')

    #유사 단어 리스트 가공 > 튜플 내의 단어만 추출.
    wordlist = []
    for word in tokenized_nouns2:
        similarWord = model.wv.most_similar(word, topn=5)
        # print(word, similarWord)
        list1 = [x[0] for x in similarWord]
        wordlist += list1

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

        tokenized_doc = okt.pos(doc)
        tokenized_nouns = ' '.join([word[0] for word in tokenized_doc if word[1] == 'Noun'])
        tokenized_nouns = tokenized_nouns.split(' ')
        df['Nouns'][i] = tokenized_nouns

    #정확도 확인
    for i in range(len(df)):
        hit = 0
        for word in wordlist:
            text = df['Nouns'][i]
            if word in text:
                hit += 1
            else:
                continue
        df['Hits'][i] = hit

    df_sorted = df.sort_values(by='Hits', ascending=False)
    df_sorted = df_sorted[["category", "title"]][:5]
    df2json = df_sorted.to_json(orient="records", force_ascii=False)
    return df2json
