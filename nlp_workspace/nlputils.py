from konlpy.tag import Mecab
from collections import Counter
import matplotlib
import matplotlib.pyplot as plt

from wordcloud import WordCloud
from matplotlib import font_manager, rc
from konlpy.tag import Okt
from sklearn.feature_extraction.text import CountVectorizer


def get_korean_morphs(words):
    mecab = Mecab(dicpath ="C:\\mecab\\mecab-ko-dic")
    return mecab.morphs(words) #문장을 형태소로 변환하는 유틸리티


def get_clean_word(words, stopwords):
    nouns =[]
    #문장들을 받아서 제외시킬 문자만 빼고 리스트에 추가해 반환한다.
    tagger = Mecab(dicpath = "C:\\mecab\\mecab-ko-dic")
    for post in words:
        for noun in tagger.nouns(post):
            if noun not in stopwords:
                nouns.append(noun)
    return nouns

def get_korean_nouns_list(malist):
    word_dic={}

    for word in malist:
        if word[1]=="Noun":
            if not( word[0] in word_dic):
                word_dic[word[0]]=0
            word_dic[word[0]]+=1
    keys = sorted(word_dic.items(), key = lambda x: x[1], reverse=True)
    return keys

def sort_by_keys(dict):
    return sorted(dict.keys(),reverse=True)

def sort_by_values(dict):
    return sorted(dict, key=dict.get, reverse=True)

def get_most_common_words(words, num):
    counter = Counter(words)
    top_words= dict(counter.most_common(num))
    return top_words

def draw_word_clowd(nouns):
    galexy_wc = WordCloud(background_color="white", font_path='c:/Windows/fonts/malgun.ttf')
    galexy_wc.generate_from_frequencies(nouns)

def draw_bar_grahp(wordInfo):

    font_location = 'c:/Windows/fonts/malgun.ttf'
    font_name = font_manager.FontProperties(fname=font_location).get_name()
    matplotlib.rc('font', family=font_name)

    plt.xlabel('주요 단어')
    plt.ylabel('빈도 수')
    plt.grid(True)

    barcount = 10  # 10개만 그리겠다.

    Sorted_Dict_Values = sorted(wordInfo.values(), reverse=True)
    print(Sorted_Dict_Values)
    print('dd')
    plt.bar(range(barcount), Sorted_Dict_Values[0:barcount], align='center')

    Sorted_Dict_Keys = sorted(wordInfo, key=wordInfo.get, reverse=True)
    print(Sorted_Dict_Keys)
    plt.xticks(range(barcount), list(Sorted_Dict_Keys)[0:barcount], rotation='70')

    plt.show()


def convert_bow(sentence, word_to_index):
    # 문장과 말뭉치를 인덱스로 바꾼 집합을 받는다.

    vector = [0] * (len(word_to_index))

    # 문장을 토큰으로 분리
    tokenizer = Okt()
    tokens = tokenizer.morphs(sentence)

    # 단어의 인덱스 위치에 1 설정
    for token in tokens:
        if token in word_to_index.keys():
            vector[word_to_index[token]] += 1

    return vector

