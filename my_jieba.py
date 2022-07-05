import jieba
import jieba.analyse
import jieba.posseg as pseg


MATH_SEG = 'data/math_seg.txt'
IDF = 'data/idf.txt'

jieba.load_userdict(str(MATH_SEG))
# jieba.analyse.set_idf_path(str(IDF))


def cut(sentence, cut_all=False, HMM=True, use_paddle=False):
    return jieba.cut(sentence, cut_all=cut_all, HMM=HMM, use_paddle=use_paddle)


def jieba_tfidf():
    return jieba.analyse.extract_tags


def extract_tags(sentence, topK=20, withWeight=False, allowPOS=(), withFlag=False):
    return jieba.analyse.extract_tags(sentence, topK, withWeight, allowPOS, withFlag)


def get_FREQ(c):
    return jieba.get_FREQ(c)


def pos_seg(question):
    """调用jieba的词性标注"""
    pairs = list(pseg.cut(question))
    return [[word, label] for word, label in pairs]
