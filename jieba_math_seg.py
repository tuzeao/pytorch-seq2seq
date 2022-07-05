from my_jieba import pos_seg
import re
from typing import Tuple, List


def load_math_keys():
    """加载数学关键词"""
    fin = open('data/math_keywords_set.txt', encoding='utf-8')
    math_keys = fin.readlines()
    fin.close()

    math_keys = set([line.strip() for line in math_keys])
    return math_keys


math_keys = load_math_keys()


def load_units():
    """加载单位"""
    fin = open('data/units.txt', encoding='utf-8')
    units = fin.readlines()
    fin.close()

    units = set([line.strip() for line in units])
    return units


units = load_units()


class MathTag:
    MATH = 'MATH'
    PER = 'PER'
    UNIT = 'UNIT'
    NUM = 'NUM'
    AMOUNT = 'AMOUNT'
    SYMBOL = 'SYMBOL'
    TIME = 'TIME'
    ENG = 'ENG'
    ALEXP = 'ALEXP'
    ORD = 'ORD'
    BLANK = 'BLANK'
    OTHER = 'OTHER'


labels = {MathTag.MATH, MathTag.PER, MathTag.UNIT, MathTag.NUM, MathTag.AMOUNT, MathTag.SYMBOL, MathTag.TIME,
          MathTag.ENG, MathTag.ALEXP, MathTag.ORD, MathTag.BLANK}

skip_units = {'年级', '班'}

symbols = {'+', '-', '×', '×', '÷', '÷', '=', '＝', ':', '%', '∠'}

people = {'爷爷', '奶奶', '爸爸', '妈妈', '哥哥', '姐姐', '弟弟', '妹妹', '叔叔', '阿姨', '老师', '师傅', '大伯', '大爷'}

# math_units是更严格的单位，即只要出现就可以判定为单位
math_units = {'平方毫米', '平方厘米', '平方分米', '平方米', '公顷', '平方千米', '亩', '立方毫米', '立方厘米', '立方分米', '立方米',
              '立方千米', '毫升', '升', '毫米', '厘米', '分米', '千米', '米', '公里', '千克', '克', '斤', '公斤', '吨', 'km',
              'mm', 'cm', 'dm', 'hm', 'm', 't', 'kg', 'g', 'ml', 'mL', 'l', 'L', '°', '°C', '℃', '元', '分钟', '小时',
              '秒', '毫秒', '千瓦', '千瓦时'}


def is_symbol(text):
    """判断是否是数学符号"""
    if text in symbols:
        return True
    angle_pt = re.compile(r'[角∠][A-Za-z\d]{1,3}')
    if angle_pt.match(text):
        return True
    return False


def pure_number(text):
    """判断是否是纯数字"""
    num_pt = re.compile(r'\d+(.\d+)?$')
    zh_num_pt = re.compile(r'[零一二两三四五六七八九十百千万亿]+(点[零一二两三四五六七八九十百千万亿]+)?$')
    if num_pt.match(text) or zh_num_pt.match(text):
        return True
    return False


def pure_eng(text):
    """判断是否是纯英文"""
    en_pt = re.compile(r'[A-Za-z]+$')
    if en_pt.match(text):
        return True
    return False


def is_zh_number(text):
    """判断是否是中文数字"""
    zh_num_pt = re.compile(r'[零一二两三四五六七八九十百千万亿]+(点[零一二两三四五六七八九十百千万亿]+)?$')
    if zh_num_pt.match(text):
        return True
    return False


def is_en_char(text):
    """判断是否是单个英文字母"""
    en_pt = re.compile(r'[A-Za-z]')
    if len(text) == 1 and en_pt.match(text):
        return True
    return False


def is_amount(word):
    """判断是否是数量"""
    if word in units:
        return False
    for unit in units:
        if word.endswith(unit):
            prefix = word[:len(word) - len(unit)]
            if prefix == '一':
                return False
            if pure_number(prefix):
                return True
    return False


def is_percent(word):
    """判断是否是百分数"""
    perc_pt = re.compile(r'[\d\.]+%$')
    zh_perc_pt = re.compile(r'百分之[零一二两三四五六七八九十百千万亿]+$')
    if perc_pt.match(word) or zh_perc_pt.match(word):
        return True
    return False


def is_ord(word):
    """判断是否是序数词"""
    if word.startswith('第'):
        prefix = word[1:]
        if pure_number(prefix):
            return True
    return False


def is_sup_power(text):
    """判断是不是<sup>2</sup>这样的形式"""
    sup_power = re.compile(r"<sup>[\d°]</sup>$")
    if sup_power.match(text):
        return True
    return False


def is_power(text):
    """判断是不是幂的形式"""
    if text in ('²', '³'):
        return True
    return is_sup_power(text)


def is_underline_blank(text):
    """判断是不是下划线"""
    un_pt = re.compile(r'_{2,}$')
    if un_pt.match(text):
        return True
    return False


def is_time(text):
    """判断字符串是否是xx:xx形式的时间"""
    time_pt = re.compile(r'\d\d:\d\d')
    if time_pt.match(text):
        return True
    return False


def search_exp(question):
    """匹配题目中的算式"""
    propor_pt = re.compile(r"[A-Za-z\d\.]+[:：][A-Za-z\d\.]+(?:=[A-Za-z\d\.]+[:：][A-Za-z\d\.]+)*")  # 匹配比例式，如1:2=2:4
    angle_pt = re.compile(r"∠\d=\d+°")  # 如：∠1=40°

    exp_pt1 = re.compile(r"[\(（)]?[A-Za-z\d\.]+[\)）]?(?:[\+\-×÷][\(（]?[A-Za-z\d\.]+[\)）]?)*=[\(（]?[A-Za-z\d\.]+[\)）]?([\+\-×÷][\(（]?[A-Za-z\d\.]+[\)）]?)*")
    exp_pt2 = re.compile(r"(?:[\(（]?[A-Za-z\d\.]+[\)）]?[\+\-×÷])+(?:[\(（]?[A-Za-z\d\.]+[\)）]?)+=?")
    en_phrase_pt = re.compile(r"[A-Za-z]{2,}\-[A-Za-z]{2,}")

    pts = [propor_pt, angle_pt, exp_pt1, exp_pt2]
    objm = None
    for pt in pts:
        m = pt.search(question)
        if not m:
            continue
        if not objm or m.start() < objm.start():
            objm = m
    if objm:
        exp = objm.group()
        if not en_phrase_pt.match(exp) and not is_time(exp):
            return exp, objm.start(), objm.end(), MathTag.ALEXP
    return None, None, None, None


def search_sup(quest):
    """通过正则提取<sup>2</sup>样式的字符串"""
    sup_pt = re.compile(r'<sup>[\d°]</sup>')
    m = sup_pt.search(quest)
    if m:
        return m.group(), m.start(), m.end(), 'x'
    return None, None, None, None


def search_zh_percent(quest):
    """通过正则提取中文的百分数"""
    per_pt = re.compile(r'百分之[零一二三四五六七八九十点]+')
    m = per_pt.search(quest)
    if m:
        return m.group(), m.start(), m.end(), MathTag.NUM
    return None, None, None, None


def search_time(quest):
    """通过正则提取时间"""
    time_pt = re.compile(r'\d\d:\d\d')
    m = time_pt.search(quest)
    if m:
        return m.group(), m.start(), m.end(), MathTag.TIME
    return None, None, None, None


def search_frac(quest):
    """通过正则提取分数"""
    frac_pt = re.compile(r'\\?frac\{[A-Za-z\d]+\}\{[A-Za-z\d]+\}')
    es_frac_pt = re.compile(r'\\?攟\{[A-Za-z\d]+\}\s?\{[A-Za-z\d]+\}')

    pts = [frac_pt, es_frac_pt]
    objm = None
    for pt in pts:
        m = pt.search(quest)
        if not m:
            continue
        if not objm or m.start() < objm.start():
            objm = m
    if objm:
        return objm.group(), objm.start(), objm.end(), MathTag.NUM
    return None, None, None, None


def search_zh_frac(quest):
    """通过正则提取中文的分数"""
    zh_frac_pt = re.compile(r'[零一二三四五六七八九十]+分之[零一二三四五六七八九十]+')
    m = zh_frac_pt.search(quest)
    if m:
        return m.group(), m.start(), m.end(), MathTag.NUM
    return None, None, None, None


def search_zh_num(quest):
    """通过正则提取中文数字"""
    zh_num_pt = re.compile(r'[零一二三四五六七八九十百千万亿点]{2,}')
    m = zh_num_pt.search(quest)
    if m:
        return m.group(), m.start(), m.end(), 'm'
    return None, None, None, None


def search_text_power(quest):
    """提取形如'{\text{m}^{2}}'这样的符号，表示平方米"""
    text_power_pt = re.compile(r'(?:\\text\{([A-Za-z])\})?\{\{\\text\{([A-Za-z])\}\}\^\{\d\}\}')
    m = text_power_pt.search(quest)
    if m:
        if m.group(1):
            symb = m.group(1) + m.group(2)
        else:
            symb = m.group(2)
        if symb in units:
            tag = MathTag.UNIT
        else:
            tag = MathTag.OTHER
        return m.group(), m.start(), m.end(), tag
    return None, None, None, None


def search_text(quest):
    """提取形如'\text{c}'这样的符号"""
    text_pt = re.compile(r'\\text\{([A-Za-z\d=]+)\}')
    m = text_pt.search(quest)
    if m:
        mid = m.group(1)
        if mid in units:
            tag = 'q'
        elif pure_number(mid):
            tag = MathTag.NUM
        else:
            tag = MathTag.ENG
        return m.group(), m.start(), m.end(), tag
    return None, None, None, None


def search_degree(quest):
    """提取形如'{}^∘\\text{C}'这样的符号"""
    degree_pt = re.compile(r'\{\}\^∘\\text\{C\}')
    m = degree_pt.search(quest)
    if m:
        return m.group(), m.start(), m.end(), MathTag.UNIT
    return None, None, None, None


def get_re(question):
    """获取所有根据正则表达式得到的词性标注结果"""
    question, records = normalization(question)  # 文本归一化，方便正则匹配
    funcs = [search_exp, search_sup, search_zh_percent, search_time, search_frac, search_zh_frac, search_zh_num,
             search_text_power, search_text, search_degree]
    outs = []
    for func in funcs:
        prefix = 0
        text, st, ed, tag = func(question)
        while text is not None:
            outs.append((text, st + prefix, ed + prefix, tag))
            prefix += ed
            text, st, ed, tag = func(question[prefix:])
    outs.sort(key=lambda x: x[1])
    # 去掉正则匹配的结果中有重叠区域的
    res = []
    for idx, item in enumerate(outs):
        if idx == 0:
            res.append(item)
            continue
        prev_item = res[-1]
        if item[1] < prev_item[2]:  # 有重叠区域，理论上是正则匹配有问题，但是不能影响后面执行，需要去掉一个
            if (item[2] - item[1]) <= (prev_item[2] - prev_item[1]):
                continue
            else:
                res[-1] = item
        else:
            res.append(item)
    # 未匹配区域与匹配区域组合，然后根据归一化的记录，将匹配文本复原
    words = []
    for item in res:
        rst, red = item[1], item[2]
        if not words:
            st, ed = 0, rst
        else:
            st, ed = words[-1][2], rst
        if st != ed:
            words.append((question[st:ed], st, ed, 'OTHER'))
        words.append(item)
    words = [(item[0], item[3]) for item in words]
    words = recover(words, records)
    # 取出复原后的匹配结果
    res = []
    locs = []
    for word, flag in words:
        if not locs:
            locs.append([0, len(word)])
        else:
            prev_st, prev_ed = locs[-1]
            cur_st = prev_ed
            cur_ed = cur_st + len(word)
            locs.append([cur_st, cur_ed])
        if flag != 'OTHER':
            res.append((word, locs[-1][0], locs[-1][1], flag))
    return res


def combine_pos_with_re(question) -> List[Tuple[str, str]]:
    """结合jieba词性标注和正则匹配的结果"""
    pos_res = pos_seg(question)
    re_items = get_re(question)
    if not re_items:
        return pos_res
    pos_st, pos_ed = 0, 0
    exp_id = 0
    outs = []
    pidx = 0
    while pidx < len(pos_res):
        word, flag = pos_res[pidx]
        if exp_id >= len(re_items):
            outs.append((word, flag))
            pidx += 1
            continue
        pos_ed += len(word)
        exp, exp_st, exp_ed, re_flag = re_items[exp_id]
        if pos_ed <= exp_st:
            outs.append((word, flag))
        elif pos_st <= exp_st < pos_ed <= exp_ed:
            pos_pre = word[: exp_st - pos_st]
            if pos_pre:
                outs.append((pos_pre, flag))  # 前缀沿用flag
            outs.append((exp, re_flag))
            if pos_ed == exp_ed:
                exp_id += 1
        elif pos_st <= exp_st and pos_ed >= exp_ed:
            pos_pre = word[: exp_st - pos_st]
            pos_suf = '' if pos_ed == exp_ed else word[exp_ed - pos_ed:]
            if pos_pre:
                outs.append((pos_pre, flag))
            outs.append((exp, re_flag))
            if pos_suf:
                outs.append((pos_suf, flag))
            exp_id += 1
        elif pos_st > exp_st and pos_ed < exp_ed:
            pass
        elif exp_st < pos_st < exp_ed <= pos_ed:
            if pos_ed == exp_ed:
                pos_suf = ''
            else:
                pos_suf = word[exp_ed - pos_ed:]
            if pos_suf:  # 后缀部分不在当前re结果范围内
                pos_res = pos_res[:pidx + 1] + [(pos_suf, flag)] + pos_res[pidx + 1:]
                pos_st -= len(pos_suf)
                pos_ed -= len(pos_suf)
            exp_id += 1
        else:
            outs.append((word, flag))
        pidx += 1
        pos_st += len(word)
    return outs


def process_m(words, idx, tags) -> Tuple[str, str, int]:
    """
    当前位置，flag==m，可能的情况有：1）纯数字，后面可能会跟单位；2）百分数；3）英文，如‘a个排球’，此时a的flag==m;
    """
    word, flag = words[idx]
    if word in symbols:
        return process_symbol(words, idx)
    elif is_percent(word):
        return word, MathTag.NUM, idx
    # elif is_ord(word):
    #     return word, 'ORD', idx
    elif pure_number(word):  # or pure_eng(word):  # 纯数字
        return process_number(words, idx)
    elif pure_eng(word):  # 纯英文
        return process_eng(words, idx, tags)
    else:
        if word in math_keys:
            return word, MathTag.MATH, idx
        if word in units:
            return process_unit(words, idx, tags)
        if is_amount(word):
            return word, MathTag.AMOUNT, idx
        return word, flag, idx


def process_number(words, idx) -> Tuple[str, str, int]:
    """当前位置是纯数字"""
    word, flag = words[idx]
    if idx == len(words) - 1:  # 已到结尾
        if is_zh_number(word):
            return word, MathTag.OTHER, idx
        return word, MathTag.NUM, idx
    else:
        next_word, next_flag = words[idx + 1]
        if next_word == '：':
            return word, MathTag.OTHER, idx
        if next_word == '%':
            return word + next_word, MathTag.NUM, idx + 1
        if next_word in ('千', '万', '十万', '百万', '千万', '亿'):
            return word + next_word, MathTag.NUM, idx + 1
        if next_word == '年' and len(word) > 3:
            return word + next_word, MathTag.TIME, idx + 1
        if next_word in ('月', '月份', '时', '日', '点', '点钟'):
            return word + next_word, 'TIME', idx + 1
        if next_word in ('个'):
            if word != '一':  # '一个'算是比较一般性的词语
                return word + next_word, MathTag.AMOUNT, idx + 1
        if next_word in skip_units:
            return word, MathTag.OTHER, idx
        if next_flag == 'q' or next_word in units:  # 下一个是单位
            if idx < len(words) - 3:
                n2_word, n2_flag = words[idx + 2]
                n3_word, n3_flag = words[idx + 3]
                if n2_word in ('·', '/', 'cdot') and n3_word in units:
                    return word + next_word + n2_word + n3_word, MathTag.AMOUNT, idx + 3
            return word + next_word, MathTag.AMOUNT, idx + 1
        if next_word in ('）', ')', '“'):
            if idx > 0:
                prev_word, prev_flag = words[idx - 1]
                if prev_word in ('（', '(', '”'):
                    return word, MathTag.OTHER, idx
        if pure_eng(next_word):
            return word + next_word, MathTag.ALEXP, idx + 1
        if is_zh_number(word):
            if idx > 0 and words[idx - 1][0] == '个':
                return word, MathTag.NUM, idx
            return word, MathTag.OTHER, idx
        if next_word == '\\':
            if idx < len(words) - 2:
                nn_word, nn_flag = words[idx + 2]
                if nn_word == '%':
                    return word + next_word + nn_word, MathTag.NUM, idx + 2
        return word, MathTag.NUM, idx


def process_nr(words, idx):
    """处理人物"""
    word, flag = words[idx]
    if word in ('多少钱'):
        return word, MathTag.OTHER, idx
    if idx == len(words) - 1:
        if len(word) == 1:
            return word, MathTag.OTHER, idx
        return word, MathTag.PER, idx
    else:
        next_word, next_flag = words[idx + 1]
        if next_flag == 'nr' or next_word in people:  # 连续的两个人名拼在一起
            return word + next_word, MathTag.PER, idx + 1
        elif len(word) == 1 and next_flag == 'n':
            return word + next_word, MathTag.PER, idx + 1
        else:
            if len(word) == 1:
                return word, MathTag.OTHER, idx
            return word, MathTag.PER, idx


def process_eng(words, idx, tags):
    """处理英文"""
    word, flag = words[idx]
    if idx < len(words) - 1:
        next_word, next_flag = words[idx + 1]
        if next_flag == 'q' or next_word in units:
            return word + next_word, MathTag.AMOUNT, idx + 1
    if word in units:
        if word == 'm':
            if tags:
                prev_word, prev_flag = tags[-1]
                if prev_flag == MathTag.BLANK or prev_word in ('多少', '几'):
                    return word, MathTag.UNIT, idx
            return word, MathTag.ENG, idx
        return word, MathTag.UNIT, idx
    return word, MathTag.ENG, idx


def process_x(words, idx):
    """处理x标签"""
    word, flag = words[idx]
    if word in ('(', '（'):
        n_word = word
        nidx = idx
        while nidx < len(words) - 1:
            next_word, next_flag = words[nidx + 1]
            if next_word == ' ' or next_word == '\xa0' or next_word == '※':
                nidx += 1
                n_word += next_word
                continue
            if next_word in (')', '）'):
                return n_word + next_word, MathTag.BLANK, nidx + 1
            break
    if word == '\xa0':
        i = idx + 1
        while i < len(words):
            if words[i][0] == '\xa0':
                i += 1
            else:
                break
        if i - idx > 2:
            return '\xa0' * (i - idx), MathTag.BLANK, i - 1
    if is_underline_blank(word):
        return word, MathTag.BLANK, idx
    if word in ('□'):
        return word, MathTag.BLANK, idx
    if word in symbols:
        return process_symbol(words, idx)
    return word, flag, idx


def process_math_keys(words, idx):
    """处理数学词汇"""
    word, flag = words[idx]
    if word == '和':
        if idx > 0:
            prev_word, prev_flag = words[idx - 1]
            if prev_word in ('的', '之'):
                return word, MathTag.MATH, idx
        if idx < len(words) - 1:
            next_word, next_flag = words[idx + 1]
            if next_word == '是':
                return word, MathTag.MATH, idx
        return word, MathTag.OTHER, idx
    return word, MathTag.MATH, idx


def process_ord(words, idx):
    """处理序数词"""
    word, flag = words[idx]
    if len(word) == 1:
        if idx < len(words) - 1:
            next_word, next_flag = words[idx + 1]
            if is_ord(word + next_word):
                return word + next_word, MathTag.ORD, idx + 1
    elif is_ord(word) or is_ord(word[:-1]):
        return word, MathTag.ORD, idx
    return word, flag, idx


def process_unit(words, idx, tags):
    """处理单位"""
    word, flag = words[idx]
    if idx < len(words) - 1:
        next_word, next_flag = words[idx + 1]
        if word + next_word in math_units:
            return word + next_word, MathTag.UNIT, idx + 1
        if idx < len(words) - 2:
            nn_word, nn_flag = words[idx + 2]
            if next_word in ('·', '/', 'cdot') and nn_word in units:
                return word + next_word + nn_word, MathTag.UNIT, idx + 2
    if word in math_units:
        return word, MathTag.UNIT, idx
    if tags:
        prev_word, prev_flag = tags[-1]
        if prev_word in ('多少', '几') or prev_flag == MathTag.BLANK:
            return word, MathTag.UNIT, idx
    if flag == MathTag.UNIT:
        return word, MathTag.UNIT, idx
    return word, MathTag.OTHER, idx


def process_symbol(words, idx):
    """处理数学符号"""
    word, flag = words[idx]
    if word == '∠':
        if idx < len(words) - 1:
            next_word, next_flag = words[idx + 1]
            if pure_number(next_word):
                return word + next_word, MathTag.MATH, idx + 1
    if word == '×':
        if 0 < idx < len(words) - 1:
            prev_word, prev_flag = words[idx - 1]
            next_word, next_flag = words[idx + 1]
            if prev_word == '“' and next_word == '”':
                return word, MathTag.OTHER, idx
    if word == '-':
        if idx < len(words) - 1:
            next_word, next_flag = words[idx + 1]
            if pure_number(next_word):
                return word + next_word, MathTag.NUM, idx + 1
    if word == '°':
        if idx < len(words) - 1:
            next_word, next_flag = words[idx + 1]
            if pure_number(next_word):
                return word + next_word, MathTag.UNIT, idx + 1
    return word, MathTag.SYMBOL, idx


def post_process(words, records=None):
    """后处理，其他标签变成 'OTHER'，相邻的time标签合并"""
    outs = []
    for word, flag in words:
        if flag not in labels:
            outs.append((word, MathTag.OTHER))
            continue
        if outs:
            prev_word, prev_flag = outs[-1]
            if flag == MathTag.TIME and prev_flag == MathTag.TIME:
                n_word = prev_word + word
                outs[-1] = (n_word, MathTag.TIME)
                continue
            if prev_flag == MathTag.TIME and word.endswith('分'):
                n_word = prev_word + word
                outs[-1] = (n_word, MathTag.TIME)
                continue
            if (flag == MathTag.SYMBOL or flag == MathTag.ALEXP) and prev_flag == MathTag.ALEXP:
                n_word = prev_word + word
                outs[-1] = (n_word, MathTag.ALEXP)
                continue
            if flag == MathTag.UNIT and prev_flag == MathTag.NUM:
                n_word = prev_word + word
                outs[-1] = (n_word, MathTag.AMOUNT)
                continue
        outs.append((word, flag))
    return recover(outs, records)


def recover(words, records):
    """将归一化时进行的操作还原回去"""

    def _update_locs(words):
        locs = []
        for word, flag in words:
            if not locs:
                locs.append([0, len(word)])
            else:
                prev_st, prev_ed = locs[-1]
                cur_st = prev_ed
                cur_ed = cur_st + len(word)
                locs.append([cur_st, cur_ed])
        return locs

    locs = _update_locs(words)
    ridx = 0
    widx = 0
    while ridx < len(records) and widx < len(words):
        rloc, rop = records[ridx]
        word, flag = words[widx]
        st, ed = locs[widx]
        if rloc >= ed:
            widx += 1
        else:
            cidx = rloc - st
            if rop == ' ':
                if rloc == st:
                    words = words[:widx] + [(rop, MathTag.OTHER)] + words[widx:]
                else:
                    n_word = word[:cidx] + ' ' + word[cidx:]
                    words[widx] = (n_word, flag)
                locs = _update_locs(words)
            else:
                n_word = word[:cidx] + rop + word[cidx + 1:]
                words[widx] = (n_word, flag)
            ridx += 1
    return words


def normalization(question):
    """归一化"""
    replaces = {'＝': '=', '∶': ':', '－': '-', ' ': ''}
    tokens = []
    records = []
    for idx, c in enumerate(list(question)):
        if c in replaces:
            tokens.append(replaces[c])
            records.append((idx, c))
        else:
            tokens.append(c)
    return ''.join(tokens), records


def remove_space(words):
    """将空格移除，避免干扰根据前后项识别词性，后面需要再把空格恢复回去"""
    records = []
    outs = []
    st, ed = 0, 0
    for word, flag in words:
        st = ed
        ed = st + len(word)
        if word == ' ':
            records.append((st, ' '))
        else:
            outs.append((word, flag))
    return outs, records


def math_pos_seg(question):
    """数学词性标注"""
    words = combine_pos_with_re(question)
    words, records = remove_space(words)
    # print(f"jieba pos tag: {words}")
    idx = 0

    outs = []
    while idx < len(words):
        word, flag = words[idx]
        if word.startswith('第'):
            n_word, n_flag, idx = process_ord(words, idx)
            outs.append((n_word, n_flag))
        elif flag == 'm':  # 词性标注为数量词，可能是纯数字，也可能是数字加单位，也可能与后面的token一起构成其他成分
            n_word, n_flag, idx = process_m(words, idx, outs)
            outs.append((n_word, n_flag))
        elif flag == 'eng' or pure_eng(word):
            n_word, n_flag, idx = process_eng(words, idx, outs)
            outs.append((n_word, n_flag))
        elif flag == MathTag.NUM or pure_number(word):  # 有些时候数词会识别错误
            n_word, n_flag, idx = process_number(words, idx)
            outs.append((n_word, n_flag))
        elif word in units or flag == 'q' or flag == MathTag.UNIT:  # 单位
            n_word, n_flag, idx = process_unit(words, idx, outs)
            outs.append((n_word, n_flag))
        elif word in math_keys:  # 数学专业词语
            n_word, n_flag, idx = process_math_keys(words, idx)
            outs.append((n_word, n_flag))
        elif flag == 'nr':  # 人物
            n_word, n_flag, idx = process_nr(words, idx)
            outs.append((n_word, n_flag))
        elif is_power(word):
            if outs:
                last_word, last_flag = outs[-1]
                if last_word.endswith('m') or pure_number(
                        last_word):  # 前一个是单位或数量或纯数字，如cm<sup>2</sup>，表示平方厘米。涉及的单位缩写都是xm
                    n_word = last_word + word
                    if '°' in word:
                        n_flag = MathTag.AMOUNT
                    else:
                        n_flag = last_flag
                    outs = outs[:-1]
                    outs.append((n_word, n_flag))
                    idx += 1
                    continue
            outs.append((word, MathTag.MATH))
        elif flag == 'x':  # 各种符号
            n_word, n_flag, idx = process_x(words, idx)
            outs.append((n_word, n_flag))
        elif word in symbols:
            n_word, n_flag, idx = process_symbol(words, idx)
            outs.append((n_word, n_flag))
        elif word in people:
            outs.append((word, MathTag.PER))
        else:
            outs.append((word, flag))
        idx += 1
    outs = post_process(outs, records)
    recover_text = ''.join([w for w, f in outs])
    if recover_text != question:
        print(f'error, recover text: {recover_text} is not same with input question text: {question}')
    return outs


if __name__ == '__main__':
    text = "小船的船桨长1.3※米，伸入水中的部分长0.2米，露出水面的部分长(※，)米。"
    pos = math_pos_seg(text)
    print(pos)
