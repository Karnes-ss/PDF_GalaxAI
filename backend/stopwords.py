"""
中英文停用词 + 学术场景下常见噪声词。
用于关键词提取阶段过滤掉无信息量的 token。
保持足够精简以便人工维护；如需扩展直接加到对应 set 里即可。
"""

from __future__ import annotations


# ---------------------------- 英文 ---------------------------- #
STOPWORDS_EN: set[str] = {
    # 冠词 / 代词 / 介词 / 连词
    "a", "an", "the", "this", "that", "these", "those",
    "i", "we", "you", "he", "she", "it", "they", "them",
    "my", "our", "your", "his", "her", "their", "its",
    "am", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "doing", "done",
    "have", "has", "had", "having",
    "will", "would", "can", "could", "shall", "should", "may", "might", "must",
    "of", "in", "on", "at", "by", "for", "with", "to", "from", "as",
    "and", "or", "but", "if", "else", "not", "no", "yes",
    "than", "then", "so", "because", "while", "when", "where", "which", "who", "whom",
    "how", "what", "why", "whose", "here", "there",
    "into", "onto", "over", "under", "between", "within", "without",
    "also", "such", "each", "every", "any", "some", "all", "many", "much",
    "more", "most", "less", "few", "several", "other", "another",
    # 学术论文中几乎无区分度的通用词
    "paper", "papers", "article", "study", "studies", "research",
    "work", "works", "method", "methods", "approach", "approaches",
    "model", "models", "result", "results", "figure", "figures",
    "table", "tables", "section", "sections",
    "show", "shows", "shown", "present", "presents", "presented",
    "propose", "proposes", "proposed", "use", "used", "using", "uses",
    "thus", "however", "therefore", "moreover", "although", "furthermore",
    "abstract", "introduction", "conclusion", "conclusions",
    "based", "respect", "regard", "respect",
    "fig", "eq", "ref", "refs", "et", "al",
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
}


# ---------------------------- 中文 ---------------------------- #
STOPWORDS_ZH: set[str] = {
    # 代词 / 介词 / 助词 / 连词
    "的", "了", "是", "在", "和", "与", "及", "或", "并",
    "我", "我们", "你", "你们", "他", "她", "他们", "它", "它们",
    "这", "这个", "这些", "那", "那个", "那些",
    "一", "一个", "一种", "一些", "该", "此", "其", "之",
    "上", "下", "中", "内", "外", "前", "后", "左", "右",
    "有", "没", "没有", "会", "可以", "能", "要", "应", "应该", "需要",
    "被", "把", "为", "由", "从", "到", "对", "对于", "关于",
    "但", "但是", "而", "而且", "并且", "所以", "因此", "如果",
    "因为", "由于", "通过", "根据", "按照", "基于", "使用", "采用",
    "还是", "或者", "以及", "以上", "以下", "以及", "之一", "之间",
    "不", "也", "就", "都", "很", "非常", "比较", "更", "最",
    # 学术写作高频无信息词
    "本文", "本研究", "研究", "方法", "模型", "结果", "实验",
    "论文", "文章", "表", "图", "节", "章", "数据",
    "表明", "显示", "说明", "介绍", "提出", "给出", "发现",
    "表 1", "图 1", "表1", "图1",
    "此外", "另外", "然而", "然后", "同时", "因而", "从而",
    "一般", "通常", "部分", "整体", "以下", "以上", "其中",
}


def is_stopword(token: str) -> bool:
    """同时判定中英文停用词。"""
    if not token:
        return True
    t = token.strip().lower()
    if t in STOPWORDS_EN:
        return True
    if token.strip() in STOPWORDS_ZH:
        return True
    return False
