# -*- coding: utf-8 -*-
"""
共享文本处理口径 (tokenization / stopwords / cleaning)。
全部 notebook 共用这里的函数，保证口径一致。

注意：原 比赛.ipynb 中不存在分词逻辑与停用词表（jieba 仅被 import 但从未调用，
词云频率来自 np.random.randint），因此本模块是新建口径，不是沿用。
"""
import re
import jieba

# ---------------- 停用词 ----------------
# 通用中文停用词
_GENERIC_STOPWORDS = set("""
的 了 是 在 我 有 和 就 不 人 都 一 一个 上 也 很 到 说 要 去 你 会 着 没有 看 好 自己 这
那 这个 那个 什么 怎么 为什么 因为 所以 但是 而且 还是 还有 如果 可以 已经 应该 可能
没 吧 呢 啊 呀 哦 嗯 哈 哈哈 嘛 啦 哎 唉 么 吗 呵 嘿 咯 咦 唔
我们 你们 他们 她们 它们 自己 大家 别人 有些 一些 一下 一样 一直 一点 有点 比较
就是 不是 不会 不用 不能 知道 觉得 感觉 时候 现在 以前 以后 之后 之前 然后 其实
真的 确实 当然 反正 只是 不过 而已 之类 等等 以及 或者 还 又 再 才 就 都 也 挺
这样 那样 怎样 这些 那些 如此 于是 因此 虽然 尽管 无论 除了 关于 对于 至于 由于
个 种 次 点 些 位 条 句 篇 楼 楼主 回复 评论 转发 点赞 谢谢 感谢
让 使 被 把 给 跟 向 从 用 为 对 与 及 并 或 且 等 呗 咋
上去 下来 起来 出来 过来 一般 非常 特别 十分 太 更 最 极 超 巨
""".split())

# 领域噪声词：搜索词本身与平台词，出现在每条里，对聚类无区分度
_DOMAIN_STOPWORDS = set("""
王老吉 凉茶 视频 UP up主 博主 知乎 抖音 b站 B站 bilibili 哔哩哔哩
关键词 内容 图片 链接 网页 手机 电脑 播放 弹幕
""".split())

STOPWORDS = _GENERIC_STOPWORDS | _DOMAIN_STOPWORDS

# 领域词典：避免被 jieba 切碎
_USER_WORDS = [
    "王老吉", "加多宝", "和其正", "国际罐", "WALOVI", "凉茶", "怕上火",
    "无糖", "零糖", "低糖", "含糖量", "糖水", "上火", "降火", "清热",
    "中草药", "草本", "植物基", "气泡水", "无糖茶", "即饮茶",
    "性价比", "智商税", "国潮", "包装设计", "代言人", "张凌赫",
    "藤椒青提", "尝鲜", "回购", "复购", "好运",
]
for _w in _USER_WORDS:
    jieba.add_word(_w)

# ---------------- 清洗 ----------------
_RE_URL       = re.compile(r'https?://\S+|www\.\S+')
_RE_HTML      = re.compile(r'<[^>]+>')
_RE_AT        = re.compile(r'@[\w\-一-鿿]{1,30}')          # @昵称 属于 PII，清掉
_RE_BRACKET   = re.compile(r'\[[^\]]{1,12}\]')                      # [doge] 等表情占位
_RE_EMOJI     = re.compile(
    '[\U0001F300-\U0001FAFF\U0001F1E6-\U0001F1FF☀-➿️⬀-⯿]')
_RE_KEEP      = re.compile(r'[^一-鿿0-9a-zA-Z。，！？、；：“”‘’（）\s]')
_RE_WS        = re.compile(r'\s+')

def clean_text(s):
    """清洗单条评论；返回清洗后的字符串（可能为空）。"""
    if s is None:
        return ""
    s = str(s)
    s = _RE_HTML.sub(' ', s)
    s = _RE_URL.sub(' ', s)
    s = _RE_AT.sub(' ', s)          # 去 @昵称（脱敏）
    s = _RE_BRACKET.sub(' ', s)
    s = _RE_EMOJI.sub(' ', s)
    s = _RE_KEEP.sub(' ', s)
    s = _RE_WS.sub(' ', s).strip()
    return s

def tokenize(s, min_len=2, drop_pure_digit=True):
    """jieba 精确模式 + 停用词过滤。返回 token list。"""
    toks = []
    for w in jieba.lcut(clean_text(s)):
        w = w.strip()
        if not w or w in STOPWORDS:
            continue
        if len(w) < min_len:
            continue
        if drop_pure_digit and w.isdigit():
            continue
        toks.append(w)
    return toks

def tokenize_joined(s, **kw):
    """给 TfidfVectorizer 用：返回空格分隔的分词串。"""
    return " ".join(tokenize(s, **kw))

# ---------------- 需脱敏的列 ----------------
PII_COLUMNS = [
    "nickname", "user_nickname", "user_id", "sec_uid", "short_user_id",
    "user_unique_id", "avatar", "user_avatar", "user_link",
    "ip_location", "sign", "user_signature", "sex", "pictures",
]

# ---------------- 品类相关性判定 ----------------
# MediaCrawler 按关键词抓「视频/帖子」，再抓该帖整个评论区，
# 因此评论区跑题的内容（奶茶店推荐、游戏、广告灌水）会一并入库。
# 用品类核心词判定一条评论是否真的在谈这个品类。
CORE_TERMS = [
    "王老吉", "凉茶", "加多宝", "和其正", "国际罐", "WALOVI",
    "上火", "降火", "怕上火", "红罐", "草本", "广药", "鸿道",
]

def is_on_topic(text):
    """是否提及品类核心词。"""
    if text is None:
        return False
    t = str(text)
    return any(k in t for k in CORE_TERMS)

# ---------------- 话题桶（02/04 共用，单一定义源） ----------------
# 可解释的关键词桶，沿用原 比赛.ipynb 雷达图的维度命名。
# 一条评论可命中多个话题。
TOPICS = {
    "甜度口味": ["甜", "太甜", "糖水", "含糖", "无糖", "低糖", "齁", "腻", "口味", "好喝", "难喝", "味道"],
    "功效上火": ["上火", "降火", "清热", "功效", "去火", "中药", "草本", "凉性", "药味", "养生"],
    "价格":     ["贵", "便宜", "价格", "性价比", "智商税", "划算", "多少钱", "涨价"],
    "包装设计": ["包装", "设计", "罐", "瓶", "颜值", "好看", "红罐", "外观"],
    "竞品商标": ["加多宝", "和其正", "商标", "官司", "分家", "侵权", "纠纷"],
    "国际化":   ["国际罐", "WALOVI", "海外", "出海", "国外", "外国", "全球", "英文"],
    "怀旧广告": ["广告", "怕上火", "小时候", "童年", "记忆", "以前", "代言", "春晚"],
}

# 报告重点关注的四个话题
FOCUS_TOPICS = ["甜度口味", "功效上火", "价格", "国际化"]

# 交叉格子样本量下限：低于此值不下结论
MIN_CELL_N = 20

def topic_hits(text):
    """返回该条评论命中的话题名列表。"""
    t = "" if text is None else str(text)
    return [name for name, kws in TOPICS.items() if any(k in t for k in kws)]

# ---------------- 活动引导特征识别 ----------------
# 案例来源：04 发现「国际化」话题 14 条里 7 条内容呈现活动引导特征——
# 逐字复制官方口号 + 互动诱导语（一键三连 / 抽奖）。
# 这类内容由品牌活动引导产生，与自发消费者意见混在一起会系统性推高正面率，
# 故在基准口径中单独标记并剔除。
#
# 说明：本模块只做「内容特征」判定，不对发布者身份或动机作任何认定。
CAMPAIGN_SIGNALS = {
    "官方口号逐字复制": ["向世界传递东方健康理念", "喝WALOVI", "全球品牌代言人",
                        "品牌全球代言人"],
    "互动诱导/抽奖":   ["一键三连", "已三连", "已3", "已经三连", "抽我", "抽奖",
                        "奖项设置", "一等奖", "周边", "应援", "转发抽", "看看我"],
    "代言人应援":       ["张凌赫"],
}

def campaign_flags(text):
    """返回命中的活动引导信号类别列表。"""
    t = "" if text is None else str(text)
    return [cat for cat, kws in CAMPAIGN_SIGNALS.items() if any(k in t for k in kws)]

def campaign_score(text):
    """命中的信号类别数（0-3）。>=2 视为高置信度活动引导内容。"""
    return len(campaign_flags(text))


# ---------------- 跨平台中文字体 ----------------
def get_cjk_font(verbose=False):
    """
    返回一个可用的中文 FontProperties，并同步设置 matplotlib 全局字体。
    按 Windows / macOS / Linux 常见字体依次探测；全部失败时回退默认字体并告警，
    不抛异常（保证 notebook 在非 Windows 环境仍能跑完）。
    """
    import warnings
    from pathlib import Path
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties, findfont, fontManager

    CANDIDATE_PATHS = [
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    ]
    CANDIDATE_NAMES = ["SimHei", "Microsoft YaHei", "PingFang SC",
                       "Hiragino Sans GB", "Noto Sans CJK SC", "WenQuanYi Zen Hei"]

    for fp in CANDIDATE_PATHS:
        if Path(fp).exists():
            font = FontProperties(fname=fp)
            plt.rcParams["font.sans-serif"] = [font.get_name(), "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            if verbose:
                print(f"中文字体: {font.get_name()}")      # 不打印系统绝对路径
            return font

    installed = {f.name for f in fontManager.ttflist}
    for name in CANDIDATE_NAMES:
        if name in installed:
            font = FontProperties(family=name)
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            if verbose:
                print(f"中文字体: {name} (按字体名)")
            return font

    warnings.warn("未找到任何中文字体，图中中文可能显示为方框。")
    plt.rcParams["axes.unicode_minus"] = False
    return FontProperties()
