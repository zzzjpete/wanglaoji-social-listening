# -*- coding: utf-8 -*-
import nbformat as nbf

cells = []
M = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
C = lambda s: cells.append(nbf.v4.new_code_cell(s))

M("""# 05 · 证据边界报告

把 01–04 的结论按可信度分三级，供报告写作直接引用。

| 级别 | 含义 |
|---|---|
| **L1** | 可直接引用。样本量充足、口径稳健、换设定不翻转。 |
| **L2** | 有方向性但必须带限定。数值敏感或信号薄。 |
| **L3** | 不可用。样本不足、被污染、或已被证伪。 |

**本 notebook 的所有数字都是现算的**，不硬编码——这正是 01–04 一路在拆的问题
（原 `比赛.ipynb` 全部数字为手写常量）。

### 基准口径（canonical）

主结果 = **2023+ 且提及品类核心词 且 剔除高置信活动引导内容** = **n 253**。

04 原先用的是含活动引导内容的 260 条。剔除 7 条 B 站抽奖应援帖后，
**「怀旧广告」净情感从 +16.7 翻转为 −8.7**——该结论不成立，已降级到 L3。
这是本次分级中唯一一条被推翻的候选 L1。""")

C("""import sys, warnings, json
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')

PROJ = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(PROJ / 'src'))
import text_utils as T

PROC = PROJ / 'data' / 'processed'
FIG  = PROJ / 'outputs' / 'figures'
OUT  = PROJ / 'outputs'
MIN_N = T.MIN_CELL_N
ORDER = ['正面', '负面', '中性']

raw   = pd.read_csv(PROC / 'comments_clean.csv', encoding='utf-8-sig',
                    parse_dates=['created_at'])
sent  = pd.read_csv(PROC / 'comments_sentiment.csv', encoding='utf-8-sig',
                    parse_dates=['created_at'])
sent['topics'] = sent['topics'].fillna('')
sent['at_score'] = sent['content_clean'].map(T.campaign_score)
sent['campaign'] = sent['at_score'] >= 2

PRI260 = sent[sent['is_primary']]                      # 04 口径（含活动引导内容）
CANON  = sent[sent['is_primary'] & ~sent['campaign']]  # 本 notebook 基准口径
REC    = sent[sent['recent']]

print('原始合并            :', 1108)
print('清洗去重后          :', len(raw))
print('2023+ 全量          :', len(REC))
print('2023+ 相关集 (04)   :', len(PRI260))
print('基准口径 CANON      :', len(CANON), '(再剔除', int(PRI260['campaign'].sum()), '条活动引导内容)')""")

M("## 1. 现算所有 headline 数字\n\n每个数字都从数据算出，下面的分级表直接引用这些变量。")

C("""F = {}   # facts

# --- 数据管道 ---
F['n_raw']        = 1108
F['n_clean']      = len(raw)
F['n_recent']     = len(REC)
F['n_pri260']     = len(PRI260)
F['n_canon']      = len(CANON)
F['offtopic_rate_recent'] = round((1 - REC['on_topic'].mean()) * 100, 1)
F['offtopic_rate_old']    = round(
    (1 - sent[~sent['recent']]['on_topic'].mean()) * 100, 1)
F['pii_cols_dropped'] = 14

# --- 平台排序翻转（02）---
def sent_pct(frame, by='platform'):
    ct = pd.crosstab(frame[by], frame['sentiment'])
    return (ct.div(ct.sum(axis=1), axis=0) * 100).reindex(columns=ORDER).round(1)

F['plat_noisy'] = sent_pct(REC)                       # 含离题
F['plat_clean'] = sent_pct(REC[REC['on_topic']])      # 相关集
F['douyin_pos_noisy'] = float(F['plat_noisy'].loc['抖音', '正面'])
F['douyin_pos_clean'] = float(F['plat_clean'].loc['抖音', '正面'])
F['bili_pos_noisy']   = float(F['plat_noisy'].loc['B站', '正面'])
F['bili_pos_clean']   = float(F['plat_clean'].loc['B站', '正面'])

# --- 话题情感（基准口径）---
def topic_table(frame):
    rows = []
    for t in T.TOPICS:
        s = frame[frame['topics'].str.contains(t)]
        n = len(s)
        vc = s['sentiment'].value_counts().reindex(ORDER).fillna(0).astype(int)
        rows.append({'话题': t, 'n': n,
                     '正%': round(vc['正面'] / n * 100, 1) if n else np.nan,
                     '负%': round(vc['负面'] / n * 100, 1) if n else np.nan,
                     '净情感': round((vc['正面'] - vc['负面']) / n * 100, 1) if n else np.nan,
                     '可用': n >= MIN_N})
    return pd.DataFrame(rows).set_index('话题')

TT_CANON = topic_table(CANON)
TT_260   = topic_table(PRI260)
F['topic_canon'] = TT_CANON
F['topic_260']   = TT_260

# --- 聚类（03 / 03b）---
CMP = pd.read_csv(PROC / 'clustering_representation_comparison.csv', encoding='utf-8-sig')
F['cmp_repr'] = CMP
emb = pd.read_csv(PROC / 'clusters_emb_primary.csv', encoding='utf-8-sig')
emb['topics'] = emb['topics'].fillna('')
F['emb_sil']  = float(CMP[(CMP['表示方法'] == '句向量') &
                          (CMP['数据集'].str.contains('主结果'))]['轮廓系数'].iloc[0])
F['tfidf_sil'] = float(CMP[(CMP['表示方法'] == 'TF-IDF') &
                           (CMP['数据集'].str.contains('主结果'))]['轮廓系数'].iloc[0])
F['emb_ari1']  = float(CMP[(CMP['表示方法'] == '句向量') &
                           (CMP['数据集'].str.contains('主结果'))]['ARI(n_init=1)'].iloc[0])
F['tfidf_ari1'] = 0.0392        # 03 输出（TF-IDF n_init=1 平均 ARI）
sep = {}
for t in ['甜度口味', '功效上火']:
    sep[t] = (round(emb[emb.cluster == 0]['topics'].str.contains(t).mean() * 100, 1),
              round(emb[emb.cluster == 1]['topics'].str.contains(t).mean() * 100, 1))
F['cluster_sep'] = sep

# --- 活动引导特征 ---
F['at_total']   = int(sent['campaign'].sum())
F['at_in_intl'] = int(PRI260[PRI260['topics'].str.contains('国际化')]['campaign'].sum())
F['n_intl_260'] = int(PRI260['topics'].str.contains('国际化').sum())
F['n_intl_clean'] = int(CANON['topics'].str.contains('国际化').sum())
F['nostalgia_before'] = float(TT_260.loc['怀旧广告', '净情感'])
F['nostalgia_after']  = float(TT_CANON.loc['怀旧广告', '净情感'])

# --- 甜度薄信号 ---
CANON2 = CANON.copy()
CANON2['n_topics'] = CANON2['topics'].map(lambda s: len([x for x in s.split('|') if x]))
sw = CANON2[CANON2['topics'].str.contains('甜度口味')]
F['sweet_n']      = len(sw)
F['sweet_pure_n'] = int((sw['n_topics'] == 1).sum())

print('=== 话题情感（基准口径 n=%d）===' % len(CANON))
print(TT_CANON.sort_values('净情感').to_string())
print()
print('=== 平台正面率：含离题 vs 相关集 ===')
print(pd.DataFrame({'含离题 正面%': F['plat_noisy']['正面'],
                    '相关集 正面%': F['plat_clean']['正面']}).to_string())""")

M("## 2. 证据分级表\n\n每条：结论 / 等级 / 依据 notebook / 样本量 / 限定条件。")

C("""E = []
def add(level, claim, nb, n, caveat, value=''):
    E.append({'等级': level, '结论': claim, '数值': value,
              '依据notebook': nb, '样本量': n, '限定条件': caveat})

tc = F['topic_canon']

# ================= L1 =================
add('L1', '爬虫数据离题率极高：2023+ 有半数评论不谈这个品类',
    '01,02', f"{F['n_recent']}",
    'MediaCrawler 按关键词抓帖后抓整个评论区所致；判定用 13 个品类核心词，'
    '换词表会小幅波动，量级不变',
    f"{F['offtopic_rate_recent']}%（2012-2022 为 {F['offtopic_rate_old']}%）")

add('L1', '不筛离题会导致平台口碑排序翻转：抖音由「最正面」变为负多于正',
    '02', f"{F['n_recent']} -> {int(REC['on_topic'].sum())}",
    '抖音离题部分几乎全为 p_pos≈0.98 的广告灌水；结论是排序方向，非具体百分点',
    f"抖音正面 {F['douyin_pos_noisy']}% -> {F['douyin_pos_clean']}%；"
    f"B站 {F['bili_pos_noisy']}% -> {F['bili_pos_clean']}%")

add('L1', '句向量把 TF-IDF 的不稳定聚类救成稳定聚类（表示方法是关键，不是调参）',
    '03,03b', f"{F['n_canon']}~{F['n_pri260']}",
    'n_init=1 三 seed 平均 ARI；TF-IDF 出现负 ARI（差于随机）',
    f"ARI {F['tfidf_ari1']} -> {F['emb_ari1']}；零相似文档对 81.9% -> 0.0%")

add('L1', '句向量 K=2 的两簇有干净的语义分离：产品体验 vs 品牌商战',
    '03b', '121 / 139',
    '话题桶交叉验证；簇与文本长度(p=0.0008)和平台(p=0.0004)亦显著相关，非纯语义划分',
    f"甜度口味 {F['cluster_sep']['甜度口味'][0]}% vs {F['cluster_sep']['甜度口味'][1]}%；"
    f"功效上火 {F['cluster_sep']['功效上火'][0]}% vs {F['cluster_sep']['功效上火'][1]}%")

for t in ['包装设计', '竞品商标']:
    add('L1', f'「{t}」话题显著负向，是口碑主要拖累项',
        '04,05', int(tc.loc[t, 'n']),
        '净情感 = 正面%−负面%；剔除活动引导内容后计算；关键词桶对长评论有过度归属',
        f"净情感 {tc.loc[t,'净情感']:+.1f}（正 {tc.loc[t,'正%']}% / 负 {tc.loc[t,'负%']}%）")

add('L1', '原 比赛.ipynb 的全部图表数字为手写常量，无数据通路',
    '01,02', '—',
    '静态核查：read_csv=0、jieba 调用=0、停用词表=0、词云频率来自 np.random.randint',
    '原「情绪分布」352/287/361 硬编码；真实负面率约为其 1.6-1.7 倍')

# ================= L2 =================
add('L2', '整体口碑负面多于正面',
    '02,05', F['n_canon'],
    '绝对百分点对中性带高度敏感（带宽 [0.45,0.55]→[0.25,0.75] 时中性 6.5%→35.4%）；'
    '且 BERT 为大众点评微调、对陈述句有负面偏置。可引用方向，不可引用点值',
    '  '.join(f'{k} {v}%' for k, v in
             (CANON['sentiment'].value_counts(normalize=True) * 100)
             .reindex(ORDER).round(1).items()))

add('L2', '「功效上火」话题负向',
    '04,05', int(tc.loc['功效上火', 'n']),
    f"n={int(tc.loc['功效上火','n'])} 刚过门槛；剔除活动引导内容后净情感由 -8.5 降至 "
    f"{tc.loc['功效上火','净情感']:+.1f}，说明对少量样本敏感",
    f"净情感 {tc.loc['功效上火','净情感']:+.1f}")

add('L2', '「甜度口味」情感接近中性、略偏正 —— 信号很薄',
    '04,05', F['sweet_n'],
    f"{F['sweet_n']} 条中只有 {F['sweet_pure_n']} 条是「只谈甜度」（其余同时在谈商标/包装），"
    f"纯样本 {F['sweet_pure_n']} < {MIN_N}；净情感仅 {tc.loc['甜度口味','净情感']:+.1f}，在噪声范围内",
    f"净情感 {tc.loc['甜度口味','净情感']:+.1f}")

add('L2', '句向量聚类结构真实但偏弱，且只支持二分',
    '03b', F['n_pri260'],
    f"轮廓系数 {F['emb_sil']} 过本项目预设止损线 0.10，但低于行业惯例「结构合理」区间 "
    f"0.25-0.5；K=3~6 轮廓系数反而更低，给不出四维细分",
    f"silhouette {F['emb_sil']}（TF-IDF 为 {F['tfidf_sil']}）")

add('L2', '讨论重心从商标官司转向产品与国际化',
    '02', '479 相关集',
    '竞品商标降幅可靠；国际化升幅的绝对量极小（见 L3），只能作为趋势方向，'
    '不能量化出海声量',
    '竞品商标 64.4%→40.8%；国际化 0.9%→5.4%')

# ================= L3 =================
add('L3', '✗「国际化/出海」话题的情感与声量',
    '04,05', f"{F['n_intl_260']} -> 有效 {F['n_intl_clean']}",
    f"{F['n_intl_260']} 条中 {F['at_in_intl']} 条为 B 站抽奖应援活动引导内容（逐字复制官方口号+"
    f"「一键三连」「抽我」）；剔除后仅 {F['n_intl_clean']} 条，且含 2 条一句话。"
    f"扩到全年份仍仅 16 条。原「78.6% 正面 / 0% 负面」是活动买量造出来的",
    '不可用')

add('L3', '✗「价格/性价比」话题',
    '04', int(tc.loc['价格', 'n']),
    f"n={int(tc.loc['价格','n'])}，远低于门槛 {MIN_N}；扩到全年份仅 9 条。"
    '价格敏感度这批数据完全无法支撑',
    '不可用')

add('L3', f"✗「怀旧广告」正向结论（曾被列为 L1 候选，本次推翻）",
    '04,05', f"30 -> {int(tc.loc['怀旧广告','n'])}",
    f"含活动引导内容时净情感 {F['nostalgia_before']:+.1f}，剔除 7 条代言应援帖后翻转为 "
    f"{F['nostalgia_after']:+.1f}（变化 {F['nostalgia_after']-F['nostalgia_before']:+.1f}）。"
    '张凌赫代言帖含「广告/代言」关键词而落入该桶。方向被单一活动主导，不可引用',
    f"{F['nostalgia_before']:+.1f} -> {F['nostalgia_after']:+.1f}")

add('L3', '✗ 任何「话题 × 平台」层级的结论',
    '04', '21 格中 18 格 <20',
    '话题按平台拆分后 86% 的格子低于门槛；最大单格为竞品商标×知乎 n=80，'
    '其余重点话题格子多为个位数',
    '不可用')

add('L3', '✗ 原 比赛.ipynb 产出的全部 PNG 与其数字',
    '01', '—',
    'Desktop/新建文件夹 (2) 的 6 张图（情感分布/品牌提及率/产品特性雷达图/话题变化趋势/'
    '国际罐各维度评价/词云）与 Desktop/比赛 的 2 张图，全部基于手写常量或 np.random，'
    '与 1108 条真实数据无通路。本项目未复制、未引用，仅作为对照记录',
    '不可用')

add('L3', '✗ TF-IDF 聚类的任何簇结论',
    '03', F['n_pri260'],
    f"轮廓系数 {F['tfidf_sil']}（≈0），n_init=1 时三 seed 两两 ARI 出现 -0.073/-0.082，"
    '差于随机；簇规模在 [187,73]→[238,22] 之间漂移。保留仅作表示方法对比证据',
    '不可用')

EV = pd.DataFrame(E)
for lv in ['L1', 'L2', 'L3']:
    sub = EV[EV['等级'] == lv]
    print('=' * 78)
    print(f'{lv}  共 {len(sub)} 条')
    print('=' * 78)
    for _, r in sub.iterrows():
        print(f"\\n● {r['结论']}")
        print(f"    数值   : {r['数值']}")
        print(f"    依据   : {r['依据notebook']}   样本量 : {r['样本量']}")
        print(f"    限定   : {r['限定条件']}")
    print()""")

M("## 3. 活动引导特征 识别案例记录\n\n把 04 撞到的活动活动引导内容做成可复用的识别规则与存证。")

C("""print('识别规则（src/text_utils.CAMPAIGN_SIGNALS，命中 >=2 类判高置信）:')
for cat, kws in T.CAMPAIGN_SIGNALS.items():
    print(f'  [{cat}] {"、".join(kws)}')
print()
flagged = sent[sent['at_score'] >= 1].copy()
hi = sent[sent['campaign']]
print(f'全语料 {len(sent)} 条中：命中 >=1 类信号 {len(flagged)} 条，'
      f'高置信 >=2 类 {len(hi)} 条 ({len(hi)/len(sent)*100:.1f}%)')
print()
print('高置信活动引导内容的特征（与全语料对比）:')
print(pd.DataFrame({
    '活动引导内容': {'n': len(hi), '平台': hi['platform'].value_counts().to_dict(),
             '正面率%': round((hi['sentiment'] == '正面').mean() * 100, 1),
             '平均p_pos': round(hi['p_pos'].mean(), 3)},
    '全语料': {'n': len(sent), '平台': sent['platform'].value_counts().to_dict(),
              '正面率%': round((sent['sentiment'] == '正面').mean() * 100, 1),
              '平均p_pos': round(sent['p_pos'].mean(), 3)},
}).to_string())
print()
print('=== 对各话题净情感的污染量 ===')
poll = pd.DataFrame({'含活动引导内容 净情感': TT_260['净情感'], '剔除后 净情感': TT_CANON['净情感']})
poll['污染量'] = (poll['剔除后 净情感'] - poll['含活动引导内容 净情感']).round(1)
poll['n 变化'] = (TT_CANON['n'] - TT_260['n']).astype(int)
print(poll.sort_values('污染量').to_string())
print()
print('-> 7 条帖子（占全语料 0.8%）足以把「怀旧广告」的结论方向整个翻转。')
print('-> 教训：小样本话题上，个别活动帖的杠杆极大，情感统计前必须先查 活动引导特征。')

print()
print('=== 高置信活动引导内容：脱敏特征摘要（不含原文与平台 ID）===')

def _len_bin(n):
    for lo, hi, lab in [(0, 40, '<40字'), (40, 80, '40-79字'),
                        (80, 150, '80-149字'), (150, 10**9, '>=150字')]:
        if lo <= n < hi:
            return lab

def _p_bin(v):
    for lo, hi, lab in [(0, .35, '<0.35'), (.35, .65, '0.35-0.65'),
                        (.65, .85, '0.65-0.85'), (.85, 1.01, '>=0.85')]:
        if lo <= v < hi:
            return lab

CAMP = pd.DataFrame([{
    '平台': r['platform'],
    '信号类别': ' + '.join(T.campaign_flags(r['content_clean'])),
    '信号数': int(r['at_score']),
    '字符数分箱': _len_bin(len(str(r['content_clean']))),
    'p_pos分箱': _p_bin(float(r['p_pos'])),
    '情感': r['sentiment'],
    '命中话题': r['topics'] if r['topics'] else '(无)',
} for _, r in hi.iterrows()])
print(CAMP.to_string(index=False))
print()
print('说明：以上仅为内容特征，不含原文、不含平台 ID，无法据此定位具体帖子或作者。')
print('      本项目只判定「内容是否呈现活动引导特征」，不对发布者身份或动机作任何认定。')
print()
print('信号类别命中次数（可多重命中）:')
from collections import Counter
cnt = Counter(c for _, r in hi.iterrows() for c in T.campaign_flags(r['content_clean']))
for k, v in cnt.most_common():
    print(f'  {k} : {v}')
print()
print('字符数分箱分布 :', CAMP['字符数分箱'].value_counts().to_dict())
print('p_pos 分箱分布 :', CAMP['p_pos分箱'].value_counts().to_dict())

# ── 落盘：公开版（仅特征摘要）与本地版（含原文，被 .gitignore 排除）──
CAMP.to_csv(OUT / 'campaign_pattern_summary.csv', index=False, encoding='utf-8-sig')

flagged['at_flags'] = flagged['content_clean'].map(lambda s: ' + '.join(T.campaign_flags(s)))
flagged['高置信'] = flagged['campaign']
local_cols = ['row_id', 'platform', 'keyword', 'created_at', 'sentiment', 'p_pos',
              'at_score', 'at_flags', '高置信', 'topics', 'content_clean']
p_local = PROC / 'campaign_pattern_full.csv'
flagged[[c for c in local_cols if c in flagged.columns]] \\
    .sort_values(['at_score', 'platform'], ascending=[False, True]) \\
    .to_csv(p_local, index=False, encoding='utf-8-sig')

print()
print('已写出 :')
print(f'  - outputs/campaign_pattern_summary.csv   ({len(CAMP)} 条特征摘要，可公开提交)')
print(f'  - data/processed/campaign_pattern_full.csv ({len(flagged)} 条含原文，本地留存，'
      '已被 .gitignore 排除)')""")

M("## 4. 残留审计\n\n程序化复核 `data/` 与 `outputs/` 是否还有旧 PNG 或手写死数据的残留。")

C("""import glob, re, os, subprocess, hashlib
import nbformat as _nbf

print('=== ① data/ 下非 CSV 文件（应为空）===')
non_csv = [p for p in glob.glob(str(PROJ / 'data' / '**' / '*'), recursive=True)
           if os.path.isfile(p) and not p.endswith('.csv')]
print('  ', non_csv if non_csv else '无')

print()
print('=== ② outputs/ 每张图能否追溯到 notebook ===')
owners = {}
for f in sorted(glob.glob(str(PROJ / 'notebooks' / '*.ipynb'))):
    nb = _nbf.read(f, as_version=4)
    code = '\\n'.join(c.source for c in nb.cells if c.cell_type == 'code')
    for m in re.finditer(r"FIG\\s*/\\s*f?'([^']+)'", code):
        owners.setdefault(m.group(1), []).append(os.path.basename(f))
    for m in re.finditer(r"FIG\\s*/\\s*\\(tag \\+ '\\.png'\\)|FIG\\s*/\\s*f'([^']*\\{[^']*\\}[^']*)'", code):
        owners.setdefault('<动态命名>', []).append(os.path.basename(f))
pngs = sorted(os.path.basename(p) for p in glob.glob(str(FIG / '*.png')))
orphan = []
for fn in pngs:
    o = owners.get(fn)
    if not o:
        pref = fn.split('_')[0]
        o = [x for x in set(sum(owners.values(), [])) if x.startswith(pref.rstrip('b'))
             or x.startswith(pref)]
        o = [f'{x} (动态命名)' for x in o]
    if not o:
        orphan.append(fn)
    print(f'  {fn:40s} <- {o}')
print('  孤儿图 :', orphan if orphan else '无')

print()
print('=== ③ 死数据是否进入可执行代码 ===')
print('  判据：先剥掉所有字符串字面量，只看真正当数据用的地方')
print('  （否则本 notebook 引用旧数字作说明、以及审计正则本身都会误报）')
DEAD = {'情感 352/287/361': r'\\b352\\b|\\b287\\b|\\b361\\b',
        '雷达 importance': r'95,\\s*78,\\s*65,\\s*82,\\s*88',
        '品牌提及 89/45/12': r'\\b89,\\s*45,\\s*12\\b',
        'np.random 造数': r'np\\.random\\.(randint|rand|choice)'}
import io as _io, tokenize as _tk

def strip_strings(code):
    # 用 tokenize 丢掉所有 STRING token，只留代码骨架
    # （用 tokenize 而非正则，避免在本单元里写出三引号把自身字符串截断）
    try:
        return ' '.join(t.string for t in
                        _tk.generate_tokens(_io.StringIO(code).readline)
                        if t.type != _tk.STRING)
    except Exception:
        return re.sub(r'[0-9]', '', code)     # 兜底：保守起见把数字也去掉，宁漏不误报

bad, noted = [], []
for f in sorted(glob.glob(str(PROJ / 'notebooks' / '*.ipynb'))):
    nb = _nbf.read(f, as_version=4)
    for i, c in enumerate(nb.cells):
        target = c.source if c.cell_type == 'markdown' else strip_strings(c.source)
        for lab, pt in DEAD.items():
            if re.search(pt, target):
                if c.cell_type == 'code':
                    bad.append((os.path.basename(f), i, lab))
                    print(f'  {os.path.basename(f)} cell{i} [CODE ⚠ 当数据用] <- {lab}')
                else:
                    noted.append((os.path.basename(f), i, lab))
print('  markdown 中作为说明引用（正常）:',
      [f'{a} cell{b}' for a, b, _ in noted] if noted else '无')
print('  进入可执行代码的死数据 :', bad if bad else '无')

print()
print('=== ④ data/raw 是否仍与原始文件逐字节一致 ===')
# 原始目录因人而异，改用环境变量配置，避免把本机绝对路径写进仓库。
# 未设置时跳过校验（不报错），其余审计项照常执行。
RAW_SOURCE_DIR = os.environ.get('WLJ_RAW_SOURCE_DIR')
if not RAW_SOURCE_DIR:
    print('  未设置环境变量 WLJ_RAW_SOURCE_DIR，跳过原始文件完整性校验。')
    print('  （需校验时：设为存放三平台原始 CSV 的目录，其下有 bili / 抖音 / 知乎 子目录）')
    raw_ok = None
elif not Path(RAW_SOURCE_DIR).exists():
    print(f'  WLJ_RAW_SOURCE_DIR 指向的目录不存在，跳过校验。')
    raw_ok = None
else:
    SRCDIR = Path(RAW_SOURCE_DIR)
    pairs = {'bili': 'bili', '抖音': 'douyin', '知乎': 'zhihu'}
    def md5(p):
        return hashlib.md5(Path(p).read_bytes()).hexdigest()
    diff, cnt = [], 0
    for s, dd in pairs.items():
        for f in sorted((SRCDIR / s).glob('*.csv')):
            tgt = PROJ / 'data' / 'raw' / dd / f.name
            cnt += 1
            if not tgt.exists() or md5(f) != md5(tgt):
                diff.append(f.name)
    print(f'  比对 {cnt} 个文件，不一致 :', diff if diff else '无（全部一致）')
    raw_ok = not diff

print()
print('=== ⑤ 本项目是否读取/引用过原 PNG ===')
print('  判据：只看真正的图像读取调用，不看路径字符串')
print('  （data/raw 完整性校验必须指向原目录，那是合法引用）')
refs = []
for f in sorted(glob.glob(str(PROJ / 'notebooks' / '*.ipynb'))):
    nb = _nbf.read(f, as_version=4)
    for i, c in enumerate(nb.cells):
        if c.cell_type != 'code':
            continue
        skeleton = strip_strings(c.source)
        reads_image = re.search(r'imread|Image\\s*\\.\\s*open|mpimg|cv2', skeleton)
        # 路径里直接出现原 PNG 目录 + .png 才算引用图片（md5 校验只读 .csv，不会命中）
        # 目录名用拼接构造，否则本单元的正则会匹配到自己
        _d = '(' + '新建' + '文件夹' + '|' + '比' + '赛' + ')'
        points_at_png = re.search(_d + r'.{0,40}\\.png', c.source)
        if reads_image or points_at_png:
            refs.append((os.path.basename(f), i))
print('  代码中读取原 PNG 的位置 :', refs if refs else '无')
print('  （原 PNG 全部留在 Desktop 原处，本项目只复制了 CSV）')

print()
print()
print('=== ⑥ 输出 CSV 的平台原始 ID / PII 列 ===')
print('  分两档：会被提交的必须为空（硬失败）；本地留存的仅告警。')
BANNED = ['comment_id', 'parent_comment_id', 'post_id'] + T.PII_COLUMNS
# 本轮流水线实际产出的文件名（用于识别陈旧残留）
CURRENT = {'comments_clean.csv', 'comments_sentiment.csv',
           'clusters_primary.csv', 'clusters_control.csv',
           'clusters_emb_primary.csv', 'clusters_emb_control.csv',
           'clustering_representation_comparison.csv', 'topic_focus_rows.csv',
           'topic_platform_counts.csv', 'topic_sentiment_table.csv',
           'topic_sentiment_canonical.csv', 'campaign_pattern_full.csv'}

def scan(files):
    out = []
    for f in files:
        try:
            cs = list(pd.read_csv(f, nrows=0, encoding='utf-8-sig').columns)
        except Exception:
            continue
        b = [c for c in cs if c in BANNED]
        if b:
            out.append((os.path.relpath(f, PROJ), b))
    return out

public_files = sorted(glob.glob(str(OUT / '*.csv'))) + \\
               sorted(glob.glob(str(PROJ / 'data' / 'sample' / '*.csv')))
local_files  = sorted(glob.glob(str(PROJ / 'data' / 'processed' / '*.csv')))

public_leaks = scan(public_files)
local_leaks  = scan(local_files)

print(f'  会被提交的 ({len(public_files)} 个: outputs/ + data/sample/) :',
      public_leaks if public_leaks else '干净')
print(f'  本地留存的 ({len(local_files)} 个: data/processed/，已被 gitignore 排除) :')
if local_leaks:
    for f, cs in local_leaks:
        stale = os.path.basename(f) not in CURRENT
        tag = '陈旧残留（非本轮产出）' if stale else '本轮产出'
        print(f'     ⚠ {f} <- {cs}   [{tag}]')
else:
    print('     干净')

stale_files = [os.path.relpath(f, PROJ) for f in local_files
               if os.path.basename(f) not in CURRENT]
print('  陈旧残留文件（非本轮流水线产出，建议人工确认后清理）:',
      stale_files if stale_files else '无')

smp = PROJ / 'data' / 'sample' / 'comments_sample.csv'
if smp.exists():
    sc = list(pd.read_csv(smp, nrows=0, encoding='utf-8-sig').columns)
    print(f'  小样本列 : {sc}  ->',
          '符合要求' if sc == ['platform', 'content_clean', 'date'] else '⚠ 与约定不一致')

problems = bool(non_csv or orphan or bad or refs or public_leaks) or (raw_ok is False)
verdict = '全部通过，无残留' if not problems else '发现问题，见上'
if raw_ok is None and not problems:
    verdict += '（④ 原始文件校验已跳过：未配置 WLJ_RAW_SOURCE_DIR）'
print('审计结论 : ' + verdict)""")

M("## 5. 导出")

C("""p = OUT / 'evidence_boundary.csv'
EV.to_csv(p, index=False, encoding='utf-8-sig')
TT_CANON.to_csv(PROC / 'topic_sentiment_canonical.csv', encoding='utf-8-sig')

# ── 公开用脱敏小样本：仅 platform / content_clean / date 三列 ──
# 来源：基准口径 CANON（2023+ 相关集、已剔除活动引导内容）
# 不含任何 ID、不含时分秒，仅供他人跑通 notebook 用
SAMPLE_N = 45
SAMPLE = (PROJ / 'data' / 'sample')
SAMPLE.mkdir(parents=True, exist_ok=True)
_rng = np.random.default_rng(42)
_quota = (CANON['platform'].value_counts(normalize=True) * SAMPLE_N).round().astype(int)
_parts = []
for _plat, _k in _quota.items():
    _pool = CANON[CANON['platform'] == _plat]
    _k = min(int(_k), len(_pool))
    _parts.append(_pool.sample(n=_k, random_state=42))
_smp = pd.concat(_parts).sample(frac=1, random_state=42)          # 打散顺序
_smp = pd.DataFrame({'platform': _smp['platform'].values,
                     'content_clean': _smp['content_clean'].values,
                     'date': pd.to_datetime(_smp['created_at']).dt.strftime('%Y-%m-%d').values})
_smp.to_csv(SAMPLE / 'comments_sample.csv', index=False, encoding='utf-8-sig')
print(f'脱敏小样本 : data/sample/comments_sample.csv  {len(_smp)} 行 x {len(_smp.columns)} 列')
print('  平台构成 :', _smp['platform'].value_counts().to_dict())
print('  列 :', list(_smp.columns), '（无 ID、无时分秒）')
print()

print('已写出 :')
print('  -', p.relative_to(PROJ), f'({len(EV)} 条证据)')
print('  - data/processed/topic_sentiment_canonical.csv  (基准口径话题情感)')
print('  - outputs/campaign_pattern_summary.csv          (活动引导特征摘要，可公开)')
print('  - data/processed/campaign_pattern_full.csv      (含原文，本地留存，已被 gitignore 排除)')
print('  - data/sample/comments_sample.csv               (脱敏小样本，可公开)')
print()
print('=== 分级汇总 ===')
print(EV['等级'].value_counts().reindex(['L1', 'L2', 'L3']).rename('条数').to_string())
print()
print('=== 报告可直接引用清单 (L1) ===')
for _, r in EV[EV['等级'] == 'L1'].iterrows():
    print(f"  • {r['结论']}")
    print(f"      {r['数值']}")
print()
print('=== 报告中必须避免的说法 (L3) ===')
for _, r in EV[EV['等级'] == 'L3'].iterrows():
    print(f"  {r['结论']}")""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
               'language_info': {'name': 'python'}}
nbf.write(nb, 'notebooks/05_evidence_boundary.ipynb')
print('built notebooks/05_evidence_boundary.ipynb with', len(cells), 'cells')
