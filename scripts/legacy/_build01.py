# -*- coding: utf-8 -*-
import nbformat as nbf

cells = []
M = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
C = lambda s: cells.append(nbf.v4.new_code_cell(s))

M("""# 01 · 预处理：合并 / 去重 / 清洗 / 脱敏

**数据来源**：`data/raw/` 下三平台评论 CSV，由 `Desktop/新建文件夹 (2)` 复制而来（原文件只读，未改动）。

**口径说明（重要）**：原 `比赛.ipynb` 中**不存在**分词逻辑与停用词表——`jieba` 仅被 import 但从未调用，
所有图表数字为硬编码，词云频率来自 `np.random.randint()`。因此本项目的分词与停用词口径由
`src/text_utils.py` **新建**，并非沿用。

**输出**：`data/processed/comments_clean.csv`

**脱敏口径**：除删除昵称/user_id/头像/IP 属地等字段外，**平台原始 ID
（`comment_id` / `post_id` / `parent_comment_id`）一律不写入输出**——这些 ID 可用于
定位原帖并反查作者，属间接标识信息。跨文件引用改用项目内随机序号 `row_id`。
去重仍在内存中使用原始 `comment_id`，逻辑不变。""")

C("""import sys, re
from pathlib import Path
import numpy as np, pandas as pd

PROJ = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(PROJ / 'src'))
import text_utils as T

RAW  = PROJ / 'data' / 'raw'
PROC = PROJ / 'data' / 'processed'
PROC.mkdir(parents=True, exist_ok=True)
pd.set_option('display.width', 200)

print('项目根目录 :', PROJ)
print('停用词数量 :', len(T.STOPWORDS))
print('待脱敏列数 :', len(T.PII_COLUMNS), '->', T.PII_COLUMNS)""")

M("## 1. 读取并统一三平台表结构\n\n三平台字段名不同，先映射到统一 schema。")

C("""PLATFORM_NAME = {'bili': 'B站', 'douyin': '抖音', 'zhihu': '知乎'}
COLMAP = {
    'bili':   {'create_time': 'created_at', 'video_id':   'post_id'},
    'douyin': {'create_time': 'created_at', 'aweme_id':   'post_id'},
    'zhihu':  {'publish_time':'created_at', 'content_id': 'post_id'},
}

def kw_from_filename(path):
    m = re.search(r'关键词[：:\s]*(.*?)\s*评论', Path(path).stem)
    return m.group(1).strip('：:,， ') if m else Path(path).stem

frames, per_file = [], []
for pdir in ['bili', 'douyin', 'zhihu']:
    for f in sorted((RAW / pdir).glob('*评论*.csv')):
        df = pd.read_csv(f, encoding='utf-8-sig')
        kw = kw_from_filename(f)
        df = df.rename(columns=COLMAP[pdir])
        df['platform'] = PLATFORM_NAME[pdir]
        df['keyword']  = kw
        frames.append(df)
        per_file.append({'平台': PLATFORM_NAME[pdir], '搜索关键词': kw, '原始条数': len(df)})

raw_all = pd.concat(frames, ignore_index=True, sort=False)

print('=== 各文件原始条数 ===')
print(pd.DataFrame(per_file).to_string(index=False))
print()
print('=== 各平台原始条数 ===')
print(raw_all['platform'].value_counts().rename('条数').to_string())
print()
print('合计原始评论:', len(raw_all))""")

M("## 2. 去重\n\n两层：① 同平台 `comment_id` 重复（同一条评论被多个搜索词命中）；② 同平台正文完全相同。")

C("""stages = []
def log(step, df):
    stages.append({'步骤': step, '剩余条数': len(df)})
    return df

d = raw_all.copy()
log('① 原始合并', d)

n_before_id = len(d)
d = d.drop_duplicates(subset=['platform', 'comment_id'], keep='first')
dup_id = n_before_id - len(d)
log('② 去重：同平台 comment_id', d)

d['_norm'] = d['content'].astype(str).str.strip()
n_before_txt = len(d)
d = d.drop_duplicates(subset=['platform', '_norm'], keep='first')
dup_txt = n_before_txt - len(d)
log('③ 去重：同平台相同正文', d)

print('关键词重叠导致的重复 :', dup_id, '条')
print('正文完全重复         :', dup_txt, '条')
print()
print('=== 去重前后对比（按平台）===')
cmp = pd.concat([
    raw_all['platform'].value_counts().rename('去重前'),
    d['platform'].value_counts().rename('去重后'),
], axis=1).fillna(0).astype(int)
cmp['减少'] = cmp['去重前'] - cmp['去重后']
cmp['减少占比'] = (cmp['减少'] / cmp['去重前'] * 100).round(1).astype(str) + '%'
cmp.loc['合计'] = [cmp['去重前'].sum(), cmp['去重后'].sum(), cmp['减少'].sum(),
                   f"{cmp['减少'].sum() / cmp['去重前'].sum() * 100:.1f}%"]
print(cmp.to_string())""")

M("## 3. 清洗与分词\n\n用 `text_utils.clean_text` / `tokenize`（去 HTML、URL、@昵称、表情、方括号占位符）。")

C("""d['content_clean'] = d['content'].map(T.clean_text)

n0 = len(d)
d = d[d['content_clean'].str.len() >= 4].copy()
log('④ 去掉清洗后空/过短(<4字)', d)
print('清洗后空或过短被剔除 :', n0 - len(d), '条')

d['tokens']   = d['content_clean'].map(T.tokenize)
d['n_tokens'] = d['tokens'].map(len)

n1 = len(d)
d = d[d['n_tokens'] >= 2].copy()
log('⑤ 去掉有效词 < 2', d)
print('有效词不足被剔除     :', n1 - len(d), '条')
print()
print('平均有效词数 :', round(d['n_tokens'].mean(), 1))
print('正文长度分位 :', d['content_clean'].str.len().describe()[['min','25%','50%','75%','max']].round(0).to_dict())""")

M("## 4. 脱敏\n\n删除昵称 / user_id / 头像 / IP 属地等字段；`@昵称` 已在清洗阶段从正文移除。")

C("""present_pii = [c for c in d.columns if c in T.PII_COLUMNS]
print('本次实际删除的 PII 列 :', present_pii)

d_safe = d.drop(columns=present_pii + ['_norm', 'content'])

d_safe['created_at'] = pd.to_datetime(
    pd.to_numeric(d_safe['created_at'], errors='coerce'), unit='s', errors='coerce')

# 平台原始 ID 不进输出：可据此定位原帖→反查作者，属间接标识信息
DROP_IDS = ['comment_id', 'parent_comment_id', 'post_id']
d_safe = d_safe.drop(columns=[c for c in DROP_IDS if c in d_safe.columns])

# 项目内随机序号（种子固定，可复现；随机排列避免序号隐含原始文件顺序）
rng = np.random.default_rng(42)
d_safe['row_id'] = rng.permutation(len(d_safe)) + 1

FINAL_COLS = ['row_id', 'platform', 'keyword', 'created_at',
              'content_clean', 'tokens', 'n_tokens',
              'like_count', 'sub_comment_count']
out = d_safe[[c for c in FINAL_COLS if c in d_safe.columns]].copy()

# --- 脱敏校验 ---
leftover = [c for c in out.columns if c in T.PII_COLUMNS]
assert not leftover, f'仍有 PII 列残留: {leftover}'
at_left = out['content_clean'].str.contains('@', na=False).sum()
assert at_left == 0, f'正文中仍有 @ 残留: {at_left} 条'
id_left = [c for c in out.columns if c in DROP_IDS]
assert not id_left, f'仍有平台原始 ID 残留: {id_left}'
assert out['row_id'].is_unique, 'row_id 不唯一'
print('校验通过：无 PII 列残留，无平台原始 ID，正文无 @昵称 残留')
print('row_id :', out['row_id'].min(), '-', out['row_id'].max(), '（随机排列，项目内唯一）')
print()
print('最终列 :', list(out.columns))
print('原始列 :', len(raw_all.columns), '列  ->  输出', len(out.columns), '列')""")

M("## 5. 落盘")

C("""out_path = PROC / 'comments_clean.csv'
to_save = out.copy()
to_save['tokens'] = to_save['tokens'].map(lambda t: ' '.join(t))   # 空格分隔，便于 02/03 直接用
to_save.to_csv(out_path, index=False, encoding='utf-8-sig')

print('已写出 :', out_path)
print('文件大小 :', round(out_path.stat().st_size / 1024, 1), 'KB')
print()
print('=== 全流程条数变化 ===')
st = pd.DataFrame(stages)
st['较上一步'] = st['剩余条数'].diff().fillna(0).astype(int)
print(st.to_string(index=False))
print()
print('=== 最终各平台条数 ===')
fin = out['platform'].value_counts().rename('条数').to_frame()
fin['占比'] = (fin['条数'] / len(out) * 100).round(1).astype(str) + '%'
print(fin.to_string())
print()
print('时间跨度 :', out['created_at'].min(), '->', out['created_at'].max())
print()
print('=== 样例 3 条 ===')
print('（用于方法说明的代表性片段，截断至 40 字，已去除所有标识信息，不指向具体用户）')
for _, r in out.head(3).iterrows():
    print(f"[{r['platform']}/{r['keyword']}] {r['content_clean'][:40]}")
    print(f"    tokens: {r['tokens'][:12]}")""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
               'language_info': {'name': 'python'}}
nbf.write(nb, 'notebooks/01_preprocessing.ipynb')
print('built notebooks/01_preprocessing.ipynb with', len(cells), 'cells')
