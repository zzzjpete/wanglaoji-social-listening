# -*- coding: utf-8 -*-
import nbformat as nbf

cells = []
M = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
C = lambda s: cells.append(nbf.v4.new_code_cell(s))

M("""# 03 · TF-IDF + K-means 聚类

**分词**：沿用 `src/text_utils.py` 的 jieba 口径（02 已把结果存进 `tokens` 列，直接复用，
保证 02/03 完全同口径）。

**两份并列**：
| 数据集 | 定义 | n |
|---|---|---|
| **主结果** | 2023+ 且提及品类核心词 | 260 |
| 对照 | 2023+ 全量（含 50.3% 离题） | 523 |

对照组的作用是**噪声诊断**：如果离题评论会被 K-means 甩成独立簇，就能量化噪声的规模和形态。

### K 值选择规则（本次新增约束）

1. 肘部法（inertia）与轮廓系数各出一张图；
2. 每个 K 同时报告**最小簇规模**；
3. **任何簇 < 20 条的 K 一律不可用**，不进入候选；
4. 在可用候选里按轮廓系数最高选 K；
5. 选定后做**稳定性检查**：换 3 个 `random_state` 重跑，用 ARI / AMI 衡量划分一致性。
   若不一致就如实报告结构不稳，不挑好看的结果留下。""")

C("""import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (silhouette_score, silhouette_samples,
                             adjusted_rand_score, adjusted_mutual_info_score)
from sklearn.metrics.pairwise import cosine_distances
warnings.filterwarnings('ignore')

PROJ = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(PROJ / 'src'))
import text_utils as T

PROC = PROJ / 'data' / 'processed'
FIG  = PROJ / 'outputs' / 'figures'
FIG.mkdir(parents=True, exist_ok=True)

font = T.get_cjk_font(verbose=True)          # 跨平台中文字体，含回退
plt.rcParams['figure.dpi'] = 110

df = pd.read_csv(PROC / 'comments_sentiment.csv', encoding='utf-8-sig',
                 parse_dates=['created_at'])
df['tokens'] = df['tokens'].fillna('')

DATASETS = {
    'primary': dict(name='主结果·2023+相关集', mask=df['is_primary']),
    'control': dict(name='对照·2023+全量',     mask=df['recent']),
}

MIN_CLUSTER = 20          # 硬约束：任何簇 < 20 条则该 K 不可用
K_RANGE     = range(2, 11)
SEEDS       = [0, 7, 2024]

print('读入 :', len(df), '条')
for k, v in DATASETS.items():
    print(f"  {k:8s} {v['name']:22s} n = {int(v['mask'].sum())}")""")

M("## 1. 各平台样本量（相关集 vs 全量）\n\n后面所有平台对比图都会标注 n。")

C("""rows = []
for key, v in DATASETS.items():
    sub = df[v['mask']]
    for plat, n in sub['platform'].value_counts().items():
        rows.append({'数据集': v['name'], '平台': plat, 'n': int(n),
                     '占比%': round(n / len(sub) * 100, 1)})
plat_tbl = pd.DataFrame(rows)

print('=== 各平台样本量 ===')
for key, v in DATASETS.items():
    t = plat_tbl[plat_tbl['数据集'] == v['name']].set_index('平台')[['n', '占比%']]
    t.loc['合计'] = [t['n'].sum(), 100.0]
    print(f"\\n{v['name']}  (总 n = {int(v['mask'].sum())})")
    print(t.to_string())

# 相关集中各平台的「存活率」：筛掉离题后还剩多少
rec = df[df['recent']]
surv = pd.DataFrame({
    '2023+全量 n': rec['platform'].value_counts(),
    '相关集 n':    rec[rec['on_topic']]['platform'].value_counts(),
})
surv['相关率%'] = (surv['相关集 n'] / surv['2023+全量 n'] * 100).round(1)
print('\\n=== 各平台相关率（筛掉离题后的存活比例）===')
print(surv.to_string())

fig, ax = plt.subplots(1, 2, figsize=(13, 4.3))
x, w = np.arange(len(surv)), 0.38
b1 = ax[0].bar(x - w/2, surv['2023+全量 n'], w, label='2023+ 全量', color='#AAB7B8')
b2 = ax[0].bar(x + w/2, surv['相关集 n'],    w, label='相关集', color='#E67E22')
for bars in (b1, b2):
    for b in bars:
        ax[0].text(b.get_x() + b.get_width()/2, b.get_height() + 3,
                   f'n={int(b.get_height())}', ha='center',
                   fontproperties=font, fontsize=9)
ax[0].set_xticks(x); ax[0].set_xticklabels(surv.index, fontproperties=font)
ax[0].set_ylabel('条数', fontproperties=font)
ax[0].set_title('各平台样本量：全量 vs 相关集', fontproperties=font, fontsize=12)
ax[0].legend(prop=font, fontsize=9); ax[0].grid(alpha=.3, axis='y')

bars = ax[1].bar(surv.index, surv['相关率%'], color='#5DADE2')
for b in bars:
    ax[1].text(b.get_x() + b.get_width()/2, b.get_height() + 1,
               f"{b.get_height():.1f}%", ha='center', fontproperties=font, fontsize=10)
ax[1].set_ylabel('相关率 %', fontproperties=font)
ax[1].set_title('各平台相关率（越低说明评论区越跑题）', fontproperties=font, fontsize=12)
ax[1].set_xticks(range(len(surv))); ax[1].set_xticklabels(surv.index, fontproperties=font)
ax[1].set_ylim(0, 100); ax[1].grid(alpha=.3, axis='y')

plt.tight_layout()
plt.savefig(FIG / '03_platform_n.png', bbox_inches='tight')
plt.show()""")

M("""## 2. 向量化 + K 值扫描

TF-IDF 用 02 已算好的 `tokens`（空格分隔），`tokenizer=str.split` 直接切，不再二次分词。""")

C("""def vectorize(sub, min_df=3, max_df=0.6):
    vec = TfidfVectorizer(tokenizer=str.split, token_pattern=None,
                          lowercase=False, min_df=min_df, max_df=max_df)
    X = vec.fit_transform(sub['tokens'])
    return vec, X


def scan_k(X, k_range=K_RANGE, seed=42):
    out = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        sizes = np.bincount(km.labels_, minlength=k)
        sil = silhouette_score(X, km.labels_, metric='cosine')
        out.append({'K': k,
                    'inertia': round(km.inertia_, 2),
                    '轮廓系数': round(sil, 4),
                    '最小簇': int(sizes.min()),
                    '最大簇': int(sizes.max()),
                    '可用': '是' if sizes.min() >= MIN_CLUSTER else '否',
                    '_sizes': sizes.tolist()})
    return pd.DataFrame(out)


scans, vecs, mats = {}, {}, {}
for key, v in DATASETS.items():
    sub = df[v['mask']].reset_index(drop=True)
    vec, X = vectorize(sub)
    vecs[key], mats[key] = vec, X
    scans[key] = scan_k(X)
    print(f"=== {v['name']} ===")
    print('n =', X.shape[0], '| 词表大小 =', X.shape[1],
          '| 稀疏度 =', f'{X.nnz / (X.shape[0] * X.shape[1]) * 100:.2f}%')
    print(scans[key].drop(columns='_sizes').to_string(index=False))
    print()""")

M("## 3. 肘部法（图 1/2）")

C("""fig, ax = plt.subplots(1, 2, figsize=(13, 4.3))
for a, (key, v) in zip(ax, DATASETS.items()):
    s = scans[key]
    a.plot(s['K'], s['inertia'], marker='o', lw=2, color='#2E86C1')
    a.set_title(f"肘部法 · {v['name']} (n={mats[key].shape[0]})",
                fontproperties=font, fontsize=12)
    a.set_xlabel('K', fontproperties=font)
    a.set_ylabel('inertia (簇内平方和)', fontproperties=font)
    a.set_xticks(list(K_RANGE))
    a.grid(alpha=.3)
plt.tight_layout()
plt.savefig(FIG / '03_elbow.png', bbox_inches='tight')
plt.show()

for key, v in DATASETS.items():
    s = scans[key]
    d1 = s['inertia'].diff()
    print(f"{v['name']} inertia 逐步降幅 :")
    print(pd.DataFrame({'K': s['K'], 'inertia': s['inertia'], '降幅': d1.round(2)})
          .to_string(index=False))
    print()""")

M("""## 4. 轮廓系数 + 最小簇规模（图 2/2）

灰色阴影 = 最小簇 < 20 的不可用区；只有落在可用区的 K 才进入候选。""")

C("""fig, axes = plt.subplots(2, 2, figsize=(13, 7.2))
for j, (key, v) in enumerate(DATASETS.items()):
    s = scans[key]
    a = axes[0, j]
    ok = s['可用'] == '是'
    a.plot(s['K'], s['轮廓系数'], marker='o', lw=2, color='#8E44AD', zorder=3)
    a.scatter(s.loc[~ok, 'K'], s.loc[~ok, '轮廓系数'], marker='x', s=110,
              color='#E74C3C', zorder=4, label='不可用 (最小簇<20)')
    a.scatter(s.loc[ok, 'K'], s.loc[ok, '轮廓系数'], s=70,
              color='#27AE60', zorder=4, label='可用候选')
    a.set_title(f"轮廓系数 · {v['name']} (n={mats[key].shape[0]})",
                fontproperties=font, fontsize=12)
    a.set_xlabel('K', fontproperties=font)
    a.set_ylabel('silhouette (cosine)', fontproperties=font)
    a.set_xticks(list(K_RANGE)); a.grid(alpha=.3)
    a.legend(prop=font, fontsize=8)

    a2 = axes[1, j]
    bars = a2.bar(s['K'], s['最小簇'],
                  color=['#27AE60' if o else '#E74C3C' for o in ok])
    a2.axhline(MIN_CLUSTER, color='#E74C3C', ls='--', lw=1.5)
    a2.axhspan(0, MIN_CLUSTER, color='grey', alpha=.18, zorder=0)
    a2.text(K_RANGE[-1], MIN_CLUSTER + 1, f'下限 {MIN_CLUSTER} 条',
            ha='right', fontproperties=font, color='#E74C3C', fontsize=9)
    for b in bars:
        a2.text(b.get_x() + b.get_width()/2, b.get_height() + 0.8,
                str(int(b.get_height())), ha='center',
                fontproperties=font, fontsize=9)
    a2.set_title(f"最小簇规模 · {v['name']}", fontproperties=font, fontsize=12)
    a2.set_xlabel('K', fontproperties=font)
    a2.set_ylabel('最小簇条数', fontproperties=font)
    a2.set_xticks(list(K_RANGE)); a2.grid(alpha=.3, axis='y')

plt.tight_layout()
plt.savefig(FIG / '03_silhouette_minsize.png', bbox_inches='tight')
plt.show()

CHOSEN = {}
for key, v in DATASETS.items():
    s = scans[key]
    cand = s[s['可用'] == '是']
    print(f"=== {v['name']} ===")
    if cand.empty:
        CHOSEN[key] = None
        print('  没有任何 K 满足最小簇 >= 20 —— 该数据集不做聚类结论')
    else:
        best = cand.loc[cand['轮廓系数'].idxmax()]
        CHOSEN[key] = int(best['K'])
        print('  可用候选 K :', cand['K'].tolist())
        print('  排除的 K   :', s.loc[s['可用'] == '否', 'K'].tolist(),
              '（最小簇分别为', s.loc[s['可用'] == '否', '最小簇'].tolist(), '）')
        print(f"  选定 K = {CHOSEN[key]}（候选中轮廓系数最高 = {best['轮廓系数']}，"
              f"最小簇 {int(best['最小簇'])} 条）")
    print()""")

M("""## 5. 稳定性检查

换 3 个 `random_state` 重跑选定的 K，两两比较划分一致性。

- **ARI / AMI**：1.0 = 完全一致，0 ≈ 随机。经验上 ARI > 0.75 算稳，< 0.5 说明结构不稳。
- 注意 `n_init=10` 本身会削弱随机性，所以这是**偏乐观**的稳定性估计；额外再用
  `n_init=1` 跑一遍，看最坏情况。""")

C("""stab_summary = {}
for key, v in DATASETS.items():
    k = CHOSEN[key]
    print('=' * 74)
    print(f"{v['name']} | K = {k}")
    if k is None:
        print('  跳过（无可用 K）'); continue
    X = mats[key]
    for n_init, tag in [(10, 'n_init=10（与主分析一致）'), (1, 'n_init=1（最坏情况）')]:
        labs = {sd: KMeans(n_clusters=k, n_init=n_init, random_state=sd).fit_predict(X)
                for sd in SEEDS}
        pairs = [(a, b) for i, a in enumerate(SEEDS) for b in SEEDS[i+1:]]
        recs = [{'对比': f'seed {a} vs {b}',
                 'ARI': round(adjusted_rand_score(labs[a], labs[b]), 4),
                 'AMI': round(adjusted_mutual_info_score(labs[a], labs[b]), 4)}
                for a, b in pairs]
        tb = pd.DataFrame(recs)
        print(f'\\n  {tag}')
        print('   ', tb.to_string(index=False).replace('\\n', '\\n    '))
        print('    簇规模 :', {sd: sorted(np.bincount(l, minlength=k).tolist(), reverse=True)
                              for sd, l in labs.items()})
        mean_ari = tb['ARI'].mean()
        verdict = ('稳定' if mean_ari > 0.75 else
                   '中等' if mean_ari > 0.5 else '不稳定')
        print(f'    平均 ARI = {mean_ari:.4f}  ->  {verdict}')
        if n_init == 10:
            stab_summary[key] = {'K': k, 'mean_ARI': round(mean_ari, 4), '判定': verdict}
        else:
            stab_summary[key]['mean_ARI_ninit1'] = round(mean_ari, 4)
    print()

print('=== 稳定性汇总 ===')
print(pd.DataFrame(stab_summary).T.to_string())""")

M("## 6. 正式聚类 + PCA 散点图")

C("""results = {}
for key, v in DATASETS.items():
    k = CHOSEN[key]
    if k is None:
        continue
    sub = df[v['mask']].reset_index(drop=True).copy()
    X = mats[key]
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    sub['cluster'] = km.labels_
    sub['sil'] = silhouette_samples(X, km.labels_, metric='cosine')
    results[key] = dict(sub=sub, km=km, X=X, vec=vecs[key], k=k)

fig, axes = plt.subplots(1, len(results), figsize=(7 * len(results), 5.2))
if len(results) == 1:
    axes = [axes]
for a, (key, r) in zip(axes, results.items()):
    Xd = r['X'].toarray()
    pca = PCA(n_components=2, random_state=42)
    xy = pca.fit_transform(Xd)
    cen = pca.transform(r['km'].cluster_centers_)
    sizes = np.bincount(r['sub']['cluster'], minlength=r['k'])
    cmap = plt.get_cmap('tab10')
    for c in range(r['k']):
        m = r['sub']['cluster'] == c
        a.scatter(xy[m, 0], xy[m, 1], s=24, alpha=.65, color=cmap(c),
                  label=f'簇{c} (n={sizes[c]})')
    a.scatter(cen[:, 0], cen[:, 1], marker='X', s=220, c='black', zorder=5)
    ev = pca.explained_variance_ratio_
    a.set_title(f"PCA 散点 · {DATASETS[key]['name']}\\n"
                f"K={r['k']}, n={len(r['sub'])}, 前两维解释方差 {ev.sum()*100:.1f}%",
                fontproperties=font, fontsize=12)
    a.set_xlabel(f'PC1 ({ev[0]*100:.1f}%)', fontproperties=font)
    a.set_ylabel(f'PC2 ({ev[1]*100:.1f}%)', fontproperties=font)
    a.legend(prop=font, fontsize=8)
    a.grid(alpha=.3)
plt.tight_layout()
plt.savefig(FIG / '03_pca_scatter.png', bbox_inches='tight')
plt.show()

for key, r in results.items():
    ev = PCA(n_components=2, random_state=42).fit(r['X'].toarray()).explained_variance_ratio_
    print(f"{DATASETS[key]['name']}: 前两维仅解释 {ev.sum()*100:.1f}% 方差 "
          f"—— 散点图只是投影示意，簇的真实分离看轮廓系数")""")

M("## 7. 每簇 top 20 特征词 + 5 条代表评论")
M("代表评论仅用于方法说明，已截断至 40 字并去除所有标识信息，不指向具体用户。")

C("""for key, r in results.items():
    sub, vec, km, X, k = r['sub'], r['vec'], r['km'], r['X'], r['k']
    terms = np.array(vec.get_feature_names_out())
    print('#' * 78)
    print(f"# {DATASETS[key]['name']}  |  K = {k}  |  n = {len(sub)}")
    print('#' * 78)
    for c in range(k):
        m = sub['cluster'] == c
        n_c = int(m.sum())
        centroid = km.cluster_centers_[c]
        top_idx = centroid.argsort()[::-1][:20]
        top = [f'{terms[i]}({centroid[i]:.3f})' for i in top_idx if centroid[i] > 0]

        # 代表评论：离质心最近的 5 条
        d = cosine_distances(X[m.values], centroid.reshape(1, -1)).ravel()
        rep = sub[m].iloc[np.argsort(d)[:5]]

        print(f"\\n{'─'*74}")
        print(f"【簇 {c}】n={n_c} ({n_c/len(sub)*100:.1f}%)  "
              f"平均轮廓={sub.loc[m,'sil'].mean():.3f}")
        print(f"  平台分布 : {sub.loc[m,'platform'].value_counts().to_dict()}")
        print(f"  情感分布 : {sub.loc[m,'sentiment'].value_counts().to_dict()}")
        if 'on_topic' in sub.columns:
            print(f"  品类相关 : {int(sub.loc[m,'on_topic'].sum())}/{n_c} "
                  f"({sub.loc[m,'on_topic'].mean()*100:.0f}%)")
        print(f"  top20 特征词 :")
        for i in range(0, len(top), 5):
            print('     ', '  '.join(top[i:i+5]))
        print(f"  代表评论 5 条 :")
        for _, rr in rep.iterrows():
            print(f"    - [{rr['platform']}/{rr['sentiment']}] {rr['content_clean'][:40]}")
    print()""")

M("## 8. 簇画像对比图（标注 n）")

C("""for key, r in results.items():
    sub, k = r['sub'], r['k']
    sizes = sub['cluster'].value_counts().sort_index()
    labels = [f'簇{c}\\n(n={sizes[c]})' for c in range(k)]

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))

    # 簇规模
    bars = ax[0].bar(range(k), sizes.values, color=plt.get_cmap('tab10')(range(k)))
    ax[0].axhline(MIN_CLUSTER, color='#E74C3C', ls='--', lw=1.4)
    for b in bars:
        ax[0].text(b.get_x()+b.get_width()/2, b.get_height()+1.5, f'n={int(b.get_height())}',
                   ha='center', fontproperties=font, fontsize=9)
    ax[0].set_xticks(range(k)); ax[0].set_xticklabels([f'簇{c}' for c in range(k)],
                                                     fontproperties=font)
    ax[0].set_title(f"簇规模 · {DATASETS[key]['name']}", fontproperties=font, fontsize=12)
    ax[0].set_ylabel('条数', fontproperties=font); ax[0].grid(alpha=.3, axis='y')

    # 平台构成
    ctp = pd.crosstab(sub['cluster'], sub['platform'], normalize='index') * 100
    bottom = np.zeros(k)
    for i, plat in enumerate(ctp.columns):
        ax[1].bar(range(k), ctp[plat], bottom=bottom, label=plat,
                  color=plt.get_cmap('Set2')(i))
        bottom += ctp[plat].values
    ax[1].set_xticks(range(k)); ax[1].set_xticklabels(labels, fontproperties=font, fontsize=8)
    ax[1].set_title('各簇平台构成 %', fontproperties=font, fontsize=12)
    ax[1].legend(prop=font, fontsize=8, loc='lower right'); ax[1].set_ylabel('%',
                                                                fontproperties=font)

    # 情感构成
    COL = {'正面': '#66c2a5', '负面': '#fc8d62', '中性': '#8da0cb'}
    cts = pd.crosstab(sub['cluster'], sub['sentiment'], normalize='index') * 100
    cts = cts.reindex(columns=['正面', '负面', '中性']).fillna(0)
    bottom = np.zeros(k)
    for s_ in ['正面', '负面', '中性']:
        ax[2].bar(range(k), cts[s_], bottom=bottom, label=s_, color=COL[s_])
        bottom += cts[s_].values
    ax[2].set_xticks(range(k)); ax[2].set_xticklabels(labels, fontproperties=font, fontsize=8)
    ax[2].set_title('各簇情感构成 %', fontproperties=font, fontsize=12)
    ax[2].legend(prop=font, fontsize=8, loc='lower right'); ax[2].set_ylabel('%',
                                                                fontproperties=font)

    plt.tight_layout()
    plt.savefig(FIG / f'03_cluster_profile_{key}.png', bbox_inches='tight')
    plt.show()

    print(f"=== {DATASETS[key]['name']} 各簇 × 平台（条数）===")
    print(pd.crosstab(sub['cluster'], sub['platform'], margins=True,
                      margins_name='合计').to_string())
    print()
    print(f"=== {DATASETS[key]['name']} 各簇 × 情感（条数）===")
    print(pd.crosstab(sub['cluster'], sub['sentiment'], margins=True,
                      margins_name='合计').to_string())
    print()""")

M("## 9. 落盘")

C("""for key, r in results.items():
    p = PROC / f'clusters_{key}.csv'
    cols = ['row_id', 'platform', 'keyword', 'created_at', 'year', 'on_topic',
            'content_clean', 'tokens', 'n_tokens', 'like_count',
            'p_pos', 'sentiment', 'topics', 'cluster', 'sil']
    r['sub'][[c for c in cols if c in r['sub'].columns]] \\
        .to_csv(p, index=False, encoding='utf-8-sig')
    print('已写出 :', p.name, '|', len(r['sub']), '条 | K =', r['k'],
          '|', round(p.stat().st_size/1024, 1), 'KB')

print()
print('图表 :')
for f in sorted(FIG.glob('03_*.png')):
    print('  -', f.name, round(f.stat().st_size / 1024), 'KB')

print()
print('=== 最终交付摘要 ===')
summ = []
for key, v in DATASETS.items():
    k = CHOSEN[key]
    row = {'数据集': v['name'], 'n': int(v['mask'].sum()), '选定K': k}
    if key in stab_summary:
        row['平均ARI(n_init=10)'] = stab_summary[key]['mean_ARI']
        row['平均ARI(n_init=1)']  = stab_summary[key].get('mean_ARI_ninit1')
        row['稳定性判定'] = stab_summary[key]['判定']
    summ.append(row)
print(pd.DataFrame(summ).to_string(index=False))""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
               'language_info': {'name': 'python'}}
nbf.write(nb, 'notebooks/03_tfidf_kmeans.ipynb')
print('built notebooks/03_tfidf_kmeans.ipynb with', len(cells), 'cells')
