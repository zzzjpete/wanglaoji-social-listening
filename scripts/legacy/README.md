# scripts/legacy —— notebook 生成脚手架（已归档）

## 这些是什么

`_build01.py` … `_build06.py` 是**一次性脚手架**，用 `nbformat` 把 notebook 的
markdown 与 code cell 拼出来并写进 `notebooks/`。当初这样做是为了绕开在 shell 里
手写含中文与嵌套引号的 `.ipynb` JSON。

## ⚠ 不要再运行它们

**`notebooks/*.ipynb` 是权威版本。** 这些脚本会**整体覆盖**对应的 notebook——
包括你在 Jupyter 里的任何手动修改和已保存的运行输出。

要改分析，直接编辑 `notebooks/` 下的 notebook。

保留这些脚本只为追溯「notebook 最初是怎么生成的」，属于历史记录，不是构建流程。

## 对应关系

| 脚本 | 生成 |
|---|---|
| `_build01.py` | `notebooks/01_preprocessing.ipynb` |
| `_build02.py` | `notebooks/02_sentiment.ipynb` |
| `_build03.py` | `notebooks/03_tfidf_kmeans.ipynb` |
| `_build03b.py` | `notebooks/03b_embedding_kmeans.ipynb` |
| `_build04.py` | `notebooks/04_topic_analysis.ipynb` |
| `_build05.py` | `notebooks/05_evidence_boundary.ipynb` |
| `_build06.py` | `notebooks/06_roi_scenario.ipynb` |

## 如果确实要重新生成（不推荐）

必须从**项目根目录**运行，因为脚本里的输出路径是相对路径：

```bash
cd <项目根目录>
python scripts/legacy/_build01.py     # 会覆盖 notebooks/01_preprocessing.ipynb
```

重新生成后 notebook 是**未执行**状态（无输出），需按依赖顺序重跑：

```
01 → 02 → 03 → 03b → 04 → 05 → 06
```

02 需要 BERT 推理、03b 需要计算句向量，两者各约 1 分钟（CPU）。

重跑完成后记得再做两件收尾：

1. 剥掉 jieba 打到 stderr 的本机缓存路径（会带上运行者的家目录）；
2. 清掉 cell 级 `execution` 时间戳。

## 与 `src/` 的区别

`src/text_utils.py` **不是**脚手架，而是 notebook 运行时 `import` 的共享模块
（分词口径、停用词、品类相关性判定、话题桶、活动引导特征识别、跨平台中文字体）。
它必须留在 `src/`。
