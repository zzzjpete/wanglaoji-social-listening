# Wanglaoji (王老吉) Gen-Z Market Analysis

Social-media comment analysis for a student advertising-competition entry on **WALOVI**,
Wanglaoji's existing international herbal-tea can. Rebuilt from the raw crawler exports so
that every number in the repository traces back to data.

The headline result is partly a negative one: after de-duplication, relevance filtering and
removal of campaign-driven content, **253 of the original 1,108 comments (22.8%) remain
usable**, and two of the four topics the entry most needed — price and internationalisation —
fall below the minimum sample size for any conclusion. The repository is organised around
making that boundary explicit rather than hiding it.

---

## 1. Context

**Competition.** 5th Guangdong–Hong Kong–Macau Student International Advertising Festival
(第五届粤港澳大学生国际广告节). Competition work ran **Nov 2025 – Jan 2026**; this analysis was
**rebuilt in Sep 2026**.

**Brand scope — important.** *WALOVI* is Wanglaoji's own existing brand name for its
international can; it was not created by the team. The team's proposal was limited to
(a) a **low-sugar variant** of that product, (b) a **slogan** — 「轻爽一口，好运上头 / Luck
tastes lighter」, and (c) a **go-to-market plan** for overseas Gen-Z consumers. Nothing in this
repository should be read as brand ownership or as an official Wanglaoji position.

**Why the analysis was rebuilt.** The charts in the original competition notebook were produced
from values written directly in the code rather than computed from the collected comments, so
the analysis was rebuilt from the raw CSVs to make the figures reproducible. The original
notebook and its images are kept outside this repository and are not reproduced here.

---

## 2. Repository structure

```
notebooks/
  01_preprocessing.ipynb        merge, de-duplicate, clean, de-identify
  02_sentiment.ipynb            Chinese BERT sentiment + topic drift by year
  03_tfidf_kmeans.ipynb         TF-IDF + K-means      (kept as negative evidence)
  03b_embedding_kmeans.ipynb    sentence embeddings + K-means
  04_topic_analysis.ipynb       topic x sentiment x platform cross-tabs
  05_evidence_boundary.ipynb    L1/L2/L3 evidence grading + residue audit
  06_roi_scenario.ipynb         ROI scenario appendix (external inputs only)

src/text_utils.py               single source of truth: tokenisation, stopwords,
                                category relevance, topic buckets, campaign-pattern
                                signals, cross-platform CJK font
scripts/legacy/                 archived notebook-generation scaffolding (do not run)
data/sample/                    45-row de-identified sample (only data committed)
outputs/                        8 CSVs + 20 figures
```

`data/raw/` (14 original crawler CSVs) and `data/processed/` (12 derived CSVs) are excluded by
`.gitignore` — see [§8 Privacy](#8-privacy-and-data-handling).

---

## 3. Data and pipeline

Collected with [MediaCrawler](https://github.com/NanmiCoder/MediaCrawler) on 2025-11-22 across
three platforms, keyword `王老吉凉茶` and variants. Comment timestamps span
**2012-08-13 to 2025-11-22**.

| Stage | Remaining | Change |
|---|---|---|
| ① Merge 7 comment CSVs | 1,108 | — |
| ② De-duplicate on `comment_id` within platform | 1,017 | −91 (keyword overlap) |
| ③ De-duplicate on identical text within platform | 1,012 | −5 |
| ④ Drop empty / under 4 characters after cleaning | 982 | −30 |
| ⑤ Drop fewer than 2 content tokens | **881** | −101 |
| ⑥ Restrict to 2023 onward | 523 | control set A |
| ⑦ Require a category core term | 260 | off-topic rate 50.3% |
| ⑧ Remove high-confidence campaign-pattern content | **253** | canonical baseline |

Platform mix at stage ⑤ (n=881): Zhihu 496 · Bilibili 279 · Douyin 106.

**Why ⑦ is necessary.** MediaCrawler matches a keyword against *posts*, then collects the
*entire* comment thread, so threads that drift off-topic are collected too. In the 2023+ slice,
**50.3%** of comments mention none of the 13 category core terms (2012–2022: 38.8%). Observed
off-topic content includes other beverage brands, unrelated game discussion and promotional
boilerplate.

---

## 4. Method

| Step | Approach |
|---|---|
| Tokenisation | jieba, precise mode, 200-word stopword list, 32-term domain dictionary |
| Sentiment | `uer/roberta-base-finetuned-dianping-chinese` (102M, binary), `p_pos` split into three classes with a neutral band of **[0.35, 0.65]** |
| Topics | 7 interpretable keyword buckets; a comment may match several |
| Clustering | TF-IDF and sentence embeddings (`shibing624/text2vec-base-chinese`, 768-d, mean pooling + L2), K scanned 2–10 |
| Cluster admissibility | any K producing a cluster below **20 items** is rejected outright |
| Stability | 3 random seeds, ARI/AMI at both `n_init=10` and `n_init=1` |
| Cell reporting | every cross-tab cell carries its `n`; cells below **n=20** are marked insufficient and no conclusion is drawn |

Thresholds were fixed before the runs, not tuned afterwards. The clustering stop-loss
(silhouette ≥ 0.10 **and** `n_init=1` mean ARI ≥ 0.50) was set in advance.

---

## 5. Findings

Directly citable. Each row states the sample size it rests on; full caveats are in
`outputs/evidence_boundary.csv`.

| Finding | Figures | n |
|---|---|---|
| **Crawler off-topic rate is very high.** Half the 2023+ comments do not discuss the category at all. | 50.3% (2012–2022: 38.8%) | 523 |
| **Failing to filter off-topic content reverses the platform ranking.** Douyin looks like the most positive platform until filtered, then turns net-negative; Bilibili becomes the most positive. Douyin's off-topic share is almost entirely promotional boilerplate scoring `p_pos ≈ 0.98`. Direction is reliable; the percentage points are not. | Douyin positive 52.8% → 37.8%; Bilibili 36.3% → 42.4% | 523 → 260 |
| **Representation, not hyper-parameters, was the blocker for clustering.** Switching from TF-IDF to sentence embeddings turned an unstable partition into a stable one. | mean ARI (`n_init=1`) 0.0392 → 0.9196; zero-similarity document pairs 81.9% → 0.0% | 253–260 |
| **The stable embedding partition (K=2) separates cleanly by meaning:** product experience vs. brand/trademark dispute. Cross-validated against the independent topic buckets. | sweetness/taste 30.6% vs 1.4%; efficacy 29.8% vs 7.9% | 121 / 139 |
| **"Packaging & design" is strongly negative** — a leading drag on sentiment. | net sentiment −32.5 (positive 22.5% / negative 55.0%) | 40 |
| **"Competitor & trademark" is strongly negative** and is also the single largest topic. | net sentiment −29.2 (positive 25.5% / negative 54.7%) | 106 |

Net sentiment = positive% − negative%, computed on the canonical baseline (n=253).

---

## 6. Limitations

These have a defensible direction but must always be cited with the caveat attached.

| Observation | Value | n | Caveat |
|---|---|---|---|
| Overall sentiment is net negative | positive 33.2% / negative 46.6% / neutral 20.2% | 253 | Absolute percentages are highly sensitive to the neutral band: widening it from [0.45, 0.55] to [0.25, 0.75] moves neutral from 6.5% to 35.4%. The model is fine-tuned on restaurant reviews and skews negative on plain declarative sentences. **Cite the direction, never the point value.** |
| "Efficacy / heat-relief" is negative | net −15.9 | 44 | n barely clears the threshold; removing campaign-pattern content moved it from −8.5 to −15.9, i.e. it is sensitive to a handful of items. |
| "Sweetness / taste" is near neutral, marginally positive | net +2.6 | 39 | Only 11 of the 39 discuss sweetness *alone*; the rest also discuss trademark or packaging. The pure subset (11) is below the n=20 threshold, and +2.6 is within noise. |
| The embedding cluster structure is real but weak, and supports only a two-way split | silhouette 0.1617 (TF-IDF: 0.0581) | 260 | Clears this project's pre-set stop-loss of 0.10 but sits below the conventional 0.25–0.50 "reasonable structure" range. K=3–6 score *lower*, so no four-dimension segmentation is available. Clusters also correlate significantly with text length (p=0.0008) and platform (p=0.0004), so the split is not purely semantic. |
| Discussion has shifted from the trademark dispute toward product and internationalisation | competitor/trademark 64.4% → 40.8%; internationalisation 0.9% → 5.4% | 479 | The trademark decline is solid. The internationalisation rise is real in direction but tiny in absolute volume (see §7) and cannot be used to quantify overseas share of voice. |

Two further method-level limits: 4 comments exceeded the 256-token limit and were truncated
before sentiment scoring, and the keyword topic buckets over-assign on long multi-topic
comments (6.9% of items match three or more topics).

---

## 7. What this data cannot support

Listed so that these claims are not made from this dataset.

| Not supported | Detail | n |
|---|---|---|
| **Internationalisation / overseas sentiment and share of voice** | 7 of the 14 matching comments carry campaign-pattern signals (verbatim official slogan plus engagement prompts); 7 remain after removal, of which 2 are single clauses. Widening to all years still yields only 16. The apparent "78.6% positive / 0% negative" reflects a single brand campaign. | 14 → 7 effective |
| **Price / value-for-money** | Far below the n=20 threshold; widening to all years yields 9. Price sensitivity cannot be assessed from this dataset at all. | 5 |
| **A positive read on "nostalgia & advertising"** — initially a candidate finding, withdrawn | Net sentiment was +16.7 including campaign-pattern content and **−8.7** after removing 7 endorsement posts (a −25.4 swing). Those posts contain "advertising"/"endorsement" keywords and so landed in this bucket; the direction was driven by one campaign. | 30 → 23 |
| **Any topic × platform conclusion** | After splitting topics by platform, 18 of 21 cells fall below n=20. The largest single cell is competitor/trademark × Zhihu at n=80; the focus-topic cells are mostly single digits. | 18 / 21 cells |
| **Any TF-IDF cluster interpretation** | Silhouette 0.0581 (≈0); at `n_init=1`, seed-pair ARI reached −0.073 and −0.082, i.e. worse than random, and cluster sizes drifted from [187, 73] to [238, 22]. Retained only as comparative evidence in `03_tfidf_kmeans.ipynb`. | 260 |

### Campaign-pattern detection

7 items (0.8% of the 881-comment corpus) match two or more signal categories — verbatim
official slogan, engagement/giveaway prompts, endorsement support. All 7 are from Bilibili,
all classified positive, all with `p_pos ≥ 0.85` (corpus mean 0.399). Their leverage is
disproportionate: removing 0.8% of the corpus reverses the direction of one topic entirely.

The detector judges **content characteristics only** and makes no determination about who
posted, or why. `outputs/campaign_pattern_summary.csv` carries de-identified feature summaries
(platform, signal categories, length bin, score bin, topics) with no text and no platform IDs.

Full evidence register: `outputs/evidence_boundary.csv`

---

## 8. Privacy and data handling

Source files were treated as read-only throughout; the 14 originals were verified
byte-identical (MD5) against their source after copying.

**Removed at stage ①:** 14 personal-information columns — `nickname`, `user_nickname`,
`user_id`, `sec_uid`, `short_user_id`, `user_unique_id`, `avatar`, `user_avatar`, `user_link`,
`ip_location`, `sign`, `user_signature`, `sex`, `pictures`. `@mentions` are stripped from the
comment text, with assertions that fail the notebook if either check does not pass.

**Platform IDs are also removed.** `comment_id`, `post_id` and `parent_comment_id` are dropped
from every output, because they allow the original post — and therefore the author — to be
located. They are indirect identifiers, not neutral keys. De-duplication still uses
`comment_id` in memory; it is simply never written out. Cross-file reference uses `row_id`,
a project-local random permutation.

**What is committed:** notebooks (with outputs), `outputs/`, `src/`, `scripts/legacy/`, and a
single 45-row sample (`platform`, `content_clean`, `date` — dates to day precision only;
Zhihu 23 / Bilibili 15 / Douyin 7). `data/raw/` and `data/processed/` are excluded via a
whitelist `.gitignore` (`data/*` with `!data/sample/`).

Verbatim comment excerpts do appear in notebook outputs, truncated to 40 characters. They are
included for method illustration, carry no identifying information, and do not point to
individual users.

---

## 9. ROI appendix (06) — read the caveat first

**This scenario model is not based on the crawled data.** Because price (n=5) and
internationalisation (n≈7 effective) are both unusable, no bottom-up ROI estimate is possible
from this dataset.

The model therefore works backwards: it estimates annual incremental gross profit from
explicit assumptions and reports the **break-even marketing spend ceiling** as a range, so that
no budget figure has to be assumed in the first place.

> Break-even spend ceiling, P10–P90: **RMB 1.44M – 8.51M** (144–851 万元) — a 5.9× spread.

**5 of the 6 input variables have no source** and are labelled `S3` placeholders. The only
externally grounded input (NielsenIQ 2024, China beverage all-channel growth +5.9% — a
*domestic* figure used only as a category-climate proxy) moves the result by ±RMB 0.34M, while
unsourced assumptions account for **98% of the total swing**. Mintel's overseas tea-drinker
motivations (relaxation 42% / natural 40% / mood 31%, cited via the team's own deck) are
motivation shares, not conversion rates, and are deliberately *not* used as a penetration rate.

Cite only as a range with assumptions stated; never the P50, never a single number. The most
valuable next step is measuring first-year trial penetration (±RMB 4.77M swing on its own).

---

## 10. Reproduction

```bash
pip install -r requirements.txt
jupyter lab      # run notebooks in order: 01 → 02 → 03 → 03b → 04 → 05 → 06
```

The two Hugging Face models (~400 MB each) download on first run; everything executes on CPU
(02 and 03b take roughly one minute each). Random seeds are fixed at 42 throughout.

**The raw crawler data is not distributed**, for platform-terms and privacy reasons: the
original exports carry user identifiers and the comments are third-party user content. `01`
therefore cannot be re-run from a fresh clone — supply your own exports in the same schema if
you need to reproduce that stage.

**The committed notebooks already contain their full outputs**, so every table, figure and
number can be read directly without running anything.

**`06_roi_scenario.ipynb` runs standalone.** It reads nothing from `data/` and depends on no
crawler-derived file, so it executes on a fresh clone as-is.

Set `WLJ_RAW_SOURCE_DIR` to enable the optional source-integrity check in `05`; it is skipped
without error when unset.

`scripts/legacy/` contains the one-off scaffolding that originally generated the notebooks.
The notebooks are now authoritative — re-running the scaffolding overwrites them. See
`scripts/legacy/README.md`.

**Scale:** 7 notebooks, 117 cells, 2,270 lines of notebook code plus 205 lines in
`src/text_utils.py`; 20 figures; 8 published CSVs.

---

## 11. Acknowledgements

Competition team: Qijun Zhong, Yuxuan Liang, Jinxi Zhang

Built with [MediaCrawler](https://github.com/NanmiCoder/MediaCrawler) for collection, and the
`uer/roberta-base-finetuned-dianping-chinese` and `shibing624/text2vec-base-chinese` models
from the Hugging Face Hub.

External figures cited in the ROI appendix belong to their respective publishers
(NielsenIQ, Mintel) and are referenced, not redistributed.

---

*Analysis code and findings are the team's own work. WALOVI and 王老吉 are trademarks of their
respective owner; this is an independent student project with no affiliation or endorsement.*
