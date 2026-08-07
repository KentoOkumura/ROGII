# 要件

## 依頼

Kaggle notebook `exp083-v12-ml-oof-known-tvt-probe` の可視化を exp238 の
保存済み OOF と strict nested selector score に差し替えた診断 notebook を、同じ
`experiments/exp238_nested_hmm_exp226_selector_rank_slot_addonly_on_exp218/` 内に追加する。
各 row で selector が最も信頼した候補を明確に識別できるようにする。

## 制約

- Route: `ml_model`。exp238 の selector score は final LightGBM への add-only 特徴であり、top-1 candidate を exp238 prediction と誤認させない。
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- selector の「信頼度最大」は、予測絶対誤差 score が最小の candidate と定義する。
- score は各 row が outer-valid だった fold の `role=valid` だけを使い、outer-train score や fold 間平均に置き換えない。
- selector、final LightGBM、PF/Beam/HMM/exp226 candidate の新規学習・再生成は行わない。
- 初回実装後、ユーザーの明示依頼により Kaggle notebook を同じ canonical kernel で実行する。competition submit は行わない。
- exp083 v12 と共通する系列は同じ色を使う。PF ANCC `#1f77b4`、Beam `#ff7f0e`、LikPF `#2ca02c`、ML OOF `#e11d48`、exp226 `#a16207`、exp209 HMM `#7c3aed` を固定する。

## 受け入れ基準

- exp238 `lgb_mean` OOF、selector top-1 candidate path、true TVT、主要 candidate を well ごとに比較できる。
- 各 plot に selector top-1 candidate の色分け帯、top2−top1 predicted-error margin、candidate 凡例がある。
- 共通系列と selector top-1 色帯の共通 candidate は exp083 v12 の配色と一致する。
- title または manifest で dominant top-1 candidate、share、exp238 OOF RMSE、selector top-1 RMSE を確認できる。
- 5 outer fold の `role=valid` が 3,783,989 row を重複・欠損なく覆うことを fail-fast で検証する。
- selector summary の candidate 順と nested score 列順を検証し、top-1 code を candidate 名へ固定対応させる。
- Jupytext percent source、`.ipynb`、CPU/internet-off の Kaggle package が揃い、実行時だけ `run_on_push=true` とする。
- `py_compile`、`ruff --select F821`、Jupytext `--test`、JSON parse が通る。
- deterministic anchor として扱う場合は、feature content SHA、model SHA、prediction SHA、submission SHA、Kaggle kernel version が記録されている。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
