# 要件

## 依頼

`selector_topn_candidate_only_features` を実装する。

exp098 の rank-slot features から、上位 n 件に入った候補だけを使う feature set を作る。`n=1,2,3` を ablation し、候補集合全体に依存する統計や選ばれない候補由来の source flag を落とす。

## 制約

- Route: `ml_model`
- 親実験: `exp098_selector_rank_slot_features_on_exp073`
- base surface: exp098 と同じ exp073/exp072 196-feature cache。
- direct selector、soft average、candidate TVT path replacement は行わない。
- `top1_candidate_only` は top1 delta、score、source code、U slope、U curvature、U residual MAD に限定する。
- `top2_candidate_only` / `top3_candidate_only` は top-n 内 score margin、candidate delta spread、U spread を追加する。
- 候補集合全体の entropy / range、全 pairwise delta、source one-hot flag、未選択候補統計は使わない。
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。

## 受け入れ基準

- `experiments/exp107_selector_topn_candidate_only_features/` が存在する。
- `config.yaml` に `top1_candidate_only`、`top2_candidate_only`、`top3_candidate_only` が定義されている。
- 学習 notebook が Kaggle train で top1/top2/top3 の GroupKFold ablation を実行できる。
- feature schema に top-n candidate-only 列だけが追加され、exp098 の full rank-slot 64 列をそのまま使っていない。
- inference は train-side review まで disabled になっている。
- deterministic anchor として扱わない。Kaggle train 実行後は feature content SHA、model SHA、prediction SHA、Kaggle kernel version を記録する。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録する。
