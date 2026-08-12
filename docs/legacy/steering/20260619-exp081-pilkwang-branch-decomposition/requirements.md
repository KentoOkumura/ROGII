# 要件

## 依頼

`pilkwang_branch_decomposition` を実装する。`exp079_public_artifact_replay_integrity_audit` v4 の保存済み output から、Pilkwang public notebook の final prediction と主要 branch の寄与を分解し、提出候補を絞るための audit table を保存する。

## 制約

- Route: `pf_beam`
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- 新規学習、直接 submit、Public LB tuning は行わない。
- 入力は `exp079` v4 の summary / submission summary / pairwise distances を正とする。
- 候補 CSV 本体が存在しない環境でも、保存済み summary だけで実行可能にする。不足する row-level diff は明示的に unavailable と記録する。

## 受け入れ基準

- `config.yaml` に `exp079` output path、branch roles、anchor labels、candidate selection policy を記録する。
- `pilkwang_branch_decomposition.py` が summary CSV / JSONL を読み、branch summary、branch-vs-final、anchor comparison、candidate decision、audit summary JSON を保存する。
- final / projected ridge-PF / pretrained LGBM / model-package-only / base / w0.50-0.60 / gated 0.003-0.010 を区別して集計する。
- exact-match / overlap override などの risk hits と、branch CSV 本体が未保存で row-level guard ができない制約を出力に含める。
- deterministic anchor として扱う場合は、feature content SHA、model SHA、prediction SHA、submission SHA、Kaggle kernel version が記録されている。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
