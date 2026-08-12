# 要件

## 依頼

exp280で確認したfold-stableなraw GR/typewell shift識別力を、exp226座標系のslow offset
exact HMMへ実装する。absolute TVTを自由探索せず、
`TVT_t = exp226_geop_t + delta_t`としてoffset `delta_t`と必要最小限のoffset-rateだけを
1 fixed grammarでdecodeする。

## 制約

- Route: `pf_beam`
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- 科学的decoder親はexp209、局所形状anchorはgroup-safe exp226 `tvt_geop`、先行readoutはexp280とする。
- exp226 `tvt_pred` / `gr_delta`をdecoderへ入れず、GRの二重利用を防ぐ。
- exp209 Gaussian raw-GR emission、known-prefix calibration、missing-GR処理を固定する。
- offset gridは`[-80, 80] ft`、step `0.35 ft`、offset-rateは41 states / span `+-0.10`へ事前固定する。
- 1 HMM variant / 773 well-runs / LightGBM config 0 / trained fold 0 / booster 0とする。
- 親/controlを再学習・再生成しない。Kaggle CPUのみ、GPU/TPU/internetはoffとする。
- guard通過と別途ユーザー承認までraw-test inference / submissionをfail-closedにする。

## 受け入れ基準

- compact self-contained Jupytext train / inference sourceと正規`.ipynb`が作成されている。
- `delta=0`がexp226局所差分をそのまま追うtransition contractをテストで確認している。
- true TVTは全773 candidate path凍結後のreadoutにだけ結合される。
- exp263 fixed OOF 8.238331を`1e-5 ft`以内で再構成できる。
- promotion guardはoverall gain 0.02以上、改善3/5 folds、near / 1000+ / hidden-like悪化0.02以下、worst-well +0.25以下とする。
- persistent-offset episode数はexp263以下、256/512行復帰率はexp263以上を必要条件にする。
- `make validate-exp`、Jupytext `--test`、構文、ruff F821、実験固有pytestが通る。
- deterministic anchor として扱う場合は、feature content SHA、model SHA、prediction SHA、submission SHA、Kaggle kernel version が記録されている。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
