# 要件

## 依頼

exp226の`K=16`だけでは見えていないsegment解像度の効果を、`K=12`と`K=24`の固定2点で監査する。
`KAGGLE_DIRECTION.md`のバックログ、実験ディレクトリ、steeringを作成して設計を確定する。
実装、Notebook採用、Kaggle実行、推論、提出はまだ行わない。

## 目的

- `k_segments`だけを変更したとき、保存済みK=16 OOFよりdirect RMSEが改善するかを測る。
- direct改善が小さくても、exp293 fixed deployable12へ追加したときのH512/whole-well oracle headroomが増えるかを測る。
- K=12/16/24の安定性を後続exp303で診断する価値があるかを、candidate novelty guardで先に判定する。

## 制約

- Route: `pf_beam`。
- 親: `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction`。
- 比較bank: `exp293_physics_only_candidate_bank_headroom_contract`の固定deployable12。
- 科学variantは`K=12`と`K=24`の2つだけ。保存済み`K=16`をcontrolとして再利用し、再生成しない。
- `theta0`、local-linear k/bandwidth/ridge、`smooth_rho`、`gate`、`field_min_proj`、`kbins`、
  `kappa_regimes`、rotation、ANCC theta bandwidth、GR correction、U projectionはexp226から固定する。
- exp226と同じ5 fold、well identity、score row、並び順を再利用する。
- candidate値、block assignment、feature/prediction contentをtruthなしでSHA freezeした後にだけtrue suffix TVTを接続する。
- HMMの`step`、`n_rates`を含む他パラメータ探索、blend、selector、decoder、inference、submissionは対象外。
- 再現性は`docs/06_reproducibility.md`に従い、gzipはdecompressed content SHAを主証拠とする。

## 実行量契約

- scientific variants: 2 (`K=12`, `K=24`)
- outer folds: 5
- variant-fold generation/evaluation: 10
- LightGBM configs / trained folds / boosters: `0 / 0 / 0`
- parent/control regeneration: 0
- GPU: 0、CPUのみ

## 受け入れ基準

- steeringのrequirements/design/tasklistに、変更1変数、固定値、入力、freeze順序、direct/candidate noveltyの
  PASS/FAIL、禁止事項が明記されている。
- exp302の`config.yaml`、`README.md`、`SESSION_NOTES.md`、`result.md`、`metrics.json`が
  `design_complete_not_implemented`を示す。
- direct PASSは、少なくとも1 variantでpooled RMSE `<=9.3771096741`、4/5 folds改善、
  1000+とhidden-like 2面がK16比`+0.02 ft`以内、by-well p95/worstが`+0.25 ft`以内をすべて満たす。
- candidate novelty PASSは、少なくとも1 variantでexp293 fixed12へのadd-one H512 oracle RMSE改善
  `>=0.03 ft`、whole-well改善`>=0.02 ft`、H512 strict unique-best比率`>=2%`、4/5 folds改善を満たす。
- direct PASSだけではexp303を開始しない。exp303の先行条件はcandidate novelty PASSとする。
- 両guard FAIL時はK値や他パラメータの救済gridを行わずbranchを閉じる。
- `make validate-exp EXP=exp302_exp226_multiscale_k_segment_candidate_audit`と`make validate-template`が通る。
- 実装source、Jupytext候補、Kaggle package、artifact、prediction、submissionを新規作成していない。

## 2026-07-20 実装承認追記

上記は設計確定時の依頼と受け入れ条件であり、同日のユーザー依頼`exp302を実装してください`により、
実装source、別名Jupytext候補、専用testの作成だけを追加承認した。正規Notebook採用、Kaggle package、
push/run、inference、submissionは引き続き未承認とする。
