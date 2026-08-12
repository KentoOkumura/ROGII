# 要件

## 依頼

次の2条件を、likelihood-PFのseed集約を`scale_5`へ固定して比較する
backlog、実験ディレクトリ、steeringを作り、実装前に設計を確定する。

- `gs × 1.0 + scale_5`
- `gs × 1.3 + scale_5`

初回承認範囲はdesign-onlyだった。2026-07-26の追加指示
「exp404を実装してください」により、別名compact self-contained source /
Notebook候補と専用testのimplementation-onlyを承認済みとした。続く
「実行してください」により正規Notebook採用、Kaggle private CPU package /
push / train実行も承認済みとする。inferenceとsubmissionは含まない。

## 制約

- Route: `pf_beam`
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- 科学parentはexp400、PF kernel parentはexp072 deterministic v2とする。
- `scale_5`を実行前にprimaryへ固定し、scale 3/8/12やtemperature gridを作らない。
- 保存exp072にはscale 5列がないため、x1.0とx1.3を同じsource・同じstable
  per-well seedで両方再生成する。
- 変更因子は`gs`倍率だけとし、base estimator、clip、particles、seeds、
  dynamics、resampling、補間、初期値、score rows、foldを固定する。
- x1.0とx1.3は同じseed labelを使う。倍率をseed keyへ含めない。
- x1.3でresampling時点以降の軌道が分岐することはtreatmentの一部とする。
- `pf_mean`はexp072/exp400とのtechnical parityにだけ使い、scientific primaryや
  候補選択に使わない。
- horizontal unknown suffixの`TVT`は両variantのpredictionとSHAをfreezeするまで
  読まない。
- 公開Notebook全体、ML、Beam、selector、hold、projection、calibration、
  model packageを再現しない。
- CPU-only、2 variants、model / booster / HMM / Beamは0とする。

## 受け入れ基準

- `exp404_scale5_sigma_gr_likelihood_pf_ablation`が採番され、route、
  lineage、2 variants、primary比較が一意に定義されている。
- `gs_base = clip(nanstd(fillna(prefix_GR, 0)-typewell_GR_at_TVT_input),10,60)`
  を固定し、x1.0は`[10,60]`、x1.3はbase clip後1回だけ乗算して`[13,78]`、
  両方post clipなしと定義されている。
- 500 particles / 128 seeds / likelihood temperature 5 / exp072 dynamics /
  common per-well stable seedが固定されている。
- 実行量が2 variants / 1,546 PF well-runs / 197,888 seed-well trajectories /
  98,944,000 particle starts / 5 reporting folds / booster 0で固定されている。
- primaryを`likpf_scale_5_x1p3`対`likpf_scale_5_x1p0`とし、gainの符号、
  overall、fold、GR missing、long-tail、hidden-like、by-well gateが
  実行前に固定されている。
- x1.0 meanのexp072 parityとx1.3 mean/scale5のexp400 parityはtechnical
  checkであり、科学判定ではないと明記されている。
- prediction生成とtruth joinの順序、入力SHA、prediction content SHA、
  artifact manifestの記録方針がある。
- deterministic anchor として扱う場合は、feature content SHA、prediction SHA、
  Kaggle kernel versionが記録されている。model / submissionは生成しないため
  SHA対象外とする。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
- `KAGGLE_DIRECTION.md`と`experiment_summary.md`に
  初回`design_frozen_unimplemented`、implementation-only完了後は
  `implementation_complete_not_run`として登録されている。
- scaffoldの設定・文書検証が通る。
- 正規Notebookを上書きせず、別名compact self-contained train / inference
  sourceとNotebook候補が存在する。
- 専用contract testがcommon seed、scale 5限定、parent kernel parity、
  truth-late、execution count、technical / scientific gateを検証する。
- implementation-only段階ではKaggle packageと実行生成物を追加せず、
  run設定はdisabledのままとする。後続の明示実行承認後はこの制約を解除する。
