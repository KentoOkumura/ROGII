# 設計

## アプローチ

exp413 Stage D の TVT 学習面を control とし、selector 由来ブロックだけを交換する。
exp501 の compact は nested outer-train 行では inner OOF score、outer-valid 行では 4 inner
model ensemble から作られているため、その保存済み 25 partitions を fold-safe な入力として
再利用する。selector 自体は再学習しない。

| ブロック | exp413 control | exp502 treatment | 扱い |
| --- | ---: | ---: | --- |
| clean base | 273 | 273 | exp413 の列と順序を固定 |
| nested selector compact | exp413 74 | exp501 77 | 74 を除去して 77 へ置換 |
| signed selector compact | exp413 23 | exp413 23 | 固定 |
| final | 370 | 373 | 置換に伴う +3 のみ |

これは add-only ではない。exp413 nested74 と exp501 compact77 の併存、両者の平均、
selector 出力を用いた gate、weight fitting は行わない。exp501 compact77 は exp264 fixed12
candidate bankに exp490 HMMを追加した selector 面、残り2ブロックは exp413 scale5 面という
hybridである。この不一致は意図したアブレーションであり、変更変数を selector block に限定する。

## 実験範囲

- 対象実験: `exp502_exp501_fixed13_selector_replacement_on_exp413`
- Route: `ml_model`
- 親実験: `exp413_scale5_likpf_full_replacement_on_exp335`
- selector source: `exp501_exp490_mean_reverting_hmm_fixed13_selector_on_exp264`
- 変更する変数: final TVT feature matrix の nested selector compact block だけ。
- 固定する変数: clean273、signed23、target、score rows、outer fold、seed、TVT LightGBM
  config 0/1/2、GPU reproducibility mode、early stopping、postprocess、評価 scope。

## 入力契約

### exp413 control / 固定ブロック

- Stage D kernel: `kentookumura/exp413-scale5-likpf-downstream-train` version 2
- saved control OOF RMSE: `7.884802794404715`
- saved control OOF SHA: `9bd2d17778b3b27d771b12cbff72def8b87e6cdf14062e1c0ba192434cef4a9d`
- Stage D model manifest SHA:
  `4b4f988154468ba6697cdd57c0a0c6bf7cc631e7b2bbe1f15fa8f51fdeb7c3df`
- retained signed compact manifest SHA:
  `7a4282a25d7e7887e314cd3d01b8a09c81ff91ba3c1b1cf62e3197079ac93323`
- removed nested compact manifest SHA:
  `507429faa4fbc336dbc00e8edfee5a45788b8a58dbc2e15a440d5e7780d5f07f`

### exp501 replacement block

- source kernel: `kentookumura/exp501-exp490-fixed13-selector-train` version 2
- selected input features: 92
- compact features: 77
- compact partitions / rows: 25 / 18,919,945
- selector model manifest SHA:
  `3adb894df634b929020693e7435852b646a99b8dafc2eab4fb88fc1c231cabbc`
- compact manifest SHA:
  `32317a715997c7a7e145d7122a8ac37733adb30710e571ccbf11a81c2d79c257`
- outer-valid candidate score SHA:
  `1641b9cb9d29aa65759d5a3cd637cef62f84ac50dd39999955b5c2aa1bd1599e`
- fold manifest SHA:
  `fa41084c5fcb4adffb88d44211b4cc5d2d2f46b5bd4d65828b6af941184b2a6d`

exp413 Stage C / Stage S と exp501 の fold manifest SHA は同一である。実装時は
`well, row_idx, downstream_outer_fold, inner_fold/partition role` を検査し、row/fold mismatch、
duplicate key、missing key を 0 に固定する。manifest一致だけで row join の正当性を代用しない。

## 学習・評価契約

- treatment: 1 variant
- TVT LightGBM configs: exp413 の `[0, 1, 2]`
- outer folds: 5
- planned GPU boosters: 15
- exp501 selector / exp413 signed selector / exp413 control retraining: 0 / 0 / 0
- HMM / PF / Beam rerun: 0 / 0 / 0
- primary delta: `exp502 RMSE - saved exp413 RMSE`

promotion は exp413 Stage D の late-stage ML gateを継承する。

- pooled RMSE gain `exp413 - exp502 >= 0.03 ft`
- nonworse folds `>= 3 / 5`
- `md_since_0_250`、`md_since_250_1000`、`md_since_1000_plus`、
  `hidden_like_spatial`、`hidden_like_typewell_purged` の各 delta `<= +0.02 ft`
- technical / leakage checks を全件 PASS
- by-well p95、worst well、`+1/+3/+5 ft`悪化well数は exp413 と同じく report-only。

PASS は同じ exp502 内の inference 実装を別承認候補にするだけで、自動実装・自動実行・
自動提出しない。FAIL は same-OOF rescue、selector blend、feature subset、weight/gate調整なしで閉じる。

## 実装境界

canonical notebook は template placeholder のままとする。2026-08-02の実装承認により、
exp413 compact self-contained Stage D を章立ての参照元にし、Jupytext percent形式の別名train
sourceと候補notebook、契約テストを実装した。正規 notebook の上書き・採用は別途確認する。

## 再現性設計

- seed policy: exp413 と同じ seed 42。LightGBM configごとの既存seed導出を変更しない。
- stochastic 処理: 新規要素は GPU LightGBM 15本だけ。保存済み selector/HMM/PF は再実行しない。
- PF/Beam / likelihood-PF / seed bagging: 実験内の新規実行は 0。
- 並列処理と乱数: `deterministic=true`、`force_col_wise=true`、固定 `n_jobs/num_threads`
  と exp413 の `gpu_repro_guard_dp_threads8` を継承し、global RNGを導入しない。
- CPU/GPU runtime: Kaggle T4を正とし、ローカルtrainは行わない。GPU bitwise一致は前提にしない。
- train cache: removed74、inserted77、retained273/23、final373のschema/order/content SHAを記録する。
- model / prediction: 15 model manifest、OOF prediction、fold/scope/by-well metricsのSHAを記録する。
- inference / submission: 現在は範囲外。将来承認された場合はhidden test再生成とsubmission SHAを
  train-sideとは別に監査する。
- Kaggle package: push前に metadata と bootstrap内 configのvariant、source kernel version、
  feature count、run flag、GPU設定を展開して一致確認する。

## リスク

- リークリスク: exp501 compactのouter-train/outer-valid生成roleを崩すとstacking leakageになる。
- CV/LB不一致: exp413はCV `7.884802794` / Public LB `7.201`で、tailもreport-onlyだった。
  exp501もpooled改善とby-well tail悪化が併存するため、CV gainをrobust adoptionと同一視しない。
- 整合性リスク: exp501 selector blockとexp413 clean/signed blockはcandidate surfaceが異なる。
  これは仮説そのものだが、他の再計算を混ぜると1差分でなくなる。
- ランタイム/メモリ: final370から373へ小幅増だが、exp413 Stage Dは約17,386秒だった。
  15 boosterを超えるconfig展開、control再学習、selector再学習はfail closedする。
- 再現性リスク: GPU LightGBMはbitwise deterministicとみなさず、config/kernel/model/prediction SHAで
  実行単位を固定する。

## 次のアクション

train-side実装と静的検証まで完了した。正規notebook採用とKaggle package/runはさらに
別承認とする。
