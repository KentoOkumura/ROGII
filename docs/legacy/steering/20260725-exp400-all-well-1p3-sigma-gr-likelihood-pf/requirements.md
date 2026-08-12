# 要件

## 依頼

Kaggle discussion 728712で共有された、公開Notebookのlikelihood-PFにおける
GR観測ノイズ幅 `gs` の約1.3倍化を、本リポジトリの決定論的exp072
likelihood-PFへ全well一律に適用する実験を設計する。

このturnではbacklog、steering、実験ディレクトリと設計文書だけを作成する。
Notebook、Jupytext source、helper、test、Kaggle package、PF実行、推論、提出は
実装・実行しない。

## 制約

- 対象実験は `exp400_all_well_1p3_sigma_gr_likelihood_pf`、Routeは `pf_beam`、
  scientific parentは `exp072_exp063_full_replay_feature_cache` に固定する。
- 公開Notebook全体のscore再現ではなく、exp072の `lik_pf` と同型の
  128-seed likelihood-PFに対する1変数ablationとする。
- 変更する変数は、known prefixから計算したGR residual scaleに掛ける
  well-level係数だけとする。
- base scaleはexp072どおり
  `clip(nanstd(fillna(horizontal_GR, 0) - typewell_GR_at_TVT_input), 10, 60)`
  とする。候補はclip後に `1.3` を正確に1回掛け、再clipしない。
- 全773 train wellsを候補生成対象とし、well selector、row selector、
  missingness gate、error gateは使わない。
- 500 particles、128 seeds、seed weighting scale `(3, 5, 8, 12)`、
  initial spread `4.5`、状態・遷移・likelihood・resampling・補間・出力集約は
  exp072から変更しない。
- primary candidateは `likpf_mean` とし、保存済みexp072
  `likpf_mean`をcontrolとして再実行しない。
- `likpf_scale_3/5/8/12` は同一PF実行から保存するsecondary diagnosticであり、
  実行後のbest scale選択やpromotion rescueには使わない。
- exp209保存Gaussian exact-HMMとの50:50 blendはfixed downstream
  non-regression guardにだけ使い、HMMを再実行しない。
- reporting foldはexp226の安全な5-fold identityを使う。model fit、
  LightGBM config、trained fold、booster、Beam、HMM、parent PF control再実行は
  すべて0とする。
- candidate predictionとlogical content SHAをunknown-suffix true TVT、
  error、fold別score、hidden-like roleの読込前にfreezeする。
- multiplier、clip、particle数、seed数、seed weighting scale、initial spread、
  transition、resampling、blend weightを実行結果後に変更して救済しない。
- 公開Notebookのselector、Beam、hold blend、Ridge、projection、contact guard、
  learned/model-package correctionは再現対象に含めない。
- inferenceとsubmissionはtrain-sideの全promotion gate PASS後も別承認まで無効とする。
- Kaggle Notebook実行を正とし、CPU / internet offで成立させる。
- `docs/06_reproducibility.md` に従い、per-well stable seed、固定並列数、
  input/prediction/content SHA、kernel versionを記録する設計にする。

## 受け入れ基準

- discussion由来の事実と、本実験で採用するlocal deterministic transferが区別されている。
- base `gs`、1.3倍の適用順、再clipなし、全well適用が一意に定義されている。
- exp072から固定するPFの粒子数、seed数、seed policy、scale readout、
  dynamics、likelihood、resampling、補間、primary outputが明記されている。
- 実行量が1 scientific variant / 773 PF well-runs / 98,944 seed-well trajectories /
  49,472,000 particle starts / 5 reporting folds / booster 0 /
  parent control再実行0で固定されている。
- overall、fold、raw-GR observed/missing、high-missing、1000+、
  hidden-like 2面、by-well tail、fixed HMM 50:50のpromotion gateが
  実行前に固定されている。
- 保存exp072 / exp209 / exp226 / exp115入力のSHAとrow identityを検証する方針がある。
- candidate生成はtarget TVTを読まず、predictionとSHAをtruth join前にfreezeする。
- `KAGGLE_DIRECTION.md` と `experiment_summary.md` にdesign-only状態が記録されている。
- experiment scaffoldの設定・文書検証が通り、実装source、test、
  Kaggle package、実行成果物が追加されていない。

## 2026-07-25 implementation-only追加承認

後続のユーザー指示「exp400を実装してください」により、次を追加で承認した。

- 別名のcompact self-contained train Jupytext source / Notebook候補
- submissionを生成しないfail-closed inference Jupytext source / Notebook候補
- exp072 x1.0 fixture parity、x1.3適用順、stable seed、truth-late、
  execution count、promotion gateの専用test
- config、steering、SESSION_NOTES、README、result、metrics、
  experiment summary、backlog状態のimplementation-complete更新

正規Notebook上書き、Kaggle package、push、PF実行、inference、
submissionはこの追加承認に含めない。

## 2026-07-25 実行承認と2026-07-26 terminal結果

後続のユーザー指示「実行してください」により、正規train Notebook採用、
Kaggle private CPU package、canonical kernelへのpush、train-side candidate
PF実行を追加承認した。inferenceとsubmissionは承認範囲外のままとした。

Kaggle version 1 / id_no `128585102`は計画どおり完走し、technical gateは
PASSした。primary candidate/control RMSEは
`12.221810980460939 / 11.594894395642696`で、candidateは
`0.6269165848182432 ft`悪化した。改善は1/5 folds、305/773 wellsに留まり、
fixed HMM 50:50と全required stress scope、by-well tailもFAILした。

受け入れ基準のFAIL時契約に従い、
`all_well_likelihood_pf_gs_x1p3_failed_close_without_rescue`として閉じる。
inference、submission、post-hoc best scale、adaptive multiplier、
version 2へ進めない。
