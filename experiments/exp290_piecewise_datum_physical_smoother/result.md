# exp290_piecewise_datum_physical_smoother 結果

## 状態

Kaggle CPU Stage 0 version 1完了。technical guardは全通過したがscientific guardは不通過となり、
固定failure policyどおりparameter/group救済なしでbranchを閉じた。Stage 1、inference、submissionは実行しない。

## 仮説

exp226 geometry上のpersistent datumだけをbounded piecewise stateとして推定し、known-prefix pseudo-cutで
well固有信頼度を決め、Type Well/近隣情報を不確実性事前に限定すれば、always-on residual-offset HMMの
中心誤差改善を残しながらcatastrophic wrong-offset tailを抑えられる。

## 設定

- 親: `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction`
- Route: `pf_beam`
- kernel: `kentookumura/exp290-piecewise-datum-physical-smoother-train` version 1、id_no `127881061`
- 実行規模: 1 Stage 0 contract、ML config / trained fold / booster `0 / 0 / 0`、control再生成0
- state: absolute correction `[-15,+15] ft`、0.5 ft step、minimum duration 256 rows
- observation: known-prefix residualとknown-prefix-calibrated robust Type Well GR likelihood
- hierarchy: well calibration + Type Well/neighbor scale/hazard/noise prior。datum mean/jump signは禁止
- solver: 61 states × 5 duration phasesのexact log-space forward-backward posterior mean
- Stage 0: 773 wells、512/256/128-row pseudo-cutの直後128行、合計296,832 rows
- oracle: 全粒度で禁止

## Stage 0結果

| 指標 | exp226 geometry | exp290 | 判定 |
| --- | ---: | ---: | --- |
| pseudo-tail RMSE | 1.436926 | 1.403407 | 改善0.033519 ft < 0.20、FAIL |
| `abs(base error)>=5` correction sign | - | 0.483111 | < 0.58、FAIL |
| 改善fold数 | - | 5/5 | >=4/5、PASS |
| well RMSE p95 | 2.437183 | 2.440010 | 非悪化条件、FAIL |

- 全5 foldsでRMSEは`0.027877～0.043636 ft`改善したが、効果量は固定guardに届かなかった。
- correction平均絶対値は0.144250 ft、最大1.258487 ftで、bound違反は0だった。
- posterior entropy平均は1.543384、reset probability平均は0.0だった。128-row windowがminimum duration 256 rows未満なので設計どおりresetは発生しない。
- technical coverageは296,832 / 296,832行、773 / 773 wells、3 / 3 windows、5 folds、finite prediction 1.0。truth-after-freezeも2,319 windowすべてPASSした。
- runtimeは587.042秒、peak RSS 1,215.824 MB、Kaggle 9時間上限への余裕は8.837時間だった。

## 判定

`scientific_guard_passed=false`。GR evidenceからbounded constant datumの弱いfold-stable改善は得られたが、
符号識別はchance未満で、tail p95も微悪化した。Stage 0が直接反証する識別性は十分でなく、duration/resetを
評価するStage 1へ進む根拠はない。grid、clip、pseudo-cut、group、neighbor、likelihoodの同一OOF救済は行わない。

## 再現性

- prediction decompressed SHA: `b0a598c82bb083b61a1c445752caf90e908899bb15afcbb45658f3baf9a3f956`
- prediction raw gzip SHA: `7312ca1e85ae3b1d4e0e03158ca16e6c9305e8780befb7b9540056f55a4b3a0a`
- state-space manifest SHA: `a004699db75078f0becba80c874e65e172eaf3d4f3c5f184a9dab3cc43c7b224`
- hyperprior content SHA: `ba69f0a69d11bab1ddc22d96bb5a06b5501acd5f49548e8b575cf91586d02b9a`
- pseudo-cut content SHA: `5dd42c2ee7abf1190ae0b5b70f49db9d0d27f5a86d8945b4ff8b6d532649caad`
- downloaded output: `/tmp/kaggle-output/exp290_piecewise_datum_physical_smoother/train_v1`
- raw/decompressed prediction SHAとstate manifest SHAは取得ファイルから再計算して一致確認した。
- RNGなしだがKaggle rerun parityは未実施のためdeterministic anchorはfalseのまま。

## 次

exp290 branchは終了する。Stage 1、raw-test inference、submission、parameter/group rescueは行わず、
本結果だけを根拠に新しい救済backlogも追加しない。
