# exp420_exp226_hmm_guided_defensive_mixture_pf 結果

## 状態

train-side実装完了。exp411 schedule prerequisite FAILのため未実行のまま停止。

## 仮説

exp226 geometry rateとuntreated HMM rate-innovation scheduleを、元transition 50%を残す
importance-corrected defensive proposalへ統合すると、absolute path driftを継承せずに
PFのfinite supportとrate-lag failureを減らせる。

## 設定

- Route: `pf_beam`
- 実装lineage parent: `exp419_exp226_guided_defensive_mixture_pf`
- scientific control: 保存済みexp404 `likpf_scale_5_x1p0`
- candidate: `exp226_hmm_guided_defensive_mixture_scale5` 1 variant
- proposal:
  inactiveはoriginal `0.5` + geometry `1x/4x/16x`各`1/6`、
  activeはoriginal `0.5` + geometry 3成分各`1/12` +
  HMM方向3成分各`1/12`
- correction: `p0/q`、clipなし、構成上上限2
- HMM schedule:
  untreated forward innovation、CUSUM `0.01 / 1.0 / 32 / 128`
- PF:
  500 particles ×128 seeds、temperature-5 full-suffix evidence weighting
- 検証:
  fixed44 mechanism preflight後、別承認で773-well full OOF
- メトリック:
  suffix-row RMSE、fold / scope / episode SSE / support / well-tail
- seed:
  stable SHA256 per well + seed index
- model / booster / GPU:
  0

## 結果

| メトリック | 値 |
| --- | --- |
| CV | - |
| Public LB | - |
| Private LB | - |

untreated HMM schedule、scheduled PF、fixed44 / full orchestration、
truth-late readout、fail-close gateのcompact self-contained候補と専用testを実装した。
候補predictionとscientific metricはまだ存在しない。exp408 / exp410 / exp419 /
exp411の値は設計・実装根拠であり、exp420の実験結果ではない。

実装検証はexp420専用`13 passed`、exp419 / notebook / scaffoldを含む対象検証
`36 passed`、Jupytext test、`py_compile`、Ruff F821、`make validate-exp`
strict validationをPASSした。all-guidance-zero exp404 parityとHMM-weight-zero
exp419 parityはsynthetic fixtureでbitwise一致し、inactive / activeの
importance ratio上限2と7 proposal成分の利用を確認した。

## 再現性

- deterministic anchor: false
- seed policy:
  `sha256("likpf::train::<well_id>") % 2147483647 + 1 + seed_index`
- kernel version: 未実行
- code / config / scientific contract SHA: 実装時に生成可能
- input / schedule / prediction / diagnostic SHA: Kaggle未実行のため未生成
- model SHA / manifest SHA: 非該当
- submission SHA: 非該当
- rerun result: 未実施

## 解釈

3モデルのprediction blendではなく、geometry rate、forward rate innovation、
continuous particle supportという長所だけをproposalへ統合する実装を完了した。
元transition 50%とimportance correctionにより、無限粒子極限のtarget posteriorは
元PFと同じである。

しかし、同一CUSUM schedule / fixed32を使うexp411 Stage 0はfuture-rate方向一致
`0.225397`、passing folds `0 / 5`、control active-row fraction `0.136119`、
persistent-control active-well差`0.0`でFAILした。exp420のStage 0も同じ
direction / control gateを要求するため、PFを実行する前にprerequisite不成立が
確定した。exp420自体のPF scientific metricは未生成であり、PF候補やanchorとして
扱わない。

## 次

現行契約の正規Notebook採用、package、push、fixed44 Stage 0を行わない。
実装は参照として保持する。再開する場合はexp411 scheduleをそのまま使わない
独立仮説として別実験を設計し、full OOF、inference、submissionへは進めない。
