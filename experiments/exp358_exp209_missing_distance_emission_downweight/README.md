# exp358 exp209 missing-distance emission downweight

## 状態

- Route: `pf_beam`
- 状態: Stage 1 scientific gate FAIL、rescueなしでclosed
- 優先度: closed
- 親: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- 履歴参照: `exp308_imputed_gr_confidence_downweight`
- Kaggle: private CPU version 2、id_no `128528105`
- inference / submission: 未実装・未実行

## 仮説

exp209の観測sigmaを変えず、raw GR欠損rowの補間evidenceだけを欠損距離に応じて
弱めれば、long-gapで補間値を過信する問題を単一変更として改善できるかを検証した。

## 変更点

- failed exp307 finite-MAD observationへの依存を削除し、trusted exp209へ直接接続。
- exp209から変更したのはraw-missing rowのGaussian log-emission multiplierだけ。
- observed rowはweight 1、missing rowは
  `max(0.25, 2^(-nearest-finite-row-distance/8))`。
- saved exp209 HMM / exp072 LikPFをcontrolとして読み、親は再実行していない。

## 検証方針

Stage 0でtruth-free weight surfaceを監査し、別承認されたStage 1では
candidate predictionをtruth結合前にfreezeした。exp209比0.05 ft、4/5 folds、
1000+、hidden-like 2面、by-well p95/worst、fixed LikPF 50:50を
事前登録AND gateとして評価した。

## 実行

- Stage 0: 0-HMM technical audit、23/23 checks PASS
- Stage 1: 1 variant / 5 reporting folds / 773 exact-HMM well-runs
- model / booster / PF / Beam / parent-control rerun: 0
- runtime: `17,475.557881 sec`

正規train Notebook:
`exp358_exp209_missing_distance_emission_downweight_train.ipynb`

Jupytext Stage 1 source:
`exp358_exp209_missing_distance_emission_downweight_stage1_compact_selfcontained_train.py`

fail-closed inference候補:
`exp358_exp209_missing_distance_emission_downweight_compact_selfcontained_inference.ipynb`

## 結果

- direct candidate / exp209 RMSE: `12.012570 / 11.938287`
- improvement: `-0.074283 ft`
- improved folds: `0/5`
- MD since 1000+ improvement: `-0.082776 ft`
- hidden-like spatial / typewell-purged:
  `-0.224970 / -0.229587 ft`
- by-well p95 delta: `+0.469370 ft`
- worst-well delta: `+6.630365 ft`
- fixed LikPF 50:50 delta: `+0.036981 ft`

事前登録したscientific gateを明確にFAILした。
formal technical gateもpost-CSVのbit-exact weight比較1項目でFAILしたが、
最大差は`5.551e-17`で、生成・emission適用契約の逸脱ではない。

## 所見

`missing_distance_exp209_failed_close_without_rescue`。
half-life/floor grid、hard mask、sigma/transition/prior変更、blend rescueは行わず、
inferenceとsubmissionへ進まない。詳細は`result.md`と`SESSION_NOTES.md`を参照。
