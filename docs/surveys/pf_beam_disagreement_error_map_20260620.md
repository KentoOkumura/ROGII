---
title: PF・Beam disagreementとwell別誤差の監査
date: 2026-06-20
types:
  - oof_analysis
  - comparison
experiments:
  - exp073
  - exp083
topics:
  - error_analysis
  - pf_beam
  - candidate_path
status: final
summary: "PF・Beam・MLを773 wellsで比較し、PF直接置換を否定する一方、disagreementを候補選択やsample weightの診断材料として残した。"
---

# PF・Beam disagreementとwell別誤差の監査

- 対応する上位仮説: なし

作成日: 2026-06-20

## 結論

- 3,783,989 rows、773 wellsのpooled RMSEはPF 14.493061、Beam 15.774328、ML 9.526375で、PFまたはBeamによるMLの直接置換を支持しない。
- 一方でPFがMLに勝つwellは234/773、3候補中PFが最良のwellは207/773あり、候補としての多様性は残る。
- PF/Beam disagreementが小さくLikPF deltaも小さいwellではtruth TVTに近い傾向がある。ただしhigh disagreementでもPFが大きく勝つwellがあるため、単純な閾値によるhard routerにはしない。
- disagreement、confidence、tail lengthは、prefix backtest、sample weight、候補coverage確認後のrankerで使う診断量として扱う。

## 対象と証拠範囲

- PF/Beam入力: `exp083`のwell summaryとplot manifest
- ML入力: `exp073` train v2のby-well OOF metrics
- 評価単位: 全体、well、PF/Beam disagreement・confidence・tail length bucket
- 本監査は固定routerやselectorを学習・評価したものではない。

## 全体結果

| Metric | Value |
| --- | ---: |
| wells | 773 |
| rows | 3,783,989 |
| PF pooled RMSE | 14.493061 |
| Beam pooled RMSE | 15.774328 |
| ML pooled RMSE | 9.526375 |
| PF minus ML RMSE | 4.966687 |
| PF minus Beam RMSE | -1.281266 |

## 関連ファイル

- 生の表: [`studies/pf_beam_disagreement_error_map/`](../../studies/pf_beam_disagreement_error_map/)
- 生成スクリプト: [`scripts/pf_beam_disagreement_error_map.py`](../../scripts/pf_beam_disagreement_error_map.py)
- 後続の統合分析: [`hmm_pf_exp226_well_pattern_readout_20260712.md`](hmm_pf_exp226_well_pattern_readout_20260712.md)

## 次のアクション

PF/Beam disagreementを使う場合は、hard routerではなく、target-freeなprefix backtest、sample weight、候補rankerの入力としてfold-safeに検証する。
