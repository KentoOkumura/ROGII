---
title: visible tailの候補別・well別誤差監査
date: 2026-06-16
types:
  - oof_analysis
  - comparison
experiments:
  - exp026
  - exp027
  - exp039
  - exp054
  - exp063
  - exp069
  - exp070
  - exp073
topics:
  - error_analysis
  - tail
  - candidate_path
status: final
summary: "visible sampleで候補予測を比較し、global blendではなくwell geometryに応じた限定的な分岐だけを後続候補とした。"
---

# visible tailの候補別・well別誤差監査

- 対応する上位仮説: なし

作成日: 2026-06-16

## 結論

- visible sampleでは`exp027`と`exp039`がほぼ完全一致するため、このスコアだけを根拠にhidden test向けの候補を混ぜない。
- non-copy候補では`exp070`が全体RMSE 4.341515、`exp073`が4.382939、`exp063`が4.533153だった。ただし`exp070`の改善は最大wellの`00bbac68`とdistance `1000+`に集中した。
- `000d7d20`と`00e12e8b`では`exp063`が`exp073`より良く、単一候補のglobal replacementを支持しない。
- 後続で使う場合は、`exp073`、`exp070`、必要なら`exp063`をwell geometryに応じて限定的に分岐する。global blendとglobal replacementは行わない。

## 対象と証拠範囲

- anchor: `exp027`および`exp073`
- 比較対象: `exp026`、`exp027`、`exp039`、`exp054`、`exp063`、`exp069`、`exp070`、`exp073`
- 評価単位: visible sampleの全体、well、tail開始からのdistance bucket
- `exp050`はローカルに`submission.csv`がなかったため対象外。
- visible sampleは公開train由来であり、この監査からhidden testの改善は判断できない。

## 全体結果

| Candidate | exp027 anchorに対するRMSE | 解釈 |
| --- | ---: | --- |
| exp027 | 0.005251 | visible sampleのほぼ完全なreplay |
| exp039 | 0.005251 | exp027と同じvisible出力 |
| exp070 | 4.341515 | non-copy候補の最良だが1 wellへの依存が強い |
| exp073 | 4.382939 | 再現可能なML比較基準 |
| exp063 | 4.533153 | 2 wellではexp073より良い |
| exp054 | 6.097466 | exp063/070/073より悪い |
| exp026 | 8.097674 | visible pseudo-tail基準として悪い |
| exp069 | 12.851383 | PF/Beam直接出力は支持されない |

`exp073`をanchorにした場合、`exp070`の差は-0.041424、`exp063`は+0.150214だった。差の平均だけではwell間の反転を表せない。

## well・distance bucket別結果

| Well | 最良non-copy候補 | RMSE | 次点 | RMSE |
| --- | --- | ---: | --- | ---: |
| `000d7d20` | exp063 | 2.207111 | exp073 | 2.244715 |
| `00bbac68` | exp070 | 4.860883 | exp073 | 5.944230 |
| `00e12e8b` | exp063 | 2.797817 | exp073 | 3.050241 |

distance `1000+`では`exp070`が4.497412で、`exp073`の4.804359を上回った。一方、短距離・中距離や他wellでは勝者が異なり、固定平均を選ぶ根拠にはならない。

## 関連ファイル

- 生の表: [`studies/metric_weighted_tail_error_map/`](../../studies/metric_weighted_tail_error_map/)
- exp073 anchorの生の表: [`studies/metric_weighted_tail_error_map_anchor_exp073/`](../../studies/metric_weighted_tail_error_map_anchor_exp073/)
- 生成スクリプト: [`scripts/metric_weighted_tail_error_map.py`](../../scripts/metric_weighted_tail_error_map.py)

## 次のアクション

guarded blendを検証する場合は、候補と分岐条件を事前に固定し、`00bbac68`に似たlong-tail geometryだけを対象にする。visible sampleの結果だけでLB改善を主張しない。
