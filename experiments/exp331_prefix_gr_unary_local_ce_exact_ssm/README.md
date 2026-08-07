# exp331_prefix_gr_unary_local_ce_exact_ssm

## 状態

- ルート: ensemble
- 状態: Stage A科学gate FAIL・branch closed
- fold 0 CV: `24.760360`
- Public LB / Private LB: 未提出 / 未提出
- 作成日: 2026-07-21
- 親実験: `exp295_prefix_anchored_wholewell_gr_alignment_ssm`

## 仮説

exp295のtimeout原因だった学習中のexact structured DPを外し、各rowの正解Type Well TVT stateをlocal CEで独立に学習する。exact SSMはmodel freeze後だけ使い、row unaryを物理的に連続なTVT posteriorへ整える。

## 変更点

- exp295のinput、encoder、prefix conditioning、fixed exp209 decoderを維持した。
- trainingとearly stoppingからstructured DPを外し、hard nearest-state local CEだけを使った。
- model freeze後にreal GR、circular shuffle、geometry-onlyを同一modelでdecodeした。

## 検証方針

complete-well GroupKFoldのfold 0 official hidden suffix 780,457 rows / 155 wellsで、exp209、shuffle、geometry-only、long-distance、hidden-like、well p95/worstを比較した。outer-valid truthはprediction freeze後にだけ結合し、Stage A全科学gate PASS時だけStage Bへ進む契約とした。

## 実行結果

Stage 0は保守的fold外挿`4.516839 h`、peak`1.924052 GB`でcompute gateをPASSした。Stage Aはfold 0、seed 42の1 neural modelをKaggle T4 version 1で完走した。

| メトリック | 値 |
| --- | ---: |
| real GR RMSE | 24.760360 |
| exp209 RMSE | 12.671087 |
| geometry-only RMSE | 32.465002 |
| circular-shuffle RMSE | 57.878820 |
| real / exp209 well RMSE p95 | 44.560719 / 26.301518 |
| worst-well regression vs exp209 | +63.109520 ft |
| runtime / peak GPU | 4.115497 h / 1.889884 GB |

GR attribution、truth-freeze、runtime/memoryはPASSしたが、exp209比較、well p95、worst-wellの3条件をFAILした。155 wells中138 wellsでexp209より悪化しており、local CE-only unaryは実用的なglobal path品質へ届かなかった。

## 所見

総合判定は`stage_a_failed_branch_closed`。事前契約どおりexp331内のrescue gridを行わず、Stage B、推論、提出を閉じる。詳細、gate、SHAは`result.md`と`SESSION_NOTES.md`を参照する。exp332は別承認なしに着手しない。

## 次

exp331で追加実行は行わない。代替設計exp332は先行条件だけ成立したdesign-only候補として保持し、別のユーザー判断がある場合にのみ着手する。
