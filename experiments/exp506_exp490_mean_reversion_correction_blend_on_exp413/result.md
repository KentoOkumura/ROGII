# exp506_exp490_mean_reversion_correction_blend_on_exp413 結果

## 状態

Stage A version 2はKaggle private CPUで完了し、事前登録したprimary all-AND gateをFAILした。
`FAIL_CLOSE_WITHOUT_WEIGHT_SCOPE_COMPONENT_OR_GATE_RESCUE`として終端し、exp413をselected predictionに維持する。

## 仮説と設定

- anchor: exp413 Stage D保存OOF、RMSE `7.884802794404715`
- primary: `anchor + lambda * (exp490 - exp357)`
- lambda: other-four-fold closed-form fit、`[0.00, 0.10]`、interceptなし
- validation: outer 5 / meta 5、suffix-row unweighted RMSE
- model / booster / HMM / PF / Beam / GPU: `0 / 0 / 0 / 0 / 0 / 0`

## 結果

| メトリック | 値 |
| --- | ---: |
| anchor CV | `7.884802794404715` |
| primary CV | `7.902068462119896` |
| anchor比gain | `-0.017265667715181 ft` |
| nonworse folds | `3 / 5` |
| nonworse fixed scopes | `0 / 5` |
| by-well delta p95 / worst | `+0.054729023 / +1.816049513 ft` |
| lambda | `[0, 0.041578388, 0, 0.004513714, 0]` |
| deployment lambda | `0.0` |
| primary gate | `FAIL` |

fold 0 / 2 / 4はlambdaが0でanchor同値、fold 1は`+0.079086327 ft`、fold 3は
`+0.002097212 ft`悪化した。MD 3面とhidden-like 2面も全て悪化した。technical / leakage / SHA
checksは全PASSしたが、pooled gain、5/5 folds、全scope、worst-well、5係数positiveをFAILした。

report-only convex controlはCV `7.7345312772318815`だったが、5係数すべて上限`0.10`へ張り付き、
事前契約上`selectable=false`かつprimary救済禁止である。この値をexp506の採用や推論判断には使わない。

## 実行履歴

- kernel: `kentookumura/exp506-exp490-mean-revert-correction-exp413-train`
- id_no / terminal version: `129631767 / 2`
- runtime: private CPU、internet off、`294.943 sec`
- version 1: 科学計算後のmetrics表示でNumPy bool serialization ERROR
- version 2: 既存`to_jsonable()`を表示にも適用し、科学契約を変えずCOMPLETE
- output: `kaggle/output/stage_a_v2/artifacts/`

## 再現性

- input: `3,783,989 rows / 773 wells`、duplicate / missing / extra / suffix / MD mismatchすべて0
- anchor file SHA: `9bd2d177...cef4a9d`
- primary prediction logical SHA: `083f379c...da725d`
- primary OOF file SHA: `d459963d...5e1098`
- primary gate SHA: `047c13ac...a1f4f6`
- reproducibility manifest SHA: `11dde33a...55050`
- deterministic anchor: false（独立再実行一致は未確認。gate FAILのためanchor化しない）
- inference / submission: 生成なし

## 解釈

`exp490-exp357`補正はexp413残差に対してfold間で符号が安定せず、5係数中3つが非負制約の下限0になった。
補正が有効だったfoldでも全体ではanchorを悪化させ、全固定scopeも同方向に悪化したため、exp490の親比改善を
このincremental correctionとしてexp413へ移植する仮説は棄却する。weight上限、component、scope、gateを
同じOOFで変更せず、inference / submissionへ進まない。

## 次

現行のP0/P1であるexp509/exp510を優先する。exp506から追加するのは低優先の保存生成物readoutだけとし、
report-only固定10% convex controlのscope / tail改善がどこから生じたかを説明する場合も、exp506 gateの再評価、
weight再fit、推論候補化は禁止する。
