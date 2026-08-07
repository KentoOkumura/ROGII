# exp276_exp264_compact_tail_risk_target_free_gate_audit

> **旧結果無効:** 親exp264 Stage C compact / Stage D add-only OOFがfeature availability leakageで無効。

## 状態

- ルート: `ml_model`
- 状態: corrected-parent Kaggle CPU version 3完了・固定guard FAIL・branch closed
- CV/LB: anchor更新なし / submissionなし
- 親実験: `exp264_exp263_candidate_confidence_dual_selector`
- kernel: `kentookumura/exp276-target-free-tail-risk-gate-audit-train` version 3 / id_no `127735777`

## 仮説

exp264 Stage C compactのscore dispersion、candidate divergence、top1-anchor差、confidence
coverageとraw geometry/contextをwell prefix/序盤/全評価区間へ集約すれば、corrected Stage D v3の
255悪化well、特に220 over-0.25 wellsをouter-validの正解を使わずに事前識別できる。

## 実装

- Stage C 25 partitions / 18,919,945 rowsとStage D OOF 3,783,989 rowsを期待SHAで固定した。
- 先頭128、先頭512、fullの166 target-free featuresを5 familyへ等重み集約した。
- 各downstream outer foldのtrain 4 partitionsだけでempirical rankとq70/q80/q90を決めた。
- risk score凍結後にだけStage D outcomeをjoinし、risk lift、recall、gated RMSE、worst-wellを評価した。
- 1 audit variant / LightGBM 0 config / trained fold 0 / booster 0 / control再学習0。

## 検証方針

- downstream outer 5 foldsをwell groupで固定し、各foldのouter-train 4 partitionsだけでrisk分布をfitする。
- outer-validのtrue TVT/error/deltaはrisk scoreとq70/q80/q90を凍結した後のreadoutにだけ使う。
- 全quantileで両bad定義のpositive lift 5/5、gated control改善5/5、改善保持50%以上、worst-well
  +0.25 ft以下を同時に要求する。

## 結果（無効実行の再現用。性能判断には使用禁止）

| Gate | risk wells | `delta>0` lift | `delta>0.25` lift | gated RMSE | 改善保持 | worst-well delta | guard |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| q70 | 223 | 0.768738 | 0.826640 | 8.461023 | 11.43% | +10.587659 | FAIL |
| q80 | 144 | 0.842656 | 0.965801 | 8.352745 | 26.06% | +10.587659 | FAIL |
| q90 | 75 | 0.947588 | 1.019909 | 8.196329 | 47.20% | +10.587659 | FAIL |

matched control 8.545568、selector compact add-only 7.805644などの数値は、無効実行の再現目的でのみ
残す。親OOFが本番入力条件を再現していないため、改善量、worst-well、positive lift、fixed gateの
成否を示す値ではない。

## 実行メモ

- version 1はhive親directoryのpartition推論とparquet内physical列の型衝突で技術エラー終了した。
- `ParquetFile.read`へ修正し回帰テストを追加したversion 2が123.079秒で完走した。
- targeted 7 tests、repository 146 tests、py_compile、F821/E9、Jupytext、strict validationはPASS。
- 詳細は`result.md`、`metrics.json`、`SESSION_NOTES.md`、`kaggle/output/train_v2/artifacts/`を参照する。

## 所見

旧version 2ではq70/q80/q90を事後選択せず、feature/weight/threshold gridで救済しなかった。
ただし旧判定はfeature availability leakageで無効。corrected-parent再検証でも同じ打ち切り規則を維持し、
current-test inferenceとsubmissionはdisabledのままとする。

## Corrected-parent再検証

- Stage C v6: 25 partitions / 18,919,945 rows / compact 74列。
- Stage D v3: 3,783,989 rows / 773 wells / 255悪化 / 220 over-0.25 wells。
- 1 fixed audit / 5 evaluation folds / 0 model / 0 trained fold / 0 booster / CPU。
- 166 target-free features、5 family等重み、prefix128/early512/full、q70/q80/q90、guardは旧事前契約を維持。
- version 3は104.017秒で完了し、technical guardと12生成物はPASS。
- q70/q80/q90の固定guardはすべてFAIL。q90は改善保持59.77%だが、positive liftは
  `delta>0` 2/5 folds、`delta>0.25` 4/5 foldsで、worst-well `+13.4413 ft`を抑えられなかった。
- 事前規則どおりgrid救済、inference、submissionは行わず、exp303の開始条件を充足する。
