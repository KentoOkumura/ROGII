# exp509_exp413_strict_public_core_final_slot 結果

## 状態

prediction-only候補実装済み、正規notebook未採用、未実行、未提出。現時点の性能数値は
上流参照値でありexp509の新規結果ではない。

## 仮説

exp497の事前固定median weightでstrict public-coreを小さく加えれば、exp413単独を避けながら
公開系trajectoryの相補性を保守的に残せる。

## 固定設定

- 親: `exp413_scale5_likpf_full_replacement_on_exp335`
- auxiliary: `exp497_strict_public_core_fold_safe_ensemble_on_exp413`
- formula: `0.8628352666928758 * exp413 + 0.13716473330712417 * strict_public_core`
- 新規CV/model/PF/Beam/GPU: すべて0
- Gold/contact/router/final postprocess: すべて禁止

## 結果

| メトリック | 値 |
| --- | --- |
| exp509 CV | 未実行 |
| Public LB | 未提出 |
| Private LB | なし |
| technical gate | 未評価 |

## 実装結果

- `exp509_exp413_strict_public_core_final_slot_compact_selfcontained_inference.py`: 694行、8章。
- 保存exp497 booster 40 + Ridge 2、保存exp413 booster 75を読む。新規fitは0。
- dynamic exp413の中間`submission.csv`をfinal名から隔離した。
- strict public-core、exp413、既存blendのvisible parityを別契約に分離した。
- exp509 finalはfloat64固定式で再構成し、component CSVの読み戻し、ID順序、finite、formula、
  SHAを全ANDで監査する。
- 専用test `6 passed`、依存exp497 test `30 passed`、Jupytext/構文/Ruff/strict validatorはPASS。

## 解釈

exp497の科学的promotion gateはFAILのままであり、本実験は最終提出portfolioの第1枠候補を
明示するreference overrideである。実装は完了したがKaggle runtimeの入力SHA、component parity、
prediction/submission SHAは未取得なので、まだ利用可能とは判断しない。

## 次

別承認があれば正規notebook採用とKaggle package readbackへ進む。Kaggle runと外部提出は
それぞれ別承認のままとする。
