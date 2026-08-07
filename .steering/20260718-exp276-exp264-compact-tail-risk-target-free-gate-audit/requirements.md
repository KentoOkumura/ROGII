# 要件

## 依頼

feature availability leakageで無効化された旧exp276を、同じ実験ID
`exp276_exp264_compact_tail_risk_target_free_gate_audit` のまま再検証する。
固定済みtarget-free risk contractは変更せず、入力だけをhidden-safeなcorrected
exp264 Stage C v6 / Stage D v3へ差し替える。

## 制約

- Route: `ml_model`。PF/Beam候補はselector compactの補助meta featureであり、監査対象の最終予測は
  downstream LightGBMが生成するcorrected exp264 ML anchorである。
- exp264 Stage C の25 compact partitionとStage D保存OOFを再生成せず、追加学習0、LightGBM config 0、fold学習0、booster 0とする。
- Stage Cの各downstream outer foldについて、outer-train 4 partitionだけからrisk rankとquantile thresholdを決め、outer-valid 1 partitionへ適用する。
- true TVT、actual error、Stage D by-well delta、worst-well IDをrisk feature、方向、重み、thresholdへ使わない。
- risk featureはraw-testで生成可能なcompact score dispersion、candidate range/std、top1-anchor差、confidence/availability、raw geometry/contextに限定する。
- 監査scopeは親exp264のshape windowを根拠に先頭128行、先頭512行、全評価区間へ固定する。
- outer-train target-free分布の`q70/q80/q90`をすべて事前固定readoutとし、結果を見て有利なquantileだけを選ばない。
- GPU再学習、candidate追加、weight grid、current-test inference、competition submit、Stage D guardの事後緩和は禁止する。
- Kaggle CPU Notebookを最初のフル実行先とし、ローカルnotebook実行は行わない。
- 再現性は`docs/06_reproducibility.md`に従い、入力byte SHA、partition SHA、risk schema/content SHA、OOF prediction SHA、出力SHAを記録する。

## 受け入れ基準

- Jupytext percent形式のself-contained train/inference sourceと、対応する正規`.ipynb`がある。
- notebook上で入力/SHA確認、25 partition集約、outer-fold risk構築、Stage D label readout、gate metric、生成物保存を追える。
- 1 risk scoreあたり5 familyを等重みとし、各family内はouter-train empirical percentile rankの等重みに固定する。
- corrected Stage D v3の255悪化well、220 over-0.25 wellを入力検算し、fold別の悪化率、lift、recall、
  改善保持率、fallback後worst-wellを保存する。
- target/error列をrisk builderへ渡せないAPIと、outer-validをthreshold fitから除外するhard checkがある。
- 1 variant / 0 model config / 0 trained fold / 0 booster、parent/control再学習0を`SESSION_NOTES.md`に記録する。
- 構文、F821、Jupytext round-trip、実験validation、targeted testsが通る。
- corrected-parent再検証の実装・Kaggle CPU実行・結果記録を同じexp276へ追記し、旧version 2の
  数値は無効履歴として隔離したままにする。

## 2026-07-21 再検証承認

- ユーザーがcorrected-parent再検証を先に進めることを明示承認した。
- 実行量は1 audit variant / 5 evaluation folds / LightGBM config 0 / trained fold 0 /
  booster 0 / parent-control再学習0。
- Kaggle private CPUの同一canonical kernelへversion追加で実行する。inferenceとsubmissionは含まない。
