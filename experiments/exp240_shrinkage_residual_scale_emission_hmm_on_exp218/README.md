# exp240_shrinkage_residual_scale_emission_hmm_on_exp218

## 状態

- Route: `ensemble`
- 親: `exp234_crossfitted_residual_scale_emission_hmm_on_exp218`
- 状態: closed（train-side有限ablation完了、追加実行・推論・提出なし）
- 推論 / 提出: 無効

## 仮説

exp234 の cross-fitted row-wise sigma は exp218 point OOF を改善したが、異なる center を使う
exp221 fixed-sigma HMM に届かなかった。同じ exp218 center の scalar `sigma=20` を対照に置き、
row-wise residual scale の分散を fixed sigma へ縮小すれば、scale ranking を一部残しながら
HMM emission の過度な強弱を抑えられる可能性がある。

## 親実験との差分

exp234 は exp218 center に cross-fitted sigma をそのまま入れた alpha `1.0` 相当だった。
exp240 は同じ center の scalar `sigma=20` を新しい対照とし、その対照完了後だけ
alpha `0.25 / 0.50` の variance shrinkage を単独実行する。HMM dynamics と lambda は変えない。

## 実行順序

1. `scalar_control`: exp218 center、`sigma=20`、`lambda=0.50`。scale fit なし。
2. scalar 結果を記録後、必要なら `shrinkage_alpha025` を単独実行。
3. alpha 0.25 の結果が支持する場合だけ `shrinkage_alpha050` を検討。

縮小式は `sqrt((1-alpha)*20^2 + alpha*sigma_cf^2)`。複数 stage の同時 enable、
未登録 alpha、scalar 未完了での shrinkage は fail-fast する。

## 検証方針

scalar control と各 shrinkage stage を別 Kaggle version で 1 本ずつ実行する。overall RMSE
だけでなく distance bucket、exp115 hidden-like 2 subgroup、by-well 最大悪化、step-delta を
同じ exp218 OOF row 集合で比較する。shrinkage の採用基準は exp218 point OOF ではなく、
同一 center の scalar control を上回ることである。

## 計算契約

- 現 active: HMM 1、residual-scale fit 0、LightGBM config 0、booster 0。
- deferred shrinkage: HMM 1、well GroupKFold scale fit 5、LightGBM booster 0。
- CPU / internet disabled / parent-control retraining なし。

## 結果

Kaggle CPU v2は3,783,989 rows / 773 wellsを約8.50時間で完走した。scalar HMMのRMSEは
`8.361307776`で、exp218 point OOF比`-0.114496982`、exp234 row-wise sigma HMM比
`-0.065923625`だった。distance bucketは6個中5個、wellは501 / 773で改善したが、
`500_1000` bucketと272 wellsは悪化したため、inference / submissionへは進めない。

alpha 0.25 v3はRMSE `8.351122273`でscalar比`-0.010185503`改善した。全6 distance bucketは
改善したが、MAE、within10、hidden-like 2群と421 / 773 wellsはscalarより悪化した。

alpha 0.50 v4はRMSE `8.336863897`でalpha 0.25比`-0.014258376`、有限grid最良だった。
ただしMAE、within10、hidden-like 2群と421 / 773 wellsはalpha 0.25より悪化した。

## 所見

same-center比較ではalpha `0.25`がscalarを主指標で小幅に上回ったが、改善は一様でない。
alpha `1.0`相当は強すぎ、alpha `0.50`は有限ablationとしてのみ検討する。

## 次

本方向性はclosed。追加alpha探索、再実行、raw-test inference、submissionは行わない。
