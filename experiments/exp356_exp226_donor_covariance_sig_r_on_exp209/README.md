# exp356 exp226 donor-covariance sig-r on exp209

## 状態

- Route: `pf_beam`
- 状態: 設計確定、未実装、未実行
- 優先度: 低・P4
- 親: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- 履歴参照: `exp324_exp226_donor_covariance_segment_sig_r`

## 仮説

fold-safe exp226 donorの局所rate dispersionがtransition uncertaintyを表すなら、
constant `sig_r=0.002`よりK16 segment別のNLL校正が良くなる可能性がある。

## 変更点

- exp323 rate scheduleとexp307--309 chainを削除した。
- exp209 rate meanを固定し、`sig_r,t`だけを変更する。
- Stage 0は0-HMM calibration readout、Stage 1はPASS・別承認時だけ773 runs。

## 検証方針

- donor/support/scale/clip/fallbackをsuffix truth前にSHA固定する。
- Stage 0はtransition NLL、coverage、fold、stress、fallback/clipをAND評価する。
- exp355 schedule、suffix variance fit、grid、blendは禁止する。

## 実行入口

- train: `exp356_exp226_donor_covariance_sig_r_on_exp209_train.ipynb`
- inference: `exp356_exp226_donor_covariance_sig_r_on_exp209_inference.ipynb`
- 現在はplaceholderであり、実行は禁止する。

## 所見

exp338ではwell-adaptive`sig_r`が全well upper clipとなったため、donor covarianceも
clipへ崩壊しないことをStage 0で先に要求する。

## 次

別承認時だけStage 0を実装する。
