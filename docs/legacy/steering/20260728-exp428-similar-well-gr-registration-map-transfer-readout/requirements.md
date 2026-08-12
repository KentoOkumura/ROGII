# 要件

## 依頼

GR波形が似たwell同士について、他wellで得られた
`Type Well GR ↔ Horizontal GR` の対応付け方を再利用できるか検証する実験を設計する。
対応付け方は、少なくとも「Type Well GRを真のTVTから何ftずらすとHorizontal GRに
一致するか」を表すregistration offsetを含む。

初回依頼ではバックログ、steering、実験scaffoldと設定を作成し、設計だけを確定した。
2026-07-28の追加依頼「exp428を実装してください」により、固定設計を変えずにcompact
self-contained train source、専用test、正規train notebookまで実装する。その後の追加依頼
「実行してください」によりKaggle CPU package / push / Stage 0 runまで承認された。
推論、提出、既存手法への統合は引き続き行わない。

## 意図の固定

- donorから転写するのは正解TVT pathやTVT増分ではなく、donor内で測った
  Type Well–Horizontal GR registration offsetである。
- same-Type-Well eligibilityはType Well GR波形の一致、donor順位はHorizontal GR波形の
  類似度で決める。
- primaryは似たdonor 1本のwell-level global shift（ft）とする。
- donorの局所offset曲線、stretch、local warpはmapping shapeの追加診断とし、
  実行後にprimaryへ差し替えない。
- registration offsetは観測モデル側のずれであり、TVT補正量と同一視しない。

## 制約

- Route: `pf_beam`
- 親: `exp423_same_typewell_gr_dtw_truth_warp_transfer_readout`
- Stage 0のtrain-side、5-fold、0-model、0-booster、CPU readoutだけを設計する。
- query wellはouter-valid、donor wellはouter-trainに完全分離する。
- queryのsuffix GRは観測可能、queryのsuffix true TVTはcandidate/artifact freeze前に
  読み込まない。
- donor true TVTはdonor自身のregistration map推定だけに使用できる。
- `exp065` native-overlap pairのrow lagをft補正に直接使わない。Type Well TVT軸の差は
  exact-overlap行の`TVT`差から決める。
- shift grid、block、support、similarity、tie-break、primary、control、gateを実行前に固定し、
  same-OOF rescue gridを行わない。
- baseline/controlの再学習、LightGBM、HMM、PF、Beam、test inference、submissionは対象外。
- 再現性は`docs/06_reproducibility.md`に従う。

## 受け入れ基準

- steering 3文書、実験scaffold、`config.yaml`、`README.md`、`SESSION_NOTES.md`、
  `result.md`、`metrics.json`、`KAGGLE_DIRECTION.md`、`experiment_summary.md`が整合する。
- donor mapの定義、符号、Type Well軸変換、GR類似donor選択、primary/control、
  query truth late join、成功条件、停止条件が実装者の追加判断なしに読める。
- exp423との違いが「truth-warp転写ではなくregistration offset転写」と明記されている。
- 実行量はaudit 1、reporting folds 5、LightGBM config / trained fold / booster /
  PF / HMM / Beam / GPU / parent replayがすべて0と記録されている。
- compact self-contained train sourceと正規train notebookが同じセル構造を持ち、
  inference notebookだけが非対象のplaceholderとして残る。
- 専用test、構文、F821、Jupytext round-trip、strict experiment validationが通る。
- gzip生成物の再現性を比較する場合はdecompressed content SHAを主証拠とし、
  独立rerun一致まではdeterministic anchorと呼ばない。

## 今回の非目標

- exp423のparameter rescueまたはnegative resultの再分類。
- donor true TVT path、geometry、rate、truth-warpの転写。
- registration offsetをquery TVTへ直接加減すること。
- local map、global shift、group medianのうちrun後に最良のものをprimaryへ変更すること。
- current testへの適用、既存candidate/emissionへの統合、submission作成。
