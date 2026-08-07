# exp369_stratified_registration_offset_pf

## 状態

- ルート: `pf_beam`
- 状態: 設計確定・未実装
- CV / LB / Submit: なし
- 作成日: 2026-07-23
- 親実験: `exp072_exp063_full_replay_feature_cache`

## 仮説

小さなGR registrationずれをphysical positionとは別stateとして持ち、各delta層の粒子を維持すれば、
PFが早期resamplingで正しいregistration modeを失わずに済む。

## 変更点

- 粒子状態を `(physical_position, rate, delta)`へ拡張する。
- deltaは`[-6,-3,0,3,6] ft`、初期countは`50/75/250/75/50`。
- 各delta層をresample後最低25粒子維持する。出力はp、GR emissionだけp+delta。

## 検証方針

- Stage 0はvisible prefixの128/64 rolling-originでdelta=0比GR NLLを評価する。
- Stage 1は全gateと別承認後だけ500 particles × 128 seeds × 773 wells。
- 保存済みexp072 controlを使い、control PFは再実行しない。

## 実行入口

生成済み train / inference Notebook は placeholder であり実行対象ではない。

## 結果

未実装・未実行。

## 所見

### 良かった点

- physical TVTとregistrationを分離し、rate predictabilityを必要としない。

### 悪かった点

- 5層へ粒子を分けるため、delta=0内のposition/rate解像度は低下する。

### リスク / 注意

- deltaをTVT出力へ足さない。層quotaがwrong modeを延命し得る。
- delta、quota、transition、particle/seed数のgridは禁止する。

## 次

実装は行わない。別承認時はknown-prefix Stage 0だけを先に実装する。
