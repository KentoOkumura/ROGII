# exp326 セッションノート

## 目的

exp323 residual-rate HMMのmomentumだけを時間変化させる。

## 現在の状態

- 2026-07-21: steering/scaffold作成、設計確定。
- terminal closed / 未実装 / 未実行。
- Stage 0: 1 diagnostic・HMM 0。Stage 1最大: 1 variant・773 HMM runs。0 model / 0 booster / control再実行0。

## 固定事項

- `s_t`、`L_t`、`m_t`の式は1本。momentum gridなし。
- exp323 `mu_r,t`、`sig_r`、`sig_p`、GR、grid、decoderは固定。
- posterior entropyや真の誤差でmomentumを変えない。

## 再現性

RNGなし。parent schedule、momentum schedule、activation、prefix readout、prediction SHAを記録する。

## 2026-07-22 閉鎖

親exp323のterminal closeにより本実験も閉鎖した。reparentや実装再開は行わない。exp338 PASS後の新exp323相当がさらにPASSした場合だけ、新番号で新exp326相当を設計する。

## 次

なし。
