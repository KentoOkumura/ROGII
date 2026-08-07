# exp325 セッションノート

## 目的

exp226 window GR algorithmをTVT補正ではなくexact HMMの時間変化する`lambda_t`として移植する。

## 現在の状態

- 2026-07-21: steering/scaffold作成、2-stage設計確定。
- terminal closed / 未実装 / 未実行。
- Stage 0は1 diagnostic・HMM 0、Stage 1最大1 variant・773 HMM runs、0 booster、control再実行0。

## 固定事項

- window 500 / stride 125 / finite 0.5 / exp226 score weights固定。
- state標準化、overlap 0.25、posterior-SD shrinkを固定しgridを作らない。
- exp231 peer atlas、exp226 correction/final prediction、blendは禁止。

## 再現性

real scoreはRNGなし。shuffle controlだけwell/window keyのstable SHA256 local RNG。window identity、score surface、lambda、input、prediction SHAを記録する。

## 2026-07-22 閉鎖

親exp323のterminal closeにより本実験も閉鎖した。reparentや実装再開は行わない。exp338 PASS後の新exp323相当がさらにPASSした場合だけ、新番号で新exp325相当を設計する。

## 次

なし。
