# exp327 セッションノート

## 目的

現行position sigma floorより大きい、grid quantization由来の時間変化`sig_p,t`を低優先で監査する。

## 現在の状態

- 2026-07-21: steering/scaffold作成、設計確定。
- terminal closed / 未実装 / 未実行。
- Stage 0は1 diagnostic・HMM 0、Stage 1最大1 variant・773 HMM runs、0 booster、control再実行0。

## 固定事項

- floor 0.1225、上限0.245、quantization formula 1本。
- grid、rate transition、GR、momentum、posterior outputは固定。
- 0.1225未満の無効sigmaやgrid searchは禁止。

## 再現性

RNGなし。transition mean、quantization scale、sigma schedule、prefix score、prediction SHAを記録する。

## 2026-07-22 閉鎖

親exp323のterminal closeにより本実験も閉鎖した。reparentや実装再開は行わない。exp338 PASS後の新exp323相当がさらにPASSした場合だけ、新番号で新exp327相当を設計する。

## 次

なし。
