# タスクリスト

## TODO（追加の明示承認が必要）

- [ ] なし。inference / submission / integrationはterminal FAILにより対象外。

## 進行中

- なし。

## ブロック中

- なし。version 2のsupport FAILが独立してterminalのため、独立rerunは不要。

## 完了

- [x] `kaggle-review-exp`の実験設計手順と`docs/06_reproducibility.md`を確認した。
- [x] steeringをexperiment scaffoldより先に作成した。
- [x] `exp428_similar_well_gr_registration_map_transfer_readout` scaffoldを作成した。
- [x] Routeを`pf_beam`、親をexp423に固定した。
- [x] exp423 truth-warp転写との相違を固定した。
- [x] Type Well軸変換、shift符号、block、shift grid、identifiabilityを固定した。
- [x] same-Type-Well eligibilityとHorizontal GR-DTW順位を固定した。
- [x] global shift primary、local mapping-shape diagnostic、controlsを固定した。
- [x] query truth late join、fold分離、SHA契約を固定した。
- [x] technical/scientific gateとfail-closed分岐を固定した。
- [x] backlog、実験一覧、実験文書をdesign-only状態へ更新した。
- [x] 初回design-onlyセッションではcanonical train/inference notebookをplaceholderのまま保持した。
- [x] 追加実装依頼を受け、compact self-contained Jupytext train sourceを実装した。
- [x] Type Well軸graph、donor registration map、GR-DTW donor選択を実装した。
- [x] target-free freeze、query truth late join、primary/control/global/local gateを実装した。
- [x] mapping-shape、hidden-like、by-well safety readoutを実装した。
- [x] 専用test 15件、構文、F821、Jupytext round-trip、strict experiment validationを通した。
- [x] 正規train notebookへ採用し、inference notebookはplaceholderのまま保持した。
- [x] Kaggle CPU packageを作成し、metadata / package configを検証した。
- [x] run前にaudit 1、reporting folds 5、model/config/booster/PF/HMM/Beam/GPU/
  parent replayが0であることを再確認した。
- [x] canonical private CPU kernel version 1を実行し、DTW欠損処理のtechnical bugを特定した。
- [x] 親exp423互換の決定的補間へ最小修正し、回帰testを追加した。
- [x] canonical kernel version 2を実行し、supported `306 / 773 = 39.586%`を確認した。
- [x] 固定technical/scientific/local gateをFAILと判定した。
- [x] support FAILがterminalのため独立rerunを行わず、deterministic anchor falseを維持した。
- [x] README / SESSION_NOTES / result / metrics / experiment summary / directionを更新した。
