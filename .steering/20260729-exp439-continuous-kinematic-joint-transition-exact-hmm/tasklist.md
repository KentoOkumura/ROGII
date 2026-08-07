# タスクリスト

## TODO

- なし。Stage 0 technical FAILによりbranchを閉鎖した。

## ブロック中

- Stage 1 full OOF、inference、submission。
- rate/position/emission/prior tuning、hard trigger、ML、selector、blend、reset/re-anchor。
- support、moment、noise、grid、rate、emission、prior、gateのsame-exp救済と再実行。

## 完了

- 2026-07-29: 既存exp437を別仮説として維持し、未使用exp439へ採番した。
- 2026-07-29: exp209のrate marginalを固定した台形joint edgeへ科学差分を限定した。
- 2026-07-29: 5/7/9セルの非負maximum-entropy moment projectionを固定した。
- 2026-07-29: fixed32の1 variant、technical/mechanism AND gate、Stage 1条件を固定した。
- 2026-07-29: route、実行量、禁止事項、truth-late、SHA契約を固定した。
- 2026-07-29: ユーザーの実装承認を受け、compact self-contained train候補を実装した。
- 2026-07-29: 5/7/9非負maximum-entropy projection、shared joint-edge
  forward/backward、rate/moment/covariance audit、brute-force HMM parityを実装した。
- 2026-07-29: fail-closed inference候補と12件のcontract testを実装した。
- 2026-07-29: py_compile、Ruff、Jupytext test、strict experiment validationを通した。
- 2026-07-29: 既存の正規notebook placeholderは上書きせず、compact候補Notebookを生成した。
- 2026-07-29: ユーザーの実行承認を受け、正規train/inference Notebookへの
  compact self-contained候補採用とStage 0 package/push/runを承認済みに更新した。
- 2026-07-29: 実行量を1 variant ×32 candidate HMM well-runs、parent rerun 0、
  LightGBM config / trained fold / booster / PF / Beam / GPU各0と再確認した。
- 2026-07-29: 正規notebookをcompact self-contained版へ採用し、private CPU、
  internet off、run-on-push、strict、`--no-src`のKaggle packageを作成した。
- 2026-07-29: 60文字slugの初回pushはKaggle APIが400で拒否したため、仮説を
  変えず43文字slugへ短縮し、kernel version 1（id_no `129058811`）を実行した。
- 2026-07-29: 最初のwell `060ab2b8`のrow 0でtarget variance
  `0.01500625 ft^2`が非負0.35 ft latticeの最小分散`0.0264 ft^2`を下回ることを
  検出し、事前登録どおりtechnical FAILで閉鎖した。
- 2026-07-29: 完了HMM well-run、prediction artifact、truth/role/fold/episode read、
  Stage 1、inference、submissionはすべて0。
