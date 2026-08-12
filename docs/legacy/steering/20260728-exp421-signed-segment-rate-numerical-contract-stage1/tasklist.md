# タスクリスト

## TODO

- なし。

## 進行中

- なし。

## ブロック中

- なし。

## 完了

- 2026-07-28のユーザー承認を得た。
- route、差分、`1e-10 ft` numerical contract、実行量、再現性を固定した。
- active variant 1、config 1、fold 5、CPU booster 5、exp226/control再学習0、
  PF/HMM/Beam 0、GPU 0を記録した。
- exp418 compact sourceからexp421 self-contained trainを作成した。
- exp418 Stage 0 eligibilityとtruth-free synthetic numerical auditを実装した。
- 正規train Notebook、contract tests、strict packageを作成・検証した。
- Kaggle v1はnumerical contract PASS後にfeature SHA guardで0 booster停止した。
- 診断v2を0 variant / 0 config / 0 fold / 0 booster / truth 0列に固定した。
- 診断v2をKaggleで完了し、canonical train row SHA `d8e932...`を再現した。
- v1期待値が3井戸current-test SHAだったことを確認し、train SHA参照だけを訂正した。
- Stage 1 v3はrow SHAを通過後、feature-freeze aggregateで0 booster停止した。
- 診断v4でfold/segment/parity一致と保存nested再読込SHA `8140e7...`を確認した。
- Stage 1 v5を5 CPU boosterで完走し、CV `9.405572476`を記録した。
- 固定gate 8 PASS / 7 FAILで`FAIL_CLOSE_BRANCH`とし、prediction/model/summary
  SHAと小規模成果物を保存した。submissionは生成していない。
