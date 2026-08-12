# タスクリスト

## 設計（今回）

- [x] exp431 の experiment/steering 雛形を作成
- [x] exp427 の完全 PASS を必須先行条件に固定
- [x] fixed 2x2 尤度、block support、total evidence、temperature を固定
- [x] preflight/full の実行量と technical/scientific gate を固定
- [x] 再現性・truth-late・SHA 契約を記載
- [x] README、SESSION_NOTES、result、metrics を waiting/design-only に更新
- [x] `KAGGLE_DIRECTION.md` と `experiment_summary.md` に登録
- [x] repository validation を通す

## 先行条件

- [x] exp427 version 2完了結果を監査
- [x] exp427 technical AND gate FAILを確認
- [x] exp427 scientific AND gate FAILを確認
- [x] exp427 scientific contract / target-free bundle SHAを記録
- [x] `closed_prerequisite_failed`としてterminal close

## 実装以降（prerequisite FAILにより中止）

- [x] compact self-contained実装を作成しない
- [x] fixed 4-well technical preflightを実行しない
- [x] full trajectory replayをpushしない
- [x] factorial metrics / trajectory / predictionを生成しない
- [x] 推論・提出候補化へ進まない
