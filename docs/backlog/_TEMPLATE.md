# 候補名

- 候補名: TODO（`KAGGLE_DIRECTION.md` の未着手表と同じ名前）
- 状態: TODO（`検討メモ・設計不可` または `設計可能・実験化未承認`）
- 対応する上位仮説: TODO（`HYP-YYYYMMDD-NN`。新しい上位仮説は `KAGGLE_DIRECTION.md` の「検証中の仮説」に同時追加）
- 関連する上位仮説: TODO（なければ`なし`。主仮説を複数にしない）
- 作成日: YYYY-MM-DD
- 最終更新日: YYYY-MM-DD
- 依頼原文: TODO
- 期待する成果: TODO
- 親実験 / 比較対象: TODO
- 優先度: TODO（`P0`、`P1`、`P2`、`P3`、`P4`のいずれか）
- 優先度の理由: TODO
- `KAGGLE_DIRECTION.md` の対応箇所: TODO

## 観測事実と根拠

- 実測済みの事実: TODO
- 根拠ファイル / 一次資料: TODO
- 利用する保存済み生成物とSHA: TODO（未確定なら未決事項へ）
- 仮定: TODO（なければ`なし`）

## この候補が直接検証する仮説と範囲

- 上位仮説のうちこの候補が検証する範囲: TODO
- この候補の具体的な仮説: TODO
- 仮説が正しい場合に期待する観測: TODO
- 仮説を棄却する観測: TODO
- この候補だけで上位仮説を判断できるか: TODO（`はい`または`いいえ`）
- 上位仮説の判断に残る検証: TODO（なければ`なし`）

## 入力・予測対象・出力・推論方法

実装区分は`docs/glossary.md`に定義したこのリポジトリ内の管理用ラベルを使う。処理内容と省略点を先に具体的に記録する。

- input: TODO
- target / objective: TODO
- output: TODO
- loss: TODO（学習しない場合は`なし`）
- decode / 推論方法: TODO
- 処理単位: TODO（`row`、`window`、`whole-well`、`set`、`field`など）
- 実装区分: TODO（参照手法がある場合は`faithful`、`staged-faithful`、`proxy`）

## 親実験からの差分

- 変更するもの: TODO
- 固定するもの: TODO
- 再利用するコード / config / 生成物: TODO
- 新しく作るもの: TODO

## 最小の反証可能な検証

- 検証方法: TODO
- variant / config / fold / booster数: TODO
- control再学習: TODO（有無と理由）
- 想定runtime / resource: TODO

## 成功条件と停止条件

- primary指標: TODO
- 成功条件: TODO（primary指標の閾値または成立を判断する具体的な観測）
- 必須guard: TODO
- 成功時の次段階: TODO（自動実行せず、必要なユーザー確認も記載）
- 失敗時の停止範囲: TODO

## 実行しないこと

- 禁止する代替実装、proxy、同一OOF上の救済探索: TODO
- 壁打ちで採らなかった案と理由: TODO

## リスク

- leakage / validation: TODO
- hidden test: TODO
- runtime / memory: TODO
- 再現性: TODO

## 未決事項

- TODO（設計可能にする場合は`なし`）

## 判断履歴

- YYYY-MM-DD: TODO

## 次セッションへの引き継ぎ確認

- 固定するものを一意に説明できる: TODO
- 変更するものを一意に説明できる: TODO
- 最小検証と停止条件を一意に説明できる: TODO
- 実行しないことを一意に説明できる: TODO
- 未決事項が明示されている: TODO
