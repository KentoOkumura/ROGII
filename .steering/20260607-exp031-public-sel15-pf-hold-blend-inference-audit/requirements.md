# 要件

## 依頼

`public_sel15_pf_hold_blend_inference_audit` を実装する。`exp027` の公開 sel15 replay を親にし、`exp030` で fold 外でも支持があった fixed `pf090_hold010` を inference flow に移植できる状態にする。

## 制約

- Route: `pf_beam`
- Kaggle Notebook inference を正とする。
- 公開 sel15 の PF/Beam selector 本体は維持し、hard selector や bucket selector は追加しない。
- 見えない test well の公開 selector 予測だけを `0.90 * selector + 0.10 * last_known_TVT_input` にする。
- visible train well の physical model 出力は変更しない。
- 提出前に Kaggle output の SHA256、exp027 anchor との差分、submit-check を記録する。

## 受け入れ基準

- `experiments/exp031_public_sel15_pf_hold_blend_inference_audit/` に config、settings、train/inference notebook、metrics、README、SESSION_NOTES、result がそろう。
- inference notebook が `submission.csv` に fixed blend 候補を書き、元 selector submission と diff/summary artifact も保存する。
- `task validate-exp` 相当の構造検証が通る。
- Kaggle 実行前の状態として、未実行の submit-check / LB がメモに明示されている。
