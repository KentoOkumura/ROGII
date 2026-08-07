# タスクリスト

## 次のアクション

なし。exp348はStage 0固定AND gate不通過でbranchを閉じた。

## 未着手・実施しない

- Stage A/B/C
- 推論、提出
- negative family/count、margin、loss、window、architecture、decoder、epochの救済

## ブロック中

なし。

## 完了

- exp348として採番し、steeringと実験scaffoldを作成した。
- exp332を再開せず、window path-rankingを独立高リスク実験として系譜固定した。
- positive 1 / negative最大16、margin/loss、Stage 0 technical/learning/runtime/memory gate、実行量、failure policyを固定した。
- compact self-contained train、fail-closed inference候補、専用contract testsを実装した。
- compact trainを正規train Notebookへ採用し、private / internet無効 / T4 packageと各SHAを固定した。
- 初回pushが週45時間GPU quotaでversion作成前に拒否されたことを確認し、quota回復後に同一科学契約でretryした。
- version 1のraw `id`列仮定による学習前ERRORを診断し、正規ID契約`{well}_{row_index}`へtechnical fixした。
- version 2を同じcanonical kernelへpushし、固定16-window Stage 0を完走した。
- report、logs、必要な6生成物を取得し、report内SHAと実ファイルSHAを照合した。
- Technical PASS、Memory PASS、Learning FAIL、Runtime FAILを記録した。
- positive top-1 `0.0`、margin `-0.388485386967659`、保守的fold runtime `75.35670035238391 h`、peak memory `1.1935901641845703 GB`を固定gateで判定した。
- `close_without_negative_bank_margin_or_science_rescue`としてbranchを閉じた。
