# exp028_public_replay_second_sel15_or_blend_audit Result

## 仮説

Public `needless090/lb-8-860-rogii-sel15-256seeds` can be replayed as a standalone Kaggle inference notebook without external artifact dependencies. A blend audit is only meaningful after this second replay output is known.

## 設定

- 親: `public_notebook_catchup_after_self_improvements`
- 検証: no local CV; Kaggle replay and submit-check only
- メトリック: RMSE
- シード: public notebook fixed settings

## 結果

| メトリック | 値 |
| --- | --- |
| CV | - |
| Public LB | - |
| Private LB | - |

## 結果メモ

Kaggle inference version 2 completed as `kentookumura/exp028-second-sel15-replay`. Runtime log reached `Done: 14151 rows` at about 384 seconds and wrote `submission.csv`.

Submit-check passed. Output SHA256 is `2b86386f19279e79e7184096f353ccf2b97785de67b268caa56aa5f85405a815`.

Prediction range is 11587.038593 to 12240.016066, mean 11903.630073. Compared with exp027, all differences are exactly zero: RMSE 0.000000 and correlation 1.000000.

## 解釈

The second replay does not add diversity. Although the source title was LB 8.860, the actual generated output is identical to the already submitted exp027 output, so a separate submit would spend quota for the same Public LB 8.781 result.

## 次

Do not submit exp028 separately. Move back to self-route work or use exp027 as the fixed public replay 基準 for future audits.
