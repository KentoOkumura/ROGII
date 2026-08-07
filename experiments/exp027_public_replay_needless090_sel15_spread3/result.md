# exp027_public_replay_needless090_sel15_spread3 Result

## 仮説

Public `needless090/lb8-781-rogii-sel15-spread3` can be replayed as a standalone Kaggle inference notebook without external artifact dependencies.

## 設定

- 親: `public_notebook_catchup_after_self_improvements`
- 検証: no local CV; Kaggle replay and submit-check only
- メトリック: RMSE
- シード: 42

## 結果

| メトリック | 値 |
| --- | --- |
| CV | - |
| Public LB | - |
| Private LB | - |

## 結果

Kaggle inference version 1 completed and produced a valid 14,151-row `submission.csv`.
Submit-check passed. The output SHA256 is `2b86386f19279e79e7184096f353ccf2b97785de67b268caa56aa5f85405a815`.

Prediction range is 11587.038593 to 12240.016066, mean 11903.630073. Compared with `exp026`, the diff RMSE is 8.098430 and correlation is 0.999685.

Competition submission ref `53420592` completed with Public LB 8.781.

## 解釈

Replay execution is successful and reproduces the public title score. This becomes the current Public LB 基準.

The initial CLI submission failed because it used the regular file-upload submission path (`-f /tmp/.../submission.csv`) instead of the Notebook-only code competition path (`-k <kernel> -v <version> -f submission.csv`). Use `task submit-code` for future CLI code submissions.

## 次

Decide whether to replay another public sel15 route or return to self-route pseudo-tail experiments.
