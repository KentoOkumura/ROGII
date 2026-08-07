# exp066_cv_submit_gate 結果

## 仮説

`exp063` の strict Pixiux public replay は、CV と推論 sanity の条件を満たしているため、code submit で LB を確認する価値がある。

## 設定

- 親: `exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit`
- 補助 probe: `exp064_train_test_well_id_assert_probe`
- 検証: 既存 `exp063` metrics / source submission / `exp064` hidden overlap probe の gate audit
- メトリック: RMSE
- シード: 42

## 結果

| メトリック | 値 |
| --- | --- |
| source CV | 9.630105 |
| selected single-model CV | 9.628965 |
| Pixiux mean vs Ravaghi mean delta | -0.930432 |
| Gate required rules | 11/11 PASS |
| Decision | `approved_for_code_submit` |
| Public LB | - |
| Private LB | - |

## 解釈

`exp063` inference v2 は complete、submit-check PASS、fallback rows 0、14,151 rows、予測範囲 11,593.674805 - 12,240.098633、SHA256 は metrics 記録と一致した。`exp064` の hidden code submission probe も complete し、train/test same `well_id` assertion は発火しなかった。

したがって、提出対象は `kentookumura/exp063-ravaghi-pixiux-strict-replay-infer` version 2 の `submission.csv` として承認する。ただしこれは LB anchor 更新ではない。Public LB は未確認で、提出後の実スコアを見てから `exp063` / `exp066` / submission history に記録する。

提出コマンド候補:

```bash
kaggle competitions submit rogii-wellbore-geology-prediction -k kentookumura/exp063-ravaghi-pixiux-strict-replay-infer -v 2 -f submission.csv -m "exp063 strict Pixiux replay lgb_mean CV 9.630105; exp066 gate approved"
```

## 次

提出回数を使う場合は上記の code submit を実行し、Public LB を `submissions/SUBMISSIONS.md` と関連実験に記録する。
