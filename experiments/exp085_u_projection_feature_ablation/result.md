# exp085_u_projection_feature_ablation 結果

## 状態

Kaggle train v1 は timeout したが、logs から 59/60 fold-model metrics を回収済み。

## 仮説

U-space を target として直接学習した exp080 は悪化したため、exp085 では target を `TVT - last_known_tvt` に固定する。PF/Beam/likelihood-PF candidate path を local U-space に写し、smooth projection からのズレと候補間 disagreement を add-only feature として渡す。

## 評価方針

exp072 deterministic full replay train cache と exp073 LightGBM config family を固定し、GroupKFold by well で以下を比較する。

- `control_exp073_base196`
- `u_projection_correction`
- `u_disagreement`
- `u_projection_correction_plus_disagreement`

## 結果

正式な pooled OOF artifact は timeout により保存されていないため、以下は logs から復元した fold RMSE の単純平均で評価する。fold size の差を反映した pooled RMSE ではない。

| variant | model | completed folds | mean fold RMSE |
| --- | --- | ---: | ---: |
| `u_projection_correction_plus_disagreement` | `lgb1` | 5 | 9.291006 |
| `u_disagreement` | `lgb2` | 5 | 9.392842 |
| `u_disagreement` | `lgb1` | 5 | 9.417772 |
| `u_projection_correction` | `lgb2` | 5 | 9.450537 |
| `u_projection_correction_plus_disagreement` | `lgb0` | 5 | 9.499037 |
| `u_projection_correction` | `lgb1` | 5 | 9.503377 |
| `control_exp073_base196` | `lgb1` | 5 | 9.534549 |
| `control_exp073_base196` | `lgb2` | 5 | 9.541831 |
| `u_projection_correction` | `lgb0` | 5 | 9.557780 |
| `control_exp073_base196` | `lgb0` | 5 | 9.633324 |
| `u_disagreement` | `lgb0` | 5 | 9.726577 |
| `u_projection_correction_plus_disagreement` | `lgb2` | 4 | 9.037692 |

`u_projection_correction_plus_disagreement` は `lgb1` で control `lgb1` から -0.243542、`lgb0` で control `lgb0` から -0.134287 改善した。`lgb2` は fold4 未完了だが、完了済み 4 folds は 9.037692 で最良水準。fold4 が他モデル同様に 10.3 前後なら平均は約 9.29 になり、`lgb1` と同程度の改善と推定できる。

`u_disagreement` 単独も `lgb1/lgb2` では control より -0.116776 / -0.148989 改善したが、`lgb0` では +0.093254 悪化した。projection correction 単独は全モデルで小幅改善またはほぼ同等。

## 次の判断

U-space projection features は target ablation とは異なり有望。次は全 variant を再実行せず、`u_projection_correction_plus_disagreement` だけに絞って full 5-fold pooled OOF / feature importance / bucket metrics を完走させる。完走後に worst-well、distance bucket、tail rank bucket を確認し、悪化 guard が許容範囲なら inference port を検討する。
