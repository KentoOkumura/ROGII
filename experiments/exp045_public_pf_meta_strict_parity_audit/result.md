# exp045_public_pf_meta_strict_parity_audit 結果

## 仮説

`exp035` の見えない test well 用 meta 処理の悪化は、exp034 の見えない test 風 train 生成物と、見えない test well 用推論側 PF 診断値の生成条件差が原因かもしれない。見えない test well 用推論側を exp029 と同じ条件の `16 seeds / 250 particles` に揃えることで、この仮説を切り分ける。

## 設定

- 親: `exp035_public_sel15_pf_meta_inference_port`
- route: `pf_beam`
- 基準予測: exp026-style pseudo-tail 基準予測
- 補正: `0.75 * clip(ridge_meta_residual, -60, 60)`
- meta training source: exp029 `public_sel15_pf_oof_features.csv.gz`
- 見えない test well 用 PF 診断値: `n_seeds=16`, `n_particles=250`, selector scales `[3, 5, 8, 12]`
- 補助 CV: exp034 original-fold 14.313668、well-hash 14.172010

## 結果

| メトリック | 値 |
| --- | --- |
| CV | - |
| Public LB | 19.177 |
| rows | 14,151 |
| submit-check | PASS |
| output SHA256 | `2b86386f19279e79e7184096f353ccf2b97785de67b268caa56aa5f85405a815` |
| changed rows / wells | 0 / 0 |
| diff RMSE vs exp026 基準予測 sample output | 0.000000 |
| 予測範囲 | 11587.038593 - 12240.016066 |
| 見えない test well 用 PF 診断値 | 16 seeds / 250 particles |
| code submit ref | `53502516` |

## 解釈

Kaggle inference version 1 は完了した。log と `public_sel15_meta_corrected_summary.json` で 見えない test well 用 PF 診断値が `hidden_pf_n_seeds=16`、`hidden_pf_n_particles=250` として実行されたことを確認した。

ただし public sample は全 3 wells が見えている train wells で物理処理を使うため、見えない test well 用 meta 処理は output には発火しなかった。`changed_rows=0`、`diff_rmse=0.000000`、submission SHA は exp035 と同じ `2b86386f19279e79e7184096f353ccf2b97785de67b268caa56aa5f85405a815`。

code submit ref `53502516` は Public LB 19.177 で完了した。exp035 13.738 から +5.439、exp027 8.781 から +10.396 悪化したため、exp029 と同じ条件の `16 seeds / 250 particles` に揃えても見えない test well 用 meta 処理は転移しない。

## 次

exp034/035-style の exp026 基準 + Ridge meta residual 見えない test well 用処理は追加チューニングしない。exp027 基準 8.781 を維持する。
