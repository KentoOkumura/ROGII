# exp003_residual_ablation 結果

## 仮説

exp002 の residual model を、sampling cap、residual shrink、feature set の one-at-a-time ablation で分解すれば、CV 改善を保ちながらより安定した設定を選べる。

## 設定

- 親: `exp002_drift_minimal`
- 検証: `well_id` GroupKFold、`TVT_input` NaN 行のみ評価
- メトリック: RMSE
- シード: 42
- モデル: `HistGradientBoostingRegressor`
- target: `TVT - last_anchor_tvt`
- baseline CV: exp002 `drift_hgb` 14.124569

## 結果

| Variant | 変更 | CV | Mean Fold RMSE | exp002 差分 |
| --- | --- | ---: | ---: | ---: |
| `feature_no_gr_signal` | raw / derived GR signal features を除外 | 13.882944 | 13.859376 | -0.241625 |
| `sample_per_well_400` | per-well sample 800 -> 400 | 14.122145 | 14.099246 | -0.002424 |
| `control_exp002` | exp002 再実行 | 14.124569 | 14.101909 | 0.000000 |
| `shrink_100` | residual shrink 0.85 -> 1.00 | 14.127689 | 14.106161 | +0.003120 |
| `sample_total_200k` | fold total sample 300k -> 200k | 14.183193 | 14.159896 | +0.058624 |

Selected variant: `feature_no_gr_signal`

Fold RMSE (`feature_no_gr_signal`): 13.246628 / 13.470734 / 13.037988 / 13.892880 / 15.648648

| Submission | Value |
| --- | --- |
| Kernel | `kentookumura/exp003-residual-ablation-inference` |
| Version | 1 |
| Submission ref | 53213975 |
| Public LB | 12.852 |
| Private LB | - |

## 解釈

GR 系特徴を外す `feature_no_gr_signal` が最良で、exp002 の 14.124569 から 13.882944 へ 0.241625 改善した。per-well sampling を 400 に下げる効果はほぼ誤差範囲、total sample 200k は悪化、shrink 1.00 もわずかに悪化した。

現時点では、exp002 の既存 GR raw / rolling / delta 特徴は well-level CV ではノイズまたは分布差を拾っている可能性が高い。次に GR を使うなら、既存 GR 特徴の単純追加ではなく、typewell alignment や fold-safe な local matcher として再設計する。

一方で public LB は exp002 の 12.533 から exp003 の 12.852 に悪化した。CV では GR signal removal が有利だが、visible public wells では exp002 の GR feature が効いていた可能性がある。次の実験前に、OOF 改善 well と public visible well の条件差を確認する。

## CV/LB 逆転の調査

追加 artifact:

- `artifacts/exp002_exp003_well_delta.csv`
- `artifacts/visible_submission_well_comparison.csv`

OOF では 773 wells のうち 408 wells が改善、365 wells が悪化した。全体の weighted SSE delta は -25.6M で改善だが、上位 20 改善 well だけで net 改善量を上回る。つまり exp003 の CV 改善は「広く薄い改善」ではなく、exp002 が 基準 より悪くなる hard wells の大きな悪化を抑えた効果が強い。

`exp002` が `last_anchor` より悪い 244 wells では exp003 の net SSE が大きく改善した。一方、`exp002` が `last_anchor` より良い 529 wells では median RMSE delta は +0.076692 で、典型的には exp003 が少し悪化する。public split が後者に寄ると、CV 改善と public LB 悪化は両立する。

visible duplicate 3 wells は train と同じ軌跡で、Kaggle submit の hidden public score そのものではないが、方向性の sanity check になる。local truth で見ると exp003 は `00e12e8b` では改善した一方、`00bbac68` で +1.239759、`000d7d20` で +0.119149 悪化した。OOF ではこの 3 wells すべてで exp003 が改善していたため、fold holdout と final full-train inference の挙動差もある。

結論として、実装バグよりも validation target の代表性と feature の効き方の問題が濃い。GR raw / rolling / delta を外すと hard well の過補正は減るが、GR が効く easy/visible/public 寄り well では悪化しうる。

## 次

1. `exp002_exp003_well_delta.csv` で worsened wells の共通条件をタグ付けする。
2. GR feature を全削除ではなく、hard well 判定に応じた gating / shrink にする。
3. GR を戻す場合は raw rolling ではなく local matcher / typewell alignment として再設計する。
