# exp096_projection_fadein_after_prefix 結果

## ステータス

Kaggle train v1 と hidden-compatible inference v2 は完了。submit-check は PASS。提出 ref `53896594` は Public LB 8.651。

## 実行

- Kernel: `kentookumura/exp096-projection-fadein-after-prefix-train` v1
- URL: https://www.kaggle.com/code/kentookumura/exp096-projection-fadein-after-prefix-train
- Output: `/tmp/kaggle-output/exp096_projection_fadein_after_prefix/train_v1`
- Runtime: 1137.359 秒
- Rows / wells: 3,783,989 / 773
- CPU / internet off

Inference:

- Kernel: `kentookumura/exp096-projection-fadein-after-prefix-inference` v2
- URL: https://www.kaggle.com/code/kentookumura/exp096-projection-fadein-after-prefix-inference
- Output: `/tmp/kaggle-output/exp096_projection_fadein_after_prefix/inference_v2`
- GPU / internet off
- exp073 train source: `kentookumura/exp073-full-replay-repro-guard-train`
- exp073 base prediction は current test から notebook 内で再生成
- submission rows: 14,151
- fallback rows: 0
- submission SHA: `41b251c4ef29aa9daee62768890b406621310b7977164c24923a979e844dbaf5`
- submit-check: PASS

Submission:

- ref: `53896594`
- Public LB: 8.651
- exp073 raw anchor 8.780 からは改善
- exp077 ML route submitted/postprocessed anchor 8.611 には届かず

## 結果

| 項目 | 値 |
| --- | --- |
| baseline exp073 OOF RMSE | 9.526374817 |
| exp094 global best RMSE | 9.399456024 |
| best fade-in RMSE | 9.397537231 |
| delta vs exp073 | -0.128837518 |
| delta vs exp094 global best | -0.001918793 |
| best variant | `degree4_beta0.75_c2_fade250_750` |
| correction p95 | 3.350098 |
| correction max | 20.202393 |
| prediction SHA | `0cdb4c7e0add9584f8847a8ab63fd02be09c3d573cc8ad776f76211f185634b3` |

## Guard

- `passes_guard=true`
- recommendation: `port_to_inference_candidate`
- max fold regression: -0.092230
- near row regression: 0.0
- short tail regression: +0.001918

best variant の bucket delta:

- distance 0-50 ft: 0.0
- distance 50-100 ft: 0.0
- distance 100-250 ft: 0.0
- distance 250-500 ft: -0.098267
- distance 500-1000 ft: -0.164007
- distance 1000+ ft: -0.134509
- tail rank 0-99: 0.0
- tail length 0-499: +0.001918

## 解釈

exp094 の global projection は overall RMSE を改善したが near-prefix を壊した。exp096 の fade-in は `md_since <= 250` で correction を完全に止めるため、distance 0-250 ft と tail rank 0-249 は baseline と同一になり、near-prefix regression は 0.0 になった。

同時に 250 ft 以降では projection gain が残り、best は exp094 global best よりもわずかに良い。train-side では採用候補として十分だが、hidden test に移す前に test-side projection feature parity、sample diff、submit-check を確認する。

提出 v1 は `Notebook Threw Exception` で失敗した。原因は inference v1 が exp073 の public inference output を読み込む public-output-copy 型で、hidden rerun の test row/well set に追従できない構造だったため。v2 では exp073 の saved booster inference を exp096 notebook 内に同梱し、exp073 train artifact から current test に対するベース予測を生成してから projection を適用する source-port 型に修正した。

v2 提出は Public LB 8.651 で、exp073 raw anchor 8.780 からは -0.129 改善した。一方で exp077 postprocess anchor 8.611 より +0.040 悪いため、ML route anchor は更新しない。OOF の exp077 比較がなかったため、exp096 は exp073 raw には有効だが exp077 の policy postprocess を置き換えるほどではなかったと判断する。

## 再現性

- exp073 OOF decompressed content SHA: `fd6c68050058c40b4960f3ff2af9905bfcb1c12d540c71e331d0aa85ca9756a4`
- exp073 OOF raw gzip SHA: `986e26c5c6617ade714623d44433e9beacdb2b1027d46c4a4e70825bc8ab87fc`
- best predictions raw gzip SHA: `87a08c68a15abea9dd543d9e81800c2c6f208bc68f843f99af7ce62db259bdae`
- best predictions decompressed content SHA: `05c6787e19f5058b8d82bd1861072cb263976a70a35355969ae548ffc0d8d6a8`
- exp073 regenerated inference decompressed content SHA: `f3f04e56f3035191d651e330d26ee48e819f42cf0497acecefc88fe985cdc219`
- exp073 regenerated inference raw gzip SHA: `e9e7826fc4cfe3b0c56e34ac34ae9b8ebcefa8a8fe16bf44864e0d7ede397bab`
- exp073 regenerated test feature SHA: `f778b7238ef333bf8a639435be4b924c97d0c3e1a685545991cfe9a3dd1b7623`
- inference test predictions raw gzip SHA: `e701f0c0f629a820e0e374695a4bfc5600ff41d9d3dc13edc396234eda5f2274`
- inference test predictions decompressed content SHA: `a99e3625ef886a67fc0731b943247ff508f5422d97ef438555b93a633885d089`
- inference submission SHA: `41b251c4ef29aa9daee62768890b406621310b7977164c24923a979e844dbaf5`
- model SHA: 対象外

## Submit-check

- sample row count: PASS
- header: PASS
- id order: PASS
- duplicate IDs: 0
- missing / non-finite tvt: 0
- prediction min / max / mean / std: 11591.730469 / 12239.677734 / 11905.696063 / 279.370888
- exp073 inference v2 との差分: RMSE 0.984292、p95 abs diff 1.953125、max diff 6.015625

## 次

1. exp096 は完了。
2. fade-in projection 単独では anchor 更新なし。別仮説へ進む。
