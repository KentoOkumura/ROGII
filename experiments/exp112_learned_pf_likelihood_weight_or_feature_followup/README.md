# exp112_learned_pf_likelihood_weight_or_feature_followup

## 状態

Kaggle train v1 完了。提出なし。

## 仮説

exp111 の learned likelihood は candidate-level AUC と topK coverage を改善したが、top1 direct replacement は `likpf_mean` 単体より within10 が悪かった。

そのため、今回は learned top1 を提出候補にせず、次の用途だけを検証する。

- target-free multiobs score に learned likelihood を弱く加える PF weight alpha
- `likpf_mean` を default にした conservative verifier gate
- exp092 系 ML add-only audit に渡せる learned likelihood feature cache

## 検証方針

- 入力: exp111 OOF likelihood long cache、exp099 v2 wide cache
- 候補: `pf_ancc`, `beam_mean`, `likpf_mean`, `sc_ens`, `hyb`
- 評価: exp111 first-fold held-out wells の train-side posthoc audit
- 主指標: RMSE、MAE、within10、switch rate、worst-well、bucket metrics
- 出力: metrics、OOF prediction、ML feature cache、SHA 付き summary

## 所見

best non-oracle は `gate_expected_error_m2p0_d20p0` で、`likpf_mean_single` RMSE 11.604410 / within10 0.784312 から RMSE 11.573266 / within10 0.785064 へ小改善した。switch rate は 0.4077%。

PF weight alpha は全 variant で大きく悪化したため不採用。ML feature cache は target-free column のみで保存済み。

## 注意

この実験は raw-test parity を作らない。`submission.csv` は生成せず、learned top1 / PF weight / verifier gate をそのまま hidden test prediction に使わない。
