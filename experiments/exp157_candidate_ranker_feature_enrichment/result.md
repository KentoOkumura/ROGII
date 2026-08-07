# exp157_candidate_ranker_feature_enrichment 結果

## 状態

Kaggle train v1 完了。Kernel は `kentookumura/exp157-cand-ranker-enrich-train` version 1、output は `experiments/exp157_candidate_ranker_feature_enrichment/kaggle/output/train_v1`。

## 仮説

exp101 の row-wise supervised ranker は `pf_ancc` を選べるようになったが、`likpf_mean` 単体を超えず path switch も多かった。候補集合に `tvt_dense` family を追加し、dense disagreement / drift / continuity 系の target-free feature を加えることで、PF/Beam/dense のどれを信用すべきかを selector がよりよく判断できるか確認した。

## 実行

- Runtime: Kaggle CPU (`enable_gpu=false`)
- 入力: exp099 v2 multiobs cache + exp072 full replay feature cache
- rows / wells: 3,783,989 / 773
- candidates: 8 (`pf_ancc`, `beam_mean`, `likpf_mean`, `sc_ens`, `hyb`, `tvt_dense`, `tvt_densew`, `tvt_dense50`)
- feature count: 97
- generated dense enrichment features: 23
- LightGBM: 3 family x 5 folds = 15 boosters
- control retraining: なし
- runtime: 10,421.758 sec

## 結果

| variant | mode | RMSE | MAE | within10 | oracle label acc |
| --- | --- | ---: | ---: | ---: | ---: |
| oracle | oracle | 4.564605 | 2.317166 | 0.960054 | 1.000000 |
| lgb_candidate_error_ranker | oof | 10.795800 | 6.476996 | 0.792505 | 0.258688 |
| lgb_candidate_binary | oof | 11.072650 | 6.753211 | 0.774043 | 0.310873 |
| lgb_multiclass | oof | 11.440172 | 6.900257 | 0.769262 | 0.305335 |
| likpf_mean_single | baseline | 11.594898 | 7.067633 | 0.772807 | 0.263997 |

best OOF は `lgb_candidate_error_ranker`。`likpf_mean_single` から RMSE -0.799098、within10 +0.019697。exp101 best OOF RMSE 11.600097 からも -0.804297 改善した。

## 分布とリスク

`lgb_candidate_error_ranker` の選択率は `likpf_mean` 37.94%、`pf_ancc` 37.14%、`tvt_densew` 13.07%、`beam_mean` 4.82%、`tvt_dense50` 2.77%、`tvt_dense` 2.34%。dense family 合計は 18.18%。

bucket でも改善は広い。near `000_050` は RMSE 1.188878 -> 0.499500、`1000_plus` は 12.704015 -> 11.850103、`pf_seed_std_q4` は 11.071060 -> 10.531303。

一方で path switch はまだ大きい。best OOF の max path switch は 357.199 / 1000 rows、worst well は `86454a6f` で RMSE 57.967201。row-wise selector のまま inference port / submit するには不安定。

## SHA

- exp099 source decompressed SHA: `1939d536b1e56f7c0ea3847cc386ef769b0d33759d16e816c9ce180f0532df9a`
- exp072 auxiliary source decompressed SHA: `99a3c70a19b2f22a8c76c5947b14692e7b8207ea45f38b8b1c5f327d320e1350`
- feature schema SHA: `891226fdf0f82c384e2fcca77f3c7d47b964d5837251ce9594249951d4e5b87c`
- model manifest SHA: `ab25fbfc0c8b92915bfbd11e62c8ffa6d84eadb3d8abf10e039927e2df7d4fb1`
- predictions decompressed SHA: `8e24ea7a55ec88360fd9245b34501fc3fc3adb868a335a93b37e8194b66dc2f0`

## 結論

train-side では supported。`candidate_ranker_feature_enrichment` は、dense 候補を selector に入れる価値があることを示した。ただし row-wise switch が大きいため、direct inference port / submit はしない。次は exp157 score surface を使った segment / Viterbi continuity selector、または confidence-gated fallback を確認する。
