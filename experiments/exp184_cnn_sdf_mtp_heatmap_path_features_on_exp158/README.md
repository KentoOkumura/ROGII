# exp184_cnn_sdf_mtp_heatmap_path_features_on_exp158

## 状態

実装済み。Kaggle train は未実行。提出なし。

## 仮説

exp182 の CNN/SDF/MTP heatmap は full-fold でも real GR signal を確認できたが、worst-well top3 が 0.0 のため direct TVT replacement には使わない。exp158 の continuity selector に heatmap topK path center、score margin、entropy、real-vs-shuffled/no-GR gap、候補との差分を add-only feature として渡すことで、PF/Beam/dense 候補の信頼度をより正しく読める可能性がある。

## 検証方針

exp157 と同じ 8候補、dense enrichment、GroupKFold、LightGBM 3 configs、exp158 Viterbi grid を維持する。exp182 `base_real_w128_b64_fullfold` validation predictions を主入力にし、`base_shuffled_w128_b64_fullfold` / `base_no_gr_w128_b64_fullfold` は confidence gap feature に限定する。sparse sample は well 内 row index で補間して exp158 の全 selector row に展開する。

比較基準は `likpf_mean` RMSE 11.594897672、exp157 row-wise RMSE 10.795799837、exp158 continuity RMSE 10.789163253、exp148 `lgb_mean` RMSE 8.501281182。

## 実行予定

- active selector variant: 1
- LightGBM configs: 3
- folds: 5
- planned boosters: 15
- control / parent retraining: なし
- runtime: Kaggle CPU

## 判定基準

global OOF が exp158 continuity を上回り、path switch、worst-well regression、near-row、distance bucket、heatmap confidence bucket、exp115 hidden-like subgroup が壊れていない場合だけ follow-up を検討する。positive でも heatmap direct replacement / inference port / submit はこの実験では行わない。

## 所見

現時点では実装のみ完了。ローカルには exp072 dense train cache 本体がないため、full feature assembly は Kaggle kernel source 上で確認する。Kaggle train 後に、exp158 continuity との差分、heatmap confidence bucket、feature importance、path switch を確認して採否を判断する。
