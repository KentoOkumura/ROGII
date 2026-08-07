# 設計

## アプローチ

exp099 train v2 の fixed candidate cache を読み、`pf_ancc`、`beam_mean`、`likpf_mean`、`sc_ens`、`hyb` を candidate-long 化する。各 candidate について、candidate TVT 周辺の typewell GR window と、評価 row 周辺の horizontal GR window を 2D image にし、`typewell GR`、`horizontal GR`、`差分`、`observed TVT_input prefix SDF`、`observed mask` の 5 channel を CNN に入力する。出力は candidate が true TVT から 10ft 以内かの probability と expected absolute error。

比較は直接 replacement ではなく frozen scorer として行う。learned probability / predicted error の candidate AUC、topK coverage、top1 RMSE、soft weighted prediction RMSE、effective sample size、candidate switch rate、worst-well を、point-GR likelihood、exp099 multi-observation score、exp111 learned likelihood、likPF baseline と比較する。real-GR と同じ sampling schedule で shuffled-GR / no-GR variants も学習し、real signal が negative control を上回るか確認する。

## 実験範囲

- 対象実験: `exp197_cnn_pf_likelihood_probe`
- Route: `pf_beam`
- 親実験: backlog `cnn_pf_likelihood_probe`
- cache 親: `exp099_pf_multi_observation_likelihood_probe`
- 比較親: `exp111_learned_pf_observation_likelihood_probe` / `exp112_learned_pf_likelihood_weight_or_feature_followup`
- CNN 参照: `exp179_cnn_sdf_mtp_heatmap_probe` / `exp182_cnn_sdf_mtp_heatmap_fullfold_geometry_probe`
- 変更する変数: local CNN/SDF likelihood scorer の導入、real/shuffled/no-GR control、candidate weighted diagnostics。
- 固定する変数: exp099 fixed candidate set、GroupKFold by well、train-side pseudo-tail score rows、PF/Beam 生成物、直接 submit しない方針。

## 再現性設計

- seed policy: global seed 固定、row subsample と shuffled-GR roll は SHA256 stable key で決める。
- stochastic 処理の有無: PyTorch CUDA conv、AdamW、DataLoader shuffle がある。deterministic anchor とは扱わない。
- PF/Beam / likelihood-PF / seed bagging の有無: 新規 PF/Beam は実行しない。exp099 upstream cache に含まれる PF/Beam/likPF 候補だけを読む。
- 並列処理と乱数の関係: `num_workers=0`。並列 worker 内 RNG は使わない。
- CPU/GPU runtime と deterministic flags: Kaggle T4 GPU 前提。`torch.use_deterministic_algorithms(True, warn_only=True)`、CuDNN benchmark off を設定する。
- train cache / test feature regeneration の SHA 記録方針: exp099 cache / schema SHA、candidate index gzip raw/decompressed SHA、OOF likelihood gzip raw/decompressed SHA を summary に記録する。raw-test は生成しない。
- model manifest / prediction / submission SHA 記録方針: variant ごとの model SHA、manifest SHA、OOF probability / expected error content SHA を記録する。submission SHA は対象外。
- Kaggle package bootstrap 確認方針: push 前に `make validate-exp` と `make prepare-kaggle-notebooks ... --strict` で metadata と bootstrap config を確認する。

## リスク

- リークリスク: true TVT を candidate window center に使うと即リークになるため禁止。candidate TVT と observed TVT_input prefix だけを window に使う。
- CV/LB 不一致リスク: train pseudo-tail の GR repeat / local match が hidden test に一般化しない可能性が高い。submit せず raw-test parity follow-up が必要。
- ランタイム/メモリリスク: full candidate-long image 化は大きすぎるため、fold0 の deterministic row subsample を使う。GPU cost は active variants 3、fold 1、model 3。
- 再現性リスク: GPU training は bitwise deterministic anchor ではない。結果は train-side diagnostic evidence としてのみ扱う。
