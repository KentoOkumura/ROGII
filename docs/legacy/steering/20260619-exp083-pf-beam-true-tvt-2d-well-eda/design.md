# 設計

## アプローチ

`exp072_exp063_full_replay_feature_cache` の `exp063_full_replay_feature_cache_pixiux_likpf_public_replay_train_features.csv.gz` を読む。これは現 anchor `exp073` の train 入力で、`id`, `well`, `target`, `last_known_tvt` と 196 feature を持つ。

EDA では `true_tvt = last_known_tvt + target` を作り、PF/Beam/likelihood-PF 候補を TVT 空間へ戻す。

- absolute 候補: `pf_ancc`, `pf_z`, `tvtF_ANCC`
- delta 候補: `beam_*_d`, `sc_ens_d`, `hyb_d`, `likpf_mean_d`, `tvt_dense_d`
- confidence / context: `pf_ancc_std`, `beam_std_d`, `sig_std`, `sc_trust`, `pfx_rmse`, `known_len`, `eval_len`

代表 well は best/worst PF RMSE、高 PF-vs-Beam 不一致、Beam が PF より良い例、anchor が PF より良い例、long tail、stable random で選ぶ。

## 実験範囲

- 対象実験: `exp083_pf_beam_true_tvt_2d_well_eda`
- Route: `pf_beam`
- 親実験: `exp072_exp063_full_replay_feature_cache`
- 固定する変数: exp072 v2 deterministic feature cache、exp073 が使う feature surface
- 変更する変数: なし。可視化と集計だけ。

## 再現性設計

- seed policy: no RNG for feature generation。代表 random 抽出だけ config seed 42。
- stochastic 処理の有無: なし。
- PF/Beam / likelihood-PF / seed bagging の有無: 既存 exp072 生成物を読むだけ。新規生成しない。
- 並列処理と乱数の関係: なし。
- CPU/GPU runtime: CPU only。GPU なし。
- train cache SHA: gzip raw SHA と decompressed content SHA を summary JSON に記録する。
- model manifest / submission SHA: 対象外。
- Kaggle package bootstrap: `prepare_kaggle_notebooks --notebook train --run-on-push --strict` で exp072 kernel source が metadata に入ることを確認する。

## リスク

- リークリスク: true TVT を見る EDA なので、直接 router や置換ルールには使わない。
- CV/LB 不一致リスク: スコア改善を主張しない。失敗地図の入力に限定する。
- ランタイム/メモリリスク: exp072 cache は大きい。集計後、PNG は代表 well のみ保存する。
- ローカル検証リスク: ローカルには exp072 の大きな gzip がないため、合成 CSV で変換ロジックだけ確認し、正実行は Kaggle とする。
