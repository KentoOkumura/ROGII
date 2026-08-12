# 設計

## アプローチ

exp073 の fold-out OOF prediction を固定 anchor とし、`target_tvt - exp073_pred_tvt` を軽量 sequence model の correction target にする。well 内の MD 順 window を作り、小さな GRU と TCN で validation fold の well を完全に外して補正を学習する。

出力は次の診断に限定する。

- GRU / TCN 単体 OOF RMSE
- exp073 baseline error との相関
- prediction diff / correction magnitude
- distance bucket 別 RMSE
- exp073 + alpha correction blend
- baseline / GRU / TCN の ridge blend

## 実験範囲

- 対象実験: `exp088_sequence_model_residual_diversity`
- Route: `ml_model`
- 親実験: `exp073_gpu_reproducibility_guard_for_exp063_full_replay`
- cache 親: `exp072_exp063_full_replay_feature_cache`
- 変更する変数: residual correction model family を LightGBM から lightweight GRU / TCN に変える。
- 固定する変数: exp073 OOF anchor、exp072 feature surface、score rows、well-level validation、PF/Beam feature cache。

## 再現性設計

- seed policy: `validation.seed`、variant 名、valid fold を `blake2b` で stable seed 化する。
- stochastic 処理の有無: PyTorch initialization、DataLoader shuffle、train window subsampling が stochastic。
- PF/Beam / likelihood-PF / seed bagging の有無: 新規生成なし。exp072 cache を読むだけ。
- 並列処理と乱数の関係: DataLoader `num_workers=0`、local RNG / torch generator を fold ごとに固定する。
- CPU/GPU runtime と deterministic flags: Kaggle GPU 有効、torch float32 固定、AMP/bfloat16 無効、CuDNN deterministic flag を設定する。
- train cache / test feature regeneration の SHA 記録方針: exp072 feature cache と exp073 OOF prediction は decompressed content SHA を summary JSON に保存する。
- model manifest / prediction / submission SHA 記録方針: submission はなし。OOF prediction decompressed content SHA と metrics SHA を保存する。モデル weight は保存しない。
- Kaggle package bootstrap 確認方針: `prepare_kaggle_notebooks --strict` 後に metadata と notebook JSON を検証する。

## リスク

- リークリスク: exp073 OOF prediction は fold-out だが、sequence model の fold が exp073 fold と一致しない可能性がある。目的は OOF residual diversity 診断であり、anchor 更新には使わない。
- CV/LB 不一致リスク: inference を作らないため LB 判断はしない。改善しても別実験で inference parity を設計する。
- ランタイム/メモリリスク: 3.8M rows の window dataset は重い。`max_train_windows_per_fold=180000` と軽量 hidden size 32 で初回を抑える。
- 再現性リスク: GPU NN は bitwise deterministic anchor として扱わない。summary に seed、runtime、input/output SHA を残す。
