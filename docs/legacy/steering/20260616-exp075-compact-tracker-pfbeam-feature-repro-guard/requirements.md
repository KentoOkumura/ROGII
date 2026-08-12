# 要件

## 依頼

`compact_tracker_pfbeam_feature_repro_guard` を実装する。PF/Beam 再生 notebook は後続実験で使い回せるように LightGBM 学習 notebook から分離する。LightGBM 学習では特徴量重要度を matplotlib でプロットする。

## 制約

- Route: `ml_model`
- train データから compact PF/Beam/likelihood-PF tracker features を生成する notebook を別に作る。
- LightGBM train notebook は生成済み `ravaghi_vs_pixiux_public_replay_tracker_features_train.csv.gz` 相当を読み込む。
- train feature CSV 相当の 2 回再生成による再現性確認は行わない。
- inference は再現性担保済みの PF/Beam 生成方法で raw test から feature を生成し、保存済み LightGBM booster で推論する。
- stochastic feature generation、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録は `docs/06_reproducibility.md` に沿って扱う。
- gzip 生成物の再現性証拠は raw gzip SHA ではなく decompressed CSV content SHA を主証拠にする。

## 受け入れ基準

- exp075 実験フォルダ、config、train/inference notebook、PF/Beam feature notebook、記録ファイルが作成されている。
- PF/Beam feature notebook が raw train から reusable tracker feature CSV を 1 回生成する構成になっている。
- train notebook が生成済み tracker feature CSV を読み、LightGBM を学習し、feature importance CSV と matplotlib PNG を保存する。
- inference notebook が raw test から tracker feature を再生成し、保存済み LightGBM booster で `submission.csv` を作る。
- `validate_experiment.py`、Python compile、notebook JSON validation が通る。
