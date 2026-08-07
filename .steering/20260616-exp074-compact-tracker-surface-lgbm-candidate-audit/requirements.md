# 要件

## 依頼

`compact_tracker_surface_lgbm_candidate_audit` を実装し、Kaggle 上で実行する。

## 制約

- Route: `ml_model`
- exp070 の 65-feature compact tracker surface を、exp063 full replay reproducibility guard ではなく LB 候補として扱う。
- 追加 GPU train は 1 回に制限する。
- train は exp063 の保存済み compact tracker train artifact を固定入力にする。
- inference は保存済み exp063 test feature CSV を使わず、current raw test から PF/Beam/likelihood-PF compact features を再生成する。
- Kaggle CLI はネットワーク許可付きで実行する。
- stochastic feature generation、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録は `docs/06_reproducibility.md` に沿って記録する。

## 受け入れ基準

- exp074 実験フォルダ、config、train/inference notebook、記録ファイルが作成されている。
- `validate_experiment.py` が通る。
- Kaggle train package が生成され、push 後に実行されている。
- train 完了後に Kaggle inference package が生成され、push 後に実行されている。
- output 取得後、feature source SHA、model manifest、prediction SHA、submission SHA、Kaggle kernel version を記録する。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録する。
