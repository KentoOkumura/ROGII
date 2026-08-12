# 要件

## 依頼

`exp153_full_rank_slot_addonly_on_exp092` を Colab 前提で実装する。

## 制約

- Route: `ml_model`
- 親実験は `exp092_u_projection_correction_disagreement_fullrun` とし、exp092 control は再学習しない。
- rank-slot source parent は `exp098_selector_rank_slot_features_on_exp073` とする。
- exp072 deterministic full replay train cache を入力に使い、Colab では DriveFS 直読みではなく `/content` にコピーしてから学習する。
- 追加特徴は target-free rank-slot feature groups の add-only に限定する。
- candidate TVT path の direct selector、soft average、blend、postprocess replacement、target 変更は入れない。
- 再現性は `docs/06_reproducibility.md` に従い、upstream PF/Beam cache、GPU LightGBM、SHA 記録、Colab runtime log を区別して扱う。

## 受け入れ基準

- `docs/legacy/steering/`、`config.yaml`、`README.md`、`SESSION_NOTES.md`、`result.md`、`metrics.json` が今回の仮説と未実行状態を正しく示す。
- train notebook は Kaggle 正本として読める構成を保ち、full rank-slot feature groups を config から実行できる。
- Colab train notebook は Drive mount、layout validation、cache copy、LightGBM GPU smoke、background full train、Drive-backed log/status check を含む。
- GPU コストガードとして active variant 1、LightGBM config 3、fold 5、合計 booster 15、control 再学習なしを記録する。
- `make validate-exp` と Python/JSON の静的検証が通る。
