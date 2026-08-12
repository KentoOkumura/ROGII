# 要件

## 依頼

`KAGGLE_DIRECTION.md` の高優先 backlog `exp092_exp098_small_rank_slot_merge` を、最新番号 `exp139_exp092_exp098_small_rank_slot_merge` として実装する。

## 制約

- Route: `ml_model`
- 親実験は `exp092_u_projection_correction_disagreement_fullrun` とし、exp098 は rank-slot feature source として扱う。
- exp092 の U-projection correction / disagreement feature surface、target、GroupKFold、LightGBM family は維持する。
- exp098 full 64 rank-slot 列の一括 union はしない。初回は代表的な 10-15 列程度の small add-only variant に限定する。
- PF/Beam candidate TVT path を direct selector、soft average、postprocess replacement として使わない。
- rank-slot ordering は target-free score のみで作り、評価区間 true TVT を rank 生成に使わない。
- 再現性: `docs/06_reproducibility.md` に従い、PF/Beam cache、GPU LightGBM、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。

## 受け入れ基準

- `experiments/exp139_exp092_exp098_small_rank_slot_merge/` に config、settings、train/inference notebook、補助 `.py`、README、SESSION_NOTES、result、metrics がある。
- `config.yaml` に `experiment.route: ml_model`、親実験、rank-slot source parent、small merge の列、比較対象が記録されている。
- train notebook は設定確認、入力確認、学習実行、metrics / projection summary / rank-slot summary / feature importance / manifest 確認をセル単位で追える。
- inference notebook は同じ U-projection と rank-slot feature generation を使うが、train-side review 後に使う前提として記録されている。
- `make validate-exp EXP=exp139_exp092_exp098_small_rank_slot_merge` が通る。
- Kaggle push 前に metadata と bootstrap 内 config の整合を確認できる状態になっている。
- deterministic anchor として扱う場合は、feature content SHA、model SHA、prediction SHA、submission SHA、Kaggle kernel version が記録されている。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
