# 要件

## 依頼

`KAGGLE_DIRECTION.md` の backlog `exp092_exp098_rank_slot_replacement_only` を、最新番号 `exp147_exp092_exp098_rank_slot_replacement_only` として実装する。

## 制約

- Route: `ml_model`
- 親実験は `exp092_u_projection_correction_disagreement_fullrun` とし、exp098 は rank-slot feature source として扱う。
- exp073/exp072 由来の base 196 features は残す。
- exp092 側の U-projection correction / disagreement 生成列のうち、rank-slot と意味が近い列だけを落とす。
- exp098 予測、OOF 予測、blend / stack、candidate TVT path の direct selector、soft average、postprocess replacement は行わない。
- add-only と replacement-only を混ぜない。落とした overlap 列と rank-slot replacement 列を同時にモデルへ渡さない。
- rank-slot ordering は target-free score のみで作り、評価区間 true TVT、oracle best、true-error rank を特徴量生成や列選択に使わない。
- 再現性: `docs/06_reproducibility.md` に従い、PF/Beam cache、GPU LightGBM、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。

## 受け入れ基準

- `experiments/exp147_exp092_exp098_rank_slot_replacement_only/` に config、settings、train/inference notebook、補助 `.py`、README、SESSION_NOTES、result、metrics がある。
- `config.yaml` に `experiment.route: ml_model`、親実験、rank-slot source parent、drop columns、replacement columns、比較対象が記録されている。
- train notebook は設定確認、入力確認、学習実行、metrics / projection summary / rank-slot summary / feature importance / manifest 確認をセル単位で追える。
- inference notebook は同じ U-projection と rank-slot feature generation を使うが、train-side review 後に使う前提として記録されている。
- `make validate-exp EXP=exp147_exp092_exp098_rank_slot_replacement_only` が通る。
- Kaggle train push 前に、active variant 1、LightGBM config 3、fold 5、合計 booster 15、control 再学習なしを `SESSION_NOTES.md` に記録している。
- deterministic anchor として扱う場合は、feature content SHA、model SHA、prediction SHA、submission SHA、Kaggle kernel version が記録されている。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
