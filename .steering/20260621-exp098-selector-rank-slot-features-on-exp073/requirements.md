# 要件

## 依頼

`selector_rank_slot_features_on_exp073` を実装する。

## 制約

- Route: `ml_model`
- 親実験は `exp073_gpu_reproducibility_guard_for_exp063_full_replay`、特徴 cache は `exp072_exp063_full_replay_feature_cache` に固定する。
- PF/Beam 候補を直接 selector / soft average / postprocess replacement として使わない。
- rank slot は target-free score だけで作り、評価区間 true TVT は rank 生成に使わない。
- 既存の exp073 target、GroupKFold by well、LightGBM family を維持する。
- Kaggle Notebook 実行を正とし、ローカル notebook 実行はしない。
- 再現性: `docs/06_reproducibility.md` に従い、upstream cache SHA、model manifest、prediction SHA を記録する。

## 受け入れ基準

- `.steering/20260621-exp098-selector-rank-slot-features-on-exp073/` に要求、設計、タスクが記録されている。
- `experiments/exp098_selector_rank_slot_features_on_exp073/` に config、train notebook、補助 `.py`、README、SESSION_NOTES、result、metrics scaffold がある。
- train notebook は設定、候補 plan、variant plan、学習実行、metrics/生成物保存をセル単位で追える。
- `rank_slot_u_disagreement` のみを同一 exp073 GroupKFold 条件で学習できる。
- 特徴量重要度の CSV、平均 CSV、上位特徴量プロットが保存される。
- `make validate-exp EXP=exp098_selector_rank_slot_features_on_exp073` が通る。
- Kaggle push 前の package 生成が strict mode で通る。
