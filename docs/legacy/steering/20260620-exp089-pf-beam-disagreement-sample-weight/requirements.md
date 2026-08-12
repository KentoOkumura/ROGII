# 要件

## 依頼

`pf_beam_disagreement_sample_weight` を実装する。

## 制約

- Route: `ml_model`
- 親実験は `exp073_gpu_reproducibility_guard_for_exp063_full_replay`、train feature cache は `exp072_exp063_full_replay_feature_cache` を使う。
- target は exp073 と同じ `TVT - last_known_tvt` に固定し、PF/Beam 予測を教師、直接置換、hard router、提出枝として使わない。
- 追加する信号は target-free な PF/Beam/likelihood-PF disagreement と density / distance diagnostics に限定する。
- 比較は小さく保つ。`control`、confidence feature add-only、sample-weight only、feature+weight の4 variantを基本にする。
- CV 改善だけで submit 候補にしない。Kaggle train 後に worst-well、distance bucket、tail bucket、well-hash 追加確認の要否を見る。
- 再現性は `docs/06_reproducibility.md` に従い、入力 cache SHA、model SHA、OOF prediction SHA、Kaggle package bootstrap の整合を記録する。

## 受け入れ基準

- `experiments/exp089_pf_beam_disagreement_sample_weight/` に config、train notebook、補助 `.py`、README、result、metrics、SESSION_NOTES がある。
- `config.yaml` に route、lineage、validation、confidence feature、sample weight policy、active variants、expected artifacts が明記されている。
- train notebook が exp072 cache preview、variant 実行、metrics / by-well / bucket / importance / weight summary 表示を含む。
- 補助 `.py` が confidence features と sample weights を fold label / target を使わずに生成し、LightGBM の train fold にだけ `sample_weight` を渡す。
- `py_compile`、notebook JSON validation、ruff、`validate_experiment.py` が通る。
