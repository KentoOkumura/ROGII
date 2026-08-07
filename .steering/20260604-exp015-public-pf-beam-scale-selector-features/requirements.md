# 要件

## 依頼

`KAGGLE_DIRECTION.md` のアイデアバックログ先頭 `public_pf_beam_scale_selector_features` を実装し、Kaggle 上で full CV を実行する。

目的は、公開 notebook 系で強い PF / beam / hold blend route をそのまま提出向け予測として移植する前に、fold-safe な OOF feature として小さく再現し、`exp012/exp013` の LightGBM no-GR raw anchor と比較すること。

## 制約

- 親実験は `exp013_model_diversity_or_postprocess` とし、raw anchor は `lightgbm_no_gr` 13.549257 を固定 control にする。
- valid fold の `TVT`、`TVT_input` hidden tail、train-only formation columns は PF/beam feature 生成に使わない。
- PF/beam snapshot は 見えない test well 推論 で利用可能な `MD`、`Z`、`GR`、known `TVT_input` prefix、paired typewell GR だけから作る。
- まず runtime を抑えた scale/candidate 数で full CV を回し、PF 128/256 seeds 相当の重い探索は次実験に回す。
- score 記録では raw CV、PF/beam feature 追加 CV、postprocess OOF-fit/held-out 値を混同しない。

## 受け入れ基準

- `.steering`、`experiments/exp015_public_pf_beam_scale_selector_features/`、`config.yaml`、train/inference notebook、`SESSION_NOTES.md` が整っている。
- `task validate-exp EXP=exp015_public_pf_beam_scale_selector_features` が通る。
- Kaggle train notebook を `--run-on-push --strict` で準備し、Kaggle full CV を push する。
- Kaggle full CV の完了後、`metrics.json`、主要 artifacts、`SESSION_NOTES.md`、`result.md`、`experiment_summary.md` を更新する。
