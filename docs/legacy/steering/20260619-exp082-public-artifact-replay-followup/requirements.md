# 要件

## 依頼

`public_artifact_replay_followup` を実装する。`exp079_public_artifact_replay_integrity_audit` で placeholder のまま残った SP45 / fle3n / Koolbox 系公開 notebook について、exact source slug を固定し、同じ no-submit integrity audit を再実行できる実験を用意する。

## 制約

- Route: `pf_beam`
- 親実験: `exp079_public_artifact_replay_integrity_audit`
- Kaggle Notebook 実行を正とする。ローカル notebook 実行はしない。
- 直接 submit はしない。候補 CSV 本体の sample 互換、SHA、pairwise distance、source risk を記録してから別途判断する。
- 再現性: `docs/06_reproducibility.md` に従い、input SHA、prediction/submission SHA、Kaggle kernel version を audit output と記録に残す。

## 受け入れ基準

- `experiments/exp082_public_artifact_replay_followup/` が作成され、`config.yaml` の `experiment.route` が `pf_beam` になっている。
- `runtime.kaggle.kernel_sources` / `dataset_sources` に SP45 / fle3n / Koolbox / SP45-Fleongg blend の exact source が入っている。
- `audit.source_specs` が各候補の required input slug、branch file、expected check を持つ。
- `.ipynb` と `.py` source の static CSV writer / hardcoded submission / public-visible branch risk を inspect できる。
- `task validate-exp` 相当と Kaggle notebook preparation が通る。
