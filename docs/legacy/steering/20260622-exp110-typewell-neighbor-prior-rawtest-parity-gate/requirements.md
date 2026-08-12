# 要件

## 依頼

`typewell_neighbor_prior_rawtest_parity_gate` を実装する。exp109 best neighbor correction を固定し、raw-test 互換の full-train prior parity と worst-well gate を検証できる train-side 実験にする。

## 制約

- Route: ensemble
- 親実験: `exp109_typewell_neighbor_prior_features`
- exp109 best `native_overlap_0p999_likpf_mean_corr_a0p2_c40` の correction 設定を固定する。
- gate の CV は fold-safe OOF prior のみで評価する。
- full-train-source prior は inference-compatible parity 診断として保存し、primary CV score として扱わない。
- validation/test true TVT を prior source に入れない。
- test-side PF/likPF surface がない段階では `submission.csv` を生成しない。
- 再現性: `docs/06_reproducibility.md` に従い、SHA 記録と gzip decompressed SHA を summary に残す。

## 受け入れ基準

- exp110 実験フォルダ、config、train/inference notebooks、helper、SESSION_NOTES、result、metrics が exp110 名で整合している。
- OOF prior と full-train-source prior を別列で生成する。
- prior parity metrics を CSV と summary JSON に保存する。
- gate grid metrics を CSV と summary JSON に保存する。
- best gate だけを OOF prediction artifact に保存する。
- inference notebook は no-op guard として、submission を生成しない理由を表示する。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
