# 要件

## 依頼

KAGGLE_DIRECTION の `cluster_outlier_prior_confidence_addonly_on_exp158_selector` backlog を実装する。

## 制約

- Route: `pf_beam`
- 親実験: `exp158_segment_continuity_selector_on_exp157`
- 補助入力: `exp181_cluster_outlier_pfbeam_prior_gate`、`exp109_typewell_neighbor_prior_features`、`exp114_spatial_neighbor_prior_signal_audit`、`exp175_cluster_outlier_typewell_prior_gate`
- 既存の 8候補と exp158 の continuity / Viterbi selector を維持する。
- 初手では既存 confidence feature を置換しない。cluster-outlier prior signal は add-only feature として扱う。
- exp181 の c20/c40 correction は feature としてのみ使い、予測へ直接加算しない。
- direct posthoc correction、candidate hard replacement、inference port、submit は対象外。
- valid/test true TVT、oracle best、true-error rank、OOF absolute error を feature source に漏らさない。
- 再現性: `docs/06_reproducibility.md` に従い、固定 seed、input/model/prediction SHA、Kaggle bootstrap 整合を記録する。

## 受け入れ基準

- exp183 用の `config.yaml`、実装 module、train/inference notebook source が作成されている。
- 追加 feature が cluster-outlier flags、typewell/spatial prior delta、prior std/count/neighbor、gate flag、well gate ratio、candidate family interaction、c20/c40 correction magnitude に限定されている。
- Kaggle train push 前の予定として、active variant 1、LightGBM config 3、fold 5、合計 booster 15、control / parent 再学習なしが `SESSION_NOTES.md` に記録されている。
- train notebook は OOF metrics、path switch、worst well、distance bucket、exp115 hidden-like subgroup、feature importance、score summary を生成物として保存する。
- Jupytext 変換、構文チェック、ruff F821/F401/E501、`validate-exp` が通る。
