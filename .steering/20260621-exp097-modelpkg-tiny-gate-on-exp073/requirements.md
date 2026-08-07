# Requirements: exp097_modelpkg_tiny_gate_on_exp073

## Goal

exp073 deterministic ML inference prediction に、Pilkwang model-package-only prediction を agreement-gated tiny correction として足す候補を実装する。

## Scope

- exp073 inference artifact を base prediction として読む。
- `submission_model_package_only.csv` を model-package prediction として読む。
- `g = gmax / (1 + (abs(pkg-base)/scale)^2)` を適用する。
- `gmax` 0.003 / 0.005 / 0.010、`scale` 4 / 5 / 8 の grid を保存する。
- selected candidate は `gmax=0.005, scale=4.0`。
- raw diff p95、correction p95、correction max の guard を通った場合だけ inference notebook で `submission.csv` を書く。

## Non-Goals

- model package の特徴量生成やモデル推論を dataset から直接再実装しない。
- model-package-only を exp073 の直接置換として採用しない。
- Public LB だけで gmax / scale を細かく探索しない。
- OOF surrogate なしに CV 改善として記録しない。

## Acceptance

- `config.yaml` に route、lineage、grid、guard、入力候補 path が記録されている。
- train / inference notebook が Kaggle notebook として準備できる構成になっている。
- 実装は selected candidate と全 grid の summary / CSV を保存する。
- `submission.csv` は guard failure 時に書かれない。
