# 要件

## 依頼

`KAGGLE_DIRECTION.md` の `exp073_exp092_pf_beam_pseudotail_failure_map` を実験化する。

旧 `pf_beam_pseudo_tail_rule_audit` は exp027 / public sel15 hidden branch の診断に寄っていたため、現在の ML route 改善に使えるよう、主対象を `exp073` raw deterministic ML anchor と `exp092` submitted ML anchor に変更する。

## 制約

- Route: `ml_model`
- 提出なし。train-side diagnostic のみ。
- `exp027` は旧 PF route / public replay の固定比較基準としてだけ扱い、実装対象や hidden branch 再チューニング対象にしない。
- `exp063` は系譜上の起点、`exp072` は PF/Beam/likPF train pseudo-tail feature cache の入力として扱う。
- `exp073` / `exp092` の OOF prediction と `exp072` feature cache を join し、target は scoring / oracle coverage 診断だけに使う。
- 再現性は `docs/06_reproducibility.md` に従う。新規 RNG / 学習 / 推論 / submission は発生させない。

## 受け入れ基準

- `experiments/exp126_exp073_exp092_pf_beam_pseudotail_failure_map/` が作成されている。
- `config.yaml` に route、親実験、入力 kernel source、再現性方針、生成物名が明記されている。
- train notebook が setup、入力確認、診断実行、生成物 preview のセル構成になっている。
- 補助モジュールが次を生成する。
  - row-level failure map
  - candidate metrics
  - bucket metrics
  - well metrics / path continuity proxy
  - summary JSON
- inference notebook は no-op で、submission を作らないことが明示されている。
- `make validate-exp EXP=exp126_exp073_exp092_pf_beam_pseudotail_failure_map` が通る。
