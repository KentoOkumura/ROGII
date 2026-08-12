# 要件

## 依頼

`public_sel15_pf_oof_multicutoff_artifact` を実装する。

`exp029_public_sel15_pf_oof_feature_generation` の public sel15 PF/Beam OOF-like 生成処理を、cutoff 0.65 単独ではなく `[0.45, 0.65, 0.82]` で実行できる artifact 生成実験にする。目的は exp055 で不足していた multi-cutoff public feature surface を用意し、後続の pseudo-tail / single-model 再訪で `0.65 control only` と `0.45/0.82 augmentation` を分けて比較できる状態にすること。

## 制約

- Route: `pf_beam`
- 親実験は `exp029_public_sel15_pf_oof_feature_generation` とする。
- PF/Beam 入力は hidden test でも使える `MD`, `X`, `Y`, `Z`, `GR`, typewell `TVT/GR`, cutoff 以前の `TVT_input` に限定する。
- cutoff 以降の true `TVT` は target / diagnostic 専用で、PF/Beam 生成には使わない。
- 既存 0.65 artifact を置き換えるだけではなく、0.65 rows を残したまま 0.45 / 0.82 rows を追加できる schema にする。
- 生成物作成だけでは採用根拠にしない。下流比較は別実験で行う。

## 受け入れ基準

- `config.yaml` で `model.cutoff_fractions: [0.45, 0.65, 0.82]` を正の run config として表現できる。
- train notebook は multicutoff artifact generation の目的、設定確認、実行、生成物確認を人間が追える構成になっている。
- 生成スクリプトは複数 cutoff を同じ CSV に保存し、`pseudo_cutoff_fraction` / `cutoff_row` / `row_idx` で downstream loader が識別できる。
- well summary は cutoff ごとの rows / PF diagnostic / beam diagnostic を記録する。
- local smoke で少なくとも 1 well x 3 cutoffs が通り、出力に 3 cutoff が含まれることを確認する。
- `task validate-exp` と静的チェックが通る。
