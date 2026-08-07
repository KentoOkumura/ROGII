# 要件

## 依頼

`KAGGLE_DIRECTION.md` のアイデアバックログ最上位「exp004 の Public LB 乖離を踏まえ、GR coverage gate を見直す」を個別実験として実装する。

## 制約

- exp002 all-GR residual model を Public LB anchor として扱う。
- exp004 の gated model bundle と notebook runner を再利用し、変更範囲は gate 条件と no-GR weight に絞る。
- train-only formation columns は使わない。
- validation は `well_id` GroupKFold、`TVT_input` NaN 行のみの RMSE とする。
- Kaggle Notebook 実行を正とし、ローカル notebook 実行は行わない。

## 受け入れ基準

- exp005 実験ディレクトリが作成され、config/docs/notebook 名が exp005 に揃っている。
- exp004 selected gate、soft gate、strict gate の比較 variants が `config.yaml` に定義されている。
- `gate_low_gr_strict_hard` が visible `000d7d20` を no-GR routing から外す意図を docs に記録している。
- `validate-exp` と Kaggle notebook prepare が通る。
