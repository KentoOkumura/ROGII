# 要件

## 依頼

`ravaghi_ncc_gr_match_features` を実装する。Ravaghi notebook 由来の
multi-scale NCC (`sc8`, `sc15`, `sc25`, `sc_ens`) と GR/typewell match residual
を、pseudo-tail single LightGBM の feature family として検証できる形にする。

## 制約

- Route: `ml_model`
- 親実験は `exp041_ravaghi_beam_exact_feature_ablation`。
- 入力 artifact は `exp029_public_sel15_pf_oof_feature_generation` の train well の途中以降を隠した疑似 test rows。
- `target_tvt` は supervised label と scoring にだけ使う。
- raw GR 値そのもの、train-only formation columns、error diagnostic columns、`exp026_oof` bridge columns は model feature に入れない。
- NCC/GR match feature は pseudo cutoff 以降の `TVT_input` を隠した状態で、known prefix `TVT_input`、horizontal `GR`、typewell `TVT/GR` から再生成する。

## 受け入れ基準

- `exp042_ravaghi_ncc_gr_match_features` の experiment folder が作成されている。
- `config.yaml` に route、lineage、feature families、variants、leakage policy が明記されている。
- train notebook と audit script が `exp042` 名で実行可能な構成になっている。
- static checks、experiment validation、small smoke が通る。
- Full Kaggle CV の結果、artifact、解釈、next action が実験記録に反映されている。
