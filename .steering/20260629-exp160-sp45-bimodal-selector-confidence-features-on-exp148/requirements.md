# 要件

## 依頼

`sp45_bimodal_selector_confidence_features_on_exp148` を実験化し、公開 SP45 / PF / Beam / bimodal selector 系の hidden-safe core を exp148 の add-only confidence feature として評価できる状態にする。

## 制約

- Route: `ml_model`
- 親実験: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- control 再学習はしない。exp148 の既存 CV / Public LB を historical baseline として参照する。
- selector 出力を direct replacement、late blend、postprocess hard gate として使わない。
- visible-prefix gold overlay、exact contact override、train/test overlap shortcut、public output CSV copy、oracle best、true-error rank、validation/test true TVT は使わない。
- 再現性: `docs/06_reproducibility.md` に従い、PF/Beam upstream cache、current-test regeneration、GPU LightGBM、SHA 記録の扱いを設計に明記する。

## 受け入れ基準

- exp160 の `config.yaml`、train notebook、inference notebook、補助 `.py` が実装されている。
- active variant は `sp45_bimodal_selector_confidence_addonly` のみ。
- Kaggle train push 前ガードとして、1 variant、3 LightGBM configs、5 folds、合計 15 boosters、control 再学習なしが `SESSION_NOTES.md` に記録されている。
- train 側で SP45/Bimodal feature summary、feature schema、feature importance、OOF predictions、model manifest を保存する設計になっている。
- inference 側で current-test replay から learned likelihood + SP45/Bimodal features を生成し、train manifest の feature group と一致しない場合に fail する。
- deterministic anchor としては扱わない。採用候補に進める場合は feature content SHA、model SHA、prediction SHA、submission SHA、Kaggle kernel version を記録する。
