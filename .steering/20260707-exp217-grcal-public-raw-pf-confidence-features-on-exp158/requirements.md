# 要件

## 依頼

`grcal_public_raw_pf_confidence_features_on_exp158` backlog を実装する。

## 制約

- Route: `pf_beam`
- 親実験: `exp158_segment_continuity_selector_on_exp157`
- 候補集合は exp158 と同じ 8 候補で固定する。
- `pubraw_pf_scale5/12` は selectable candidate、direct replacement、blend、postprocess、PF weight replacement、inference、submit に使わない。
- exp214 scoped output をそのまま join せず、full train/current-test compatible に target-free 再生成する。
- valid/test true TVT、oracle best、true-error rank、OOF absolute error を feature source に漏らさない。
- 再現性: `docs/06_reproducibility.md` に従い、PF seed policy、feature SHA、model manifest、prediction SHA を記録する。

## 受け入れ基準

- exp217 実験ディレクトリ、config、train/inference notebook、helper、README/result/metrics が exp217 名で揃っている。
- Kaggle train 前に active variant 1、LightGBM 3 configs x 5 folds = 15 boosters、control/parent retraining なしが記録されている。
- static validation、Jupytext conversion、`make validate-exp` が通る。
- 実行後は exp158 RMSE 10.789163253、exp184 RMSE 10.560650325、exp191 RMSE 10.598006880 と比較できる生成物を保存する。
