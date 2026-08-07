# 要件

## 依頼

`KAGGLE_DIRECTION.md` の backlog `tvt_input_pfbeam_offset_calibration` を実験化する。

## 制約

- Route: `pf_beam`
- 既存 exp072 PF/Beam tail candidate cache を評価対象にする。
- known prefix の `TVT_input` だけを offset 推定に使い、評価区間の true TVT / oracle / true-error rank は補正条件に使わない。
- GPU 学習は行わない。LightGBM config 数 0、fold 数 0、booster 数 0。
- 再現性: `docs/06_reproducibility.md` に従い、upstream cache SHA、prefix replay 出力 SHA、gzip decompressed SHA を記録する。

## 受け入れ基準

- 各 well の known prefix 末尾を holdout として PF/Beam replay し、candidate 別 `candidate_tvt - TVT_input` の robust offset / IQR / slope / prefix error を出力する。
- exp072 tail candidate に constant / capped / fade-in offset correction を適用し、補正前後 RMSE、distance bucket、near row、longtail、worst-well regression を比較する。
- 実験結果は train-side diagnostic として記録し、positive でも raw-test parity と boundary discontinuity review なしに inference port / submit しない。
