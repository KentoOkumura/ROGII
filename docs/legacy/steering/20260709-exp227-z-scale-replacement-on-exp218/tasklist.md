# タスクリスト

## TODO

- なし

## 進行中

- なし

## ブロック中

- なし

## 完了

- `exp218_gr_wavelet_rotation_confidence_features_on_exp148` から派生した exp224 実装を元に `exp227_z_scale_replacement_on_exp218` を作成した。
- `z_scale_replacement_on_exp218.py` に `drop_base_columns` 対応を追加した。
- `z_scale_replacement` variant に `drop_base_columns: [z, dz, dzdmd, slp_z]` を設定した。
- CPU split train notebook `train_lgb0` / `train_lgb1` / `train_lgb2` を exp227 名に揃えた。
- 再現性設計を `design.md` に記入した。
- Kaggle `train_lgb0` / `train_lgb1` / `train_lgb2` を version 1 として push した。
- Kaggle status / logs / output を取得し、3 split とも `COMPLETE` を確認した。
- `aggregate_split_oof.py` で split OOF aggregate、feature importance、bucket、worst-well readout を作成した。
- 結果を `SESSION_NOTES.md`、`result.md`、`README.md`、`metrics.json`、`experiment_summary.md` に記録した。
- CV が exp218 より悪いため inference port / submit は行わないと判断した。
