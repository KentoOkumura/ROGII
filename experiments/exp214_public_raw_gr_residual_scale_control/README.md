# exp214_public_raw_gr_residual_scale_control

## 状態

- Route: `pf_beam`
- 状態: `completed_train_side_audit`
- CV: train-side diagnostic only、Kaggle train v1 完了
- Public LB: なし
- Private LB: なし

## 仮説

公開 PF lineage の基準は、GR を denoise / affine 補正するのではなく、raw horizontal GR と raw typewell GR を使い、known prefix の residual scale で likelihood の強さを調整する。この public-like raw control を固定しておけば、GRCAL-PFBEAM 系の補正実験が本当に raw public baseline を上回ったかを読める。

## 検証方針

- exp211/213 と同じ exp072-compatible pseudo-tail 評価面を使う。
- 対象は最大 64 wells。
- PF は 500 particles x 128 seeds、scale 3/5/8/12 を保存する。
- `gs = clip(std(GR - typewell_GR(TVT_input)), 10, 60)` を known prefix だけから計算する。
- 評価 tail の true TVT は scoring のみに使い、PF 生成には使わない。

## 所見

Kaggle train v1 で 478,958 rows / 64 wells の public-like raw control を生成した。primary `pf_raw_scale_5` は RMSE 15.596465、best non-oracle `pf_raw_scale_12` は RMSE 15.223857。oracle diagnostic は RMSE 11.104328 まで下がるため、scale / seed / path confidence は後段 selector feature の材料として残す。

direct inference / submit は行わない。この実験は GRCAL-PFBEAM 系の固定 control として扱う。

## 参照ファイル

- `config.yaml`
- `public_raw_gr_residual_scale_control.py`
- `exp214_public_raw_gr_residual_scale_control_train.py`
- `SESSION_NOTES.md`
