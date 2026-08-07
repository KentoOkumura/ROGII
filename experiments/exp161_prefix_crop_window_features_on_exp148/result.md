# exp161_prefix_crop_window_features_on_exp148 結果

## 結論

prefix crop window feature の add-only は、単体 LGB config の best CV で exp148 を改善しなかった。

- best single: `lgb2`
- CV RMSE: 8.56472499591314
- exp148 `lgb_mean` CV: 8.50128118189582
- 差分: +0.06344381401732 悪化

現時点では提出候補にしない。split train のため、3 config 横断の `lgb_mean` ensemble は未評価。評価するには lgb0/lgb1/lgb2b の OOF prediction artifact を取得して結合する必要がある。

## 実装内容

exp148 の feature surface に `prefix_crop_window` group を add-only で追加した。最終的な学習では、Kaggle timeout と memory 対策として prefix crop feature 生成を別 notebook に分離し、学習 notebook は cache を必須入力として読み込む構成にした。

最終 feature cache:

- rows: 3,783,989
- wells: 773
- prefix crop features: 48
- feature cache: `exp161_prefix_crop_window_features_on_exp148_prefix_crop_train_features.csv.gz`
- cache sha256: `86b22a14b30425b079e532de0d3796f1e33bb9a25b1f61f6a5fcfc47d951a69b`

学習は timeout 対策として LGB config ごとに 3 notebook へ分割した。

## CV

| model | kernel | elapsed_seconds | CV RMSE |
|---|---:|---:|---:|
| lgb0 | `kentookumura/exp161-prefix-crop-exp148-train-lgb0` | 16,193.874 | 8.573959480512093 |
| lgb1 | `kentookumura/exp161-prefix-crop-exp148-train-lgb1` | 9,408.618 | 8.575152249652412 |
| lgb2 | `kentookumura/exp161-prefix-crop-exp148-train-lgb2b` | 14,034.888 | 8.56472499591314 |

各 split notebook 内の `lgb_mean` は、選択済み 1 config の平均なので単体 config と同値。

## 比較基準

- exp148 `lgb_mean`: CV 8.50128118189582 / Public LB 7.960
- exp092 `lgb1`: CV 9.322479895503927 / Public LB 8.350
- exp160: CV 8.463718774 / Public LB 8.061

## 次アクション

この実験単体は提出しない。必要なら split 3本の OOF prediction artifact を取得して横断 `lgb_mean` ensemble CV だけ確認する。
