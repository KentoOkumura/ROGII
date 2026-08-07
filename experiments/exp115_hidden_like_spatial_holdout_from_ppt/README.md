# exp115_hidden_like_spatial_holdout_from_ppt

## 状態

- ルート: ml_model
- 状態: kaggle_train_v1_verified
- CV: -
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-06-23
- 親実験: exp092_u_projection_correction_disagreement_fullrun

## 仮説

公式 PPT slide10 の赤い Verification well 分布を exact hidden split としてではなく、空間分布の proxy として使うと、通常 GroupKFold とは別の stress holdout を作れる。これにより `exp092` / `exp073` / `exp098` などの ML route anchor を、hidden-like な空間・eval length・GR coverage 条件で比較できる。

## 変更点

- `hidden_like_spatial_holdout_from_ppt.py` を追加し、PPTX を zip として読み、slide10 の埋め込み PNG から赤 component を標準ライブラリだけで抽出する。
- train CSV から well-level metadata を再計算する。
- `verification_like_spatial` と `verification_like_typewell_purged` の 2 種類の holdout を保存する。
- inference notebook は no-submission と明示した。

## 検証方針

- Fold: 固定 holdout。通常の 5-fold CV ではない。
- Group: `well_id` 単位。
- Stratification: PPT red component への centroid 最近傍距離、spatial bin、azimuth bin、eval length、prefix length、GR coverage、TVT bin。
- Leakage Check: selection に true TVT tail は使わない。typewell-purged variant では valid の exact typewell group mate を `purged_train_excluded` として明示する。

## 実行入口

- 学習 notebook: `exp115_hidden_like_spatial_holdout_from_ppt_train.ipynb`
- 推論 notebook: `exp115_hidden_like_spatial_holdout_from_ppt_inference.ipynb`
- Kaggle 準備: `make prepare-kaggle-notebooks EXP=exp115_hidden_like_spatial_holdout_from_ppt EXTRA_ARGS="--notebook train --run-on-push --strict --title 'exp115 hidden like spatial holdout from ppt train'"`
- notebook 実行: Kaggle kernel run を正とする。ローカル実行は `--allow-local` を付けた smoke debug のみに限定する。
- Kaggle kernel: `kentookumura/exp115-hidden-like-spatial-holdout-from-ppt-train` v1
- Kaggle output: `kaggle/output/train_v1/`

## 結果

| メトリック | 値 |
| --- | --- |
| PPT red component | 45 |
| Train wells | 773 |
| `verification_like_spatial` valid wells | 200 |
| `verification_like_typewell_purged` valid wells | 200 |
| Purged train excluded wells | 16 |
| Spatial median PPT red distance | 0.018609910 |
| Spatial max PPT red distance | 0.080668542 |
| CV | - |
| Public LB | - |
| Private LB | - |

## 所見

### 良かった点

- ローカル smoke と Kaggle train v1 の両方で PPT slide10 から赤 component 45 件を抽出できた。
- local / Kaggle の追加画像ライブラリに依存しない PNG decoder を実装した。
- Kaggle train v1 は `COMPLETE` で完了し、output を保存できた。

### 悪かった点

- この実験自体はスコアを出さないため、次の readout 実験が必要。
- PPT/PNG は説明資料なので、この holdout は exact hidden split ではない。

### リスク / 注意

- PPT 画像の赤 component 閾値に依存する。閾値を変えた場合は別 version として扱う。
- この実験自体はスコア改善や提出候補を作らない。

## 次

- 次の follow-up で、保存済み Kaggle output を正として `exp092` / `exp073` / `exp098` の OOF 予測をこの holdout 上で再採点する。

## 表記

用語は `KAGGLE_DIRECTION.md` の表記方針と `docs/glossary.md` に合わせ、実験名や設定名を除いて日本語優先で記録する。
