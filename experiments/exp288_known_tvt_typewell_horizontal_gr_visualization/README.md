# exp288_known_tvt_typewell_horizontal_gr_visualization

## 状態

- ルート: pf_beam
- 状態: kaggle_train_v1_complete_diagnostic_only
- CV: -
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-07-19
- 親実験: `exp168_gr_matching_pair_visualization`

## 仮説

known区間は `TVT_input`、予測対象区間はtrain true `TVT`をType Wellの`TVT -> GR`曲線へ
補間した参照GRとhorizontal GRを全train wellで上下に並べれば、full-wellの一致・振幅差・欠損を
指標化前に目視確認できる。

## 変更点

- 全train wellを対象に、known + prediction-target区間のType Well参照GRを生成する。
- 横軸full-well MDを共有し、上段を参照GR、下段をhorizontal GRとしてwellごとにPNG保存する。
- `TVT_input.isna()`の予測対象区間を着色し、train true TVTを使うEDAであることを図中に明記する。
- 既存exp168のshift-scan候補図は変更しない。
- quality metric、calibration、offset search、モデル学習、推論、提出は行わない。

## 検証方針

- Fold: なし。可視化診断のみ。
- Group: 全train wellを1 wellずつ処理する。
- Stratification: なし。
- Leakage Check: prediction-target true `TVT`をtrain-only EDA図示に使うが、特徴量、学習、候補選択、推論、提出には使わない。

## 実行入口

- 学習 notebook: `exp288_known_tvt_typewell_horizontal_gr_visualization_train.ipynb`
- 推論 notebook: `exp288_known_tvt_typewell_horizontal_gr_visualization_inference.ipynb`
- Kaggle 準備: `task prepare-kaggle-notebooks EXP=exp288_known_tvt_typewell_horizontal_gr_visualization`
- notebook 実行: Kaggle kernel run を正とする。ローカル実行は `--allow-local` を付けた smoke debug のみに限定する。

## 結果

| メトリック | 値 |
| --- | --- |
| CV | - |
| Public LB | - |
| Private LB | - |
| Kaggle train | version 1 `COMPLETE` |
| 保存PNG / skip | 773 / 0 |
| 実処理時間 | 388.999秒 |

## 所見

### 良かった点

- 全773 wellsを処理し、1 well 1 PNGをskipなしで保存した。
- manifestに対して全PNGの存在、byte size、SHA256が一致した。
- 代表PNGで上下2段とprediction-target着色を目視確認した。

### 悪かった点

- 可視化のみで、residual/NCC/affine等の定量指標はまだ計算していない。

### リスク / 注意

- Type Well参照GRは同一TVT上のテンプレートであり、horizontal GRの無偏期待値とは限らない。
- 予測対象区間の参照GRはtrain true TVTを使うため、raw testへ移植できないEDA専用表示である。
- horizontal GR欠損は補完せず、そのまま線の欠損として表示する。

## 次

- HTML indexで全wellを目視し、定量化が必要なら別実験として品質指標を設計する。

## 表記

用語は `backlog/KAGGLE_DIRECTION.md` の表記方針と `docs/glossary.md` に合わせ、実験名や設定名を除いて日本語優先で記録する。
