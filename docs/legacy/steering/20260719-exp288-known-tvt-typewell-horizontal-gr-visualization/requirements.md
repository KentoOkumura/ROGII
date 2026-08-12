# 要件

## 依頼

全 train well について、known区間は `TVT_input`、予測対象区間はtrain true `TVT` を
Type Well の `TVT -> GR` 曲線へ線形補間して参照 GR を作り、同じ horizontal row の
実測 `GR` と上下 2 段に並べた full-well PNG を保存する Kaggle-first EDA notebook を作成する。

## 制約

- Route: `pf_beam`。Type Well / GR 観測モデルの可視化診断であり、ML 学習や ensemble は行わない。
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- 正の編集対象は Jupytext percent 形式の self-contained train `.py` と、そこから生成する正規 `.ipynb` とする。
- 既存 `exp168` notebook は上書きしない。
- 横軸は horizontal well 全体の `MD`、上段は Type Well 参照 GR、下段は horizontal GR とし、両段で同じ GR 表示範囲を使う。
- `TVT_input.isna()` を予測対象区間とし、両パネルで着色する。
- known区間は `TVT_input`、予測対象区間はtrain true `TVT`を補間座標にする。true TVTはEDA図示専用で、特徴量、学習、推論、提出へ使わない。
- Type Well の finite `TVT,GR` 範囲外は外挿せず NaN とする。horizontal GR の欠損は補完せず、線の欠損として表示する。
- 生データや生成 PNG を Git 管理対象へ常設しない。PNG は notebook 実行時の `artifacts/` 出力とする。
- 初回実装では residual scale、NCC、affine、自己相関、entropy 等の品質指標は推定しない。

## 受け入れ基準

- `data/raw/train` または Kaggle competition input を解決し、horizontal/typewell pair を全件列挙できる。
- 各 well について `{well}.png` を 1 枚保存し、上段/下段、full-well共有 MD 軸、共通 GR 範囲、known/予測対象 row 数を確認できる。
- 予測対象区間でも、true TVTがfiniteかつType Well範囲内なら参照GRを描画し、horizontal GRもfull-wellで描画する。
- Type Well の重複 TVT は同一 TVT の GR median へ集約し、finite 2 点未満の well は fail-closed ではなく理由付き skip として manifest に残す。
- PNG manifest CSV、HTML index、summary JSON を保存し、known/予測対象それぞれのreference/paired row数、入力 well 数、保存数、skip 数、出力パスを notebook 上に表示する。
- train/inference Jupytext 変換、`--test`、`py_compile`、`ruff --select F821`、strict experiment validation が通る。
- deterministic anchor とは扱わない。RNG、model、prediction、submission は存在しないことを記録する。
