# exp166_prefix_crop_midwindow_replacement_on_exp148

## 状態

- Route: `ml_model`
- Status: `completed_train_side_rejected_no_submit`
- 親実験: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- 比較基準: exp148 `lgb_mean` CV 8.50128118189582 / Public LB 7.960
- 実行: CPU、2段階構成。prefix crop cache v1 作成後、split LightGBM notebook `lgb0` / `lgb1` / `lgb2` v2 を完了。
- 提出: なし

## 仮説

exp148 の full-prefix 系特徴は、known prefix 序盤の TVT 急降下を含むと hidden well でノイズ化しやすい。案2では last50 より広い `tail500` / `tail1000` を先に検証し、急降下区間を薄めながら anchor 近傍の prefix 情報を残す。

add-only ではなく replacement-only とし、full-prefix のうち急降下の影響を受けやすい slope、calibration、SC/NCC、learned multiobs score/MAE/NCC を落として、対応する crop-window features に差し替えた。

## 結果

最良は `prefix_crop_tail500_replacement` / `lgb0` の CV 8.566426970340796。exp148 `lgb_mean` CV 8.50128118189582 から +0.065145788444976 悪化した。

`tail1000` の最良は `lgb2` CV 8.574216682757848。exp148 から +0.072935500862028 悪化した。

## 検証方針

control は再学習せず、保存済み exp148 の CV / Public LB を historical baseline として比較した。有効 variant は `prefix_crop_tail500_replacement` と `prefix_crop_tail1000_replacement`。学習 notebook は prefix crop cache を必須入力とし、LightGBM 学習中には crop feature を再生成しない。

判定は exp148 `lgb_mean` CV 8.50128118189582 を主基準とした。best single が exp148 を明確に悪化したため、Kaggle output download、cross-lgb ensemble、inference port、submit は行わない。

## 所見

tail500/tail1000 replacement-only は、急降下ノイズの除去よりも既存 full-prefix 情報を落とす損失が大きかったと見る。exp161 last50 add-only best single 8.56472499591314 にも届かず、prefix crop-window 系は現状のまとめ足し/まとめ置換では exp148 anchor を改善しない。

案1の last50 replacement-only は isolated test としては残るが、優先度は下げる。実施するなら今回より置換対象を狭めた ablation にする。

## 注意

- 学習・評価行は crop しない。
- crop 境界は raw hidden test でも再現できる `MD >= anchor_md - 500/1000` を使った。
- PF/Beam 生成、U-projection、learned probability/error model は crop-window 版へ再生成していない。
- v1 は prefix crop 96列の一括 concat によるメモリピークで失敗。v2 は variant ごとに必要な48列だけを読み込む構造で完走した。
