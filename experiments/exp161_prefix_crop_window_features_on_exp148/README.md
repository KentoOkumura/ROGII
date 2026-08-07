# exp161_prefix_crop_window_features_on_exp148

## 状態

- Route: `ml_model`
- Status: `kaggle_train_v1_running`
- 親実験: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- 比較基準: exp148 `lgb_mean` CV 8.50128118189582 / Public LB 7.960
- 実行: CPU LightGBM train v1 running、1 variant、3 configs、5 folds、15 boosters
- 提出: なし

## 仮説

exp148 の known prefix 全体統計は、序盤 build section の影響で hidden well に対して不安定になる可能性がある。既存 exp148 features は置換せず、anchor 近傍の `tail1000` / `tail2000` / `last50` crop-window 版を add-only で渡すと、prefix trust、SC/NCC、multi-observation likelihood の判断材料が増える。

## 検証方針

`prefix_crop_window_addonly` だけを学習する。control は再学習せず、保存済み exp148 の CV / Public LB を historical baseline として参照する。PF/Beam 生成、U-projection、learned probability/error model は crop-window 版へ置き換えない。

## 所見

Kaggle CPU train v1 を `kentookumura/exp161-prefix-crop-exp148-train` に push 済み。metadata は `enable_gpu=false`、`machine_shape=None`、`enable_internet=false`。

## 注意

- 学習・評価行は crop しない。
- crop 境界は raw hidden test でも再現できる `MD >= anchor_md - 1000/2000` と known prefix 末尾 50 行だけを使う。
- global OOF が改善しても、near-row、`1000_plus`、worst-well、raw-test/current-test parity が弱ければ submit しない。
