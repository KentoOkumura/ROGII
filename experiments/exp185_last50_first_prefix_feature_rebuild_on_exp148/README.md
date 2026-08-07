# exp185_last50_first_prefix_feature_rebuild_on_exp148

## 概要

exp148 ML route submitted anchor に対し、known prefix を先に last50 に切ってから prefix 由来特徴を作り直す実験。exp161/166/172 の後付け add-only / replacement-only が negative だったため、prefix source 自体を last50 にした feature rebuild として分離する。

## 実行構成

- Route: `ml_model`
- 親: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- feature cache notebook: `exp185_last50_first_prefix_feature_rebuild_on_exp148_prefix_crop_features.ipynb`
- train notebook: `train_lgb0` / `train_lgb1` / `train_lgb2` の3分割
- Train runtime: Kaggle GPU, `gpu_repro_guard_dp_threads8`
- control 再学習: なし
- 予定 booster: 1 variant x 1 mode x 3 configs x 5 folds = 15

## 仮説

exp148 の既存 prefix 由来特徴は full known prefix の集計を含む。hidden/current test で anchor 近傍の prefix 末尾情報だけが効く場合、source frame を先に last50 へ切ってから TVT aggregate、trajectory/geometry、calibration、SC/NCC、multiobs score を再計算した方が、exp148 anchor を改善する可能性がある。

## 検証方針

学習・評価行は crop しない。`last_known_tvt`、anchor row、PF/Beam 候補値、U-projection、learned probability/error surface は exp148/exp145 の既存値を読む。active variant では full-prefix 由来 base columns と learned multiobs columns を落とし、last50-first rebuild group を追加する。

評価は pooled OOF RMSE、fold 別 score、near-row、`1000_plus` bucket、worst-well regression、feature importance、raw-test/current-test parity で行う。global OOF が小幅改善しても guard が弱ければ inference port / submit しない。

## 所見

未実行。feature cache と Kaggle GPU split train 完了後に、CV、bucket、worst-well、feature importance を記録する。

## 状態

実装中。Kaggle GPU train push 前。
