# exp172_prefix_crop_last50_replacement_only_on_exp148

## 概要

exp148 ML route submitted anchor に対する prefix crop-window replacement-only 実験。known prefix 末尾 50 行だけで作った crop 特徴に、learned multiobs 系を差し替える。

## 実行構成

- Route: `ml_model`
- 親: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- Runtime: CPU
- feature cache notebook: `exp172_prefix_crop_last50_replacement_only_on_exp148_prefix_crop_features.ipynb`
- train notebook: `lgb0` / `lgb1` / `lgb2` の3分割
- control 再学習: なし

## 比較基準

- exp148 `lgb_mean` CV 8.50128118189582 / Public LB 7.960
- exp092 `lgb1` CV 9.322479895503927 / Public LB 8.350

## 仮説

learned multiobs 系特徴に known prefix 序盤の TVT 急降下が混ざることで、hidden well では anchor 近傍の挙動とずれたノイズになる可能性がある。last50 に限定した replacement-only なら、exp148 が使っていた learned likelihood confidence を残しつつ、ノイズ化しやすい prefix 集計だけを近傍化できる。

## 検証方針

Kaggle CPU で last50 prefix crop cache を作成し、`lgb0` / `lgb1` / `lgb2` の split train notebook がその cache を必須入力として学習する。control は再学習せず、保存済み exp148 CV / Public LB を historical baseline とする。

評価は pooled OOF RMSE、fold 別 score、near-row、`1000_plus` bucket、worst-well regression、feature importance で行う。exp148 historical CV を改善しない場合、inference port / submit は行わない。

## 所見

Kaggle CPU split train 完了。best single は `lgb1` の CV 8.57512684958155 で、exp148 `lgb_mean` CV 8.50128118189582 から +0.07384566768572931 悪化した。exp161 last50 add-only best single 8.56472499591314 と exp166 tail500 replacement-only best single 8.566426970340796 にも届かない。

## 状態

完了/不採用。推論化・提出は行わない。
