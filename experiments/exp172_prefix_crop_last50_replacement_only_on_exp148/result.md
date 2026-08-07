# exp172_prefix_crop_last50_replacement_only_on_exp148 結果

Kaggle CPU split train 完了。train-side OOF は exp148 historical `lgb_mean` より悪化したため、推論化・提出はしない。

## 仮説

exp148 の learned multiobs 系特徴は、known prefix 序盤の TVT 急降下を含むと hidden well でノイズ化しやすい可能性がある。last50 replacement-only では、learned multiobs 系を known prefix 末尾 50 行だけの特徴に差し替え、exp148 anchor を改善できるかを検証した。

## 比較基準

- 主 baseline: exp148 `lgb_mean` CV 8.50128118189582 / Public LB 7.960
- control 再学習: なし
- 旧基準: exp092 `lgb1` CV 9.322479895503927 / Public LB 8.350
- 参考: exp161 last50 add-only best single CV 8.56472499591314
- 参考: exp166 tail500 replacement-only best single CV 8.566426970340796

## 実行

- feature cache: `kentookumura/exp172-prefix-crop-last50-exp148-features` v1
- train:
  - `kentookumura/exp172-prefix-crop-last50-exp148-train-lgb0` v1
  - `kentookumura/exp172-prefix-crop-last50-exp148-train-lgb1` v1
  - `kentookumura/exp172-prefix-crop-last50-exp148-train-lgb2` v1
- active variant: `prefix_crop_last50_multiobs_replacement`
- rows / wells: 3,783,989 / 773
- features: 309
- prefix crop cache: 48 features generated, 30 last50 multiobs features loaded for training
- feature join coverage: pass, dropped rows 0, dropped wells 0

## CV

| model | pooled RMSE | delta vs exp148 | delta vs exp161 best | delta vs exp166 best |
|---|---:|---:|---:|---:|
| `lgb0` | 8.583559279 | +0.082278097 | +0.018834283 | +0.017132309 |
| `lgb1` | 8.575126850 | +0.073845668 | +0.010401854 | +0.008699879 |
| `lgb2` | 8.586986606 | +0.085705424 | +0.022261610 | +0.020559636 |

best single は `lgb1` の 8.57512684958155。exp148 `lgb_mean` 8.50128118189582 から +0.07384566768572931 悪化した。

## 解釈

last50 multiobs replacement-only は exp148 を改善しなかった。全系統を置換した exp166 よりは置換対象を狭めたが、learned multiobs を last50 版に差し替えるだけでも exp148 の既存 learned likelihood surface から有効な情報を失っている可能性が高い。

exp161 last50 add-only、exp166 tail500/tail1000 replacement-only、今回の last50 multiobs replacement-only がすべて exp148 より悪化したため、prefix crop-window 系は現状の「後付け add-only / replacement-only」では優先度を下げる。

## 判断

- exp172 は不採用。
- inference port / submit はしない。
- cross-kernel `lgb_mean` は output archive を取得して計算していない。全 single config が exp148 から明確に悪いため、追加 output download は行わない。
- prefix crop を続ける場合は、既存特徴の一部差し替えではなく、別 backlog の `last50_first_prefix_feature_rebuild_on_exp148` のように prefix source を先に last50 へ切ってから特徴量を作り直す方向だけを低優先で残す。
