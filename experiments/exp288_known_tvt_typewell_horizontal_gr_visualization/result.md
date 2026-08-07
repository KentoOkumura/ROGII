# exp288_known_tvt_typewell_horizontal_gr_visualization 結果

## 仮説

known `TVT_input`とprediction-target train true `TVT`からType Well参照GRを生成し、horizontal GRと
full-well MD軸で上下表示すると、物理モデルへ品質指標を追加する前に、両者の一致、振幅差、欠損、
Type Well範囲外を全wellで確認できる。

## 設定

- 親: `exp168_gr_matching_pair_visualization`
- 検証: 全train wellのknown + prediction-target visualization diagnostic
- メトリック: PNG保存数、skip数、known/target別reference/paired row coverage。CV/LBなし。
- シード: RNGなし

## 結果

| メトリック | 値 |
| --- | --- |
| CV | - |
| Public LB | - |
| Private LB | - |
| Notebook | Kaggle train v1 `COMPLETE` |
| 入力well | 773 |
| 保存PNG | 773 |
| Skip | 0 |
| Notebook実処理時間 | 388.999秒 |
| PNG合計サイズ | 157,474,395 bytes |

## 実装検証

- train/inference Jupytext変換・round-trip: PASS
- `py_compile`: PASS
- Ruff `F821,F401,E9`: PASS
- strict experiment validation: PASS
- raw train file pair: horizontal 773 / typewell 773 / missing 0
- train notebook: 15 cells、645行、kernelspec `python3`
- Kaggle package: private CPU、GPU/TPU/internet off
- target拡張後のJupytext、compile、Ruff、strict validation: PASS
- synthetic known/target補間・target MD span test: PASS
- loose/package config SHAとtrain source SHA: 一致
- kernel: `kentookumura/exp288-known-tvt-typewell-horizontal-gr-viz-train` version 1、id_no `127877148`
- manifest 773行 / saved status 773 / PNG 773: 一致
- PNG missing / byte size mismatch / SHA256 mismatch: 0 / 0 / 0
- 代表PNG `000d7d20.png`: 上下2段、full-well MD、prediction-target黄色着色を目視確認

## 再現性

- deterministic anchor: いいえ。可視化診断でありPNG byteは描画library versionに依存し得る。
- seed policy: RNGなし、well名辞書順、single process。
- kernel version: 1。
- manifest SHA256: `8cd742699beb4a649247d68b533fb58a2fd1c492e51bdadb9c2ca5cdd0b68eb1`。
- HTML index SHA256: `1ebdddeb7fcc1aacd51de21d5797cf9c0aab0cc0d5ea48c507ee97f532bba20c`。
- model SHA / manifest SHA: 対象外。
- prediction SHA: 対象外。
- submission SHA: 対象外。
- rerun result: 未実行。

## 解釈

Type Well参照GRはknown rowでは `interp(typewell.TVT, typewell.GR, TVT_input)`、prediction-target
rowでは `interp(typewell.TVT, typewell.GR, true TVT)` として作る。target区間は着色する。
true TVTはtrain-only EDA図示専用で、初回はresidual/NCC/affine等の指標も推定しない。

## 次

Kaggle実行と生成物検証は完了。HTML indexから全wellを目視し、必要なら次の別実験で
residual/NCC/affine等の定量指標を設計する。target true TVTを使う本実験の図を推論入力には流用しない。
