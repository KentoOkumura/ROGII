# exp261_lightgbm_extra_trees_ablation_on_exp218 結果

## 状態

Kaggle train v1完了。全guard不通過のため回帰variantは不採用。

## 仮説

exp218の回帰LightGBMに `extra_trees=True` だけを加えることで、random threshold由来の
汎化または親OOFとの補完性が得られるかを検証する。

## 設定

- 親: `exp218_gr_wavelet_rotation_confidence_features_on_exp148`。
- 変更: 選択LightGBM configの `extra_trees=True` のみ。
- 固定: 380 features、well GroupKFold 5 folds、config別seed、その他parameter、GPU deterministic mode、early stopping。
- control: 保存済みexp218 boosters/OOFを推論利用し、再学習しない。
- 実行: `full_family`、1 variant × 3 configs × 5 folds = 15 boosters。単一train notebook内で実行する。
- メトリック: RMSE。

## 結果

| メトリック | 値 |
| --- | --- |
| 親3-config mean RMSE | 8.475793752 |
| `extra_trees=True` 3-config mean RMSE | 8.755217124 |
| delta | +0.279423372 |
| lgb0 / lgb1 / lgb2 delta | +0.307747552 / +0.239395929 / +0.233455688 |
| 改善fold | 1/5 |
| 1000+ delta | +0.315774718 |
| hidden-like spatial / typewell-purged delta | +0.243318548 / +0.250390215 |
| worst-well regression | +11.324423 |
| best fixed blend（extra weight 0.25）delta | +0.031917897 |
| adoption supported | false |
| Public LB | 未実行 |
| Private LB | - |

## 評価

- 対応config別matched control RMSEとdelta。
- 選択config平均のoverall / fold / distance / 1000+ / hidden-like / by-well / worst-well。
- frozen exp218 `lgb_mean`とのOOF相関と固定blend `0.25/0.50/0.75`。
- full family時の保存済みbooster再推論とfrozen exp218 OOF parity。
- feature importance、feature content SHA、model manifest/model SHA、OOF decompressed SHA。
- 親3-config予測の再構築はmean/max absolute difference 0で、保存OOF RMSEと完全一致した。
- parameter auditは全3 configで変更keyが`extra_trees`だけだった。

## 再現性

- deterministic anchor: GPU rerun未実施のため扱わない。
- seed policy: exp218のGroupKFold/config seedを固定。
- kernel: `kentookumura/exp261-lgb-extra-trees-exp218-train` version 1、runtime `19931.364`秒。
- feature content SHA: `f6ff78f6a95e47b0ed8e76a22c31d3403d0a9e78471b7d64f37eef7a2a398e29`。
- OOF decompressed SHA: `021fd47ac556b7ce98f7991b0c97aa6996b359f07c07a7e6a339c84d84100f00`。
- model manifest SHA: `d76ca02ad54184310eafb2758bd287a48cf20cbd95911db5eaef103c8e8de476`。
- summary SHA: `34cb4c1d62076f0f62f402a52975b159ae1c966ddfff12fcbdf24384f6705ad4`。
- submission SHA: inference/submit未実装。

## 解釈

仮説は不支持。`extra_trees=True`は全3 configを悪化させ、特にfold 3/4とlong-tailで回帰が大きい。
親との相関が非常に高いため多様性の追加もほぼなく、0.25 blendでもoverallとhidden-likeは救済できない。
一部near bucketの小改善だけを根拠に推論化するとposthoc過適合のリスクが高い。

## 次

回帰LightGBMへの`extra_trees=True`は閉じ、inference / submit /追加parameter gridへ進めない。
再訪は保存OOFだけを使うnear-range安定性readoutに限定し、selector LightGBMはexp262で独立判定する。
