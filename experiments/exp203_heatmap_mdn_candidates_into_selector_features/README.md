# exp203_heatmap_mdn_candidates_into_selector_features

## 状態

Kaggle train v1 完了。train-side no submit。

## 仮説

exp202 の heatmap MDN topK は単独候補としては既存 PF/Beam union より弱いが、既存候補 union に足す oracle headroom は大きい。topK を直接選ばず、既存 selector の confidence / candidate-distance feature として渡せば、exp184 selector が候補選択を改善できる可能性がある。

## 目的

exp202 の heatmap MDN topK candidate signal を、既存の exp184 selector に add-only feature として渡す。selector が選べる候補は exp184 と同じ 8 候補に固定し、heatmap MDN topK は直接予測や候補追加には使わない。

## 方針

- 親: `exp184_cnn_sdf_mtp_heatmap_path_features_on_exp158`
- route: `pf_beam`
- 追加 feature prefix: `hmdn_`
- 既存 `hmpf_` feature は維持
- direct replacement / weighted average / PF weight replacement / postprocess / submit は対象外

## 評価

LightGBM multiclass / binary / candidate-error ranker と exp184 同等の Viterbi grid を評価する。主に global RMSE、path switch、worst-well、distance bucket、exp115 hidden-like subgroup、`hmdn_` confidence / sparse-distance bucket を確認する。

## 検証方針

Kaggle Notebook train を正とする。実行前に 1 active selector variant、3 LightGBM configs、5 folds、15 boosters、control retraining なしを確認する。結果取得後、exp184 best Viterbi RMSE 10.560650325 と比較し、global だけでなく hmdn sparse-distance bucket と exp115 subgroup を見る。

## 所見

Kaggle train v1 は `kentookumura/exp203-hmdn-selector-train` で COMPLETE。3,783,989 rows / 773 wells、298 features、15 boosters を学習し、`hmdn_` generated feature は 75 個だった。

best Viterbi は RMSE 10.665741318 / MAE 6.350286735 / within10 0.797977743、path switches 12,807 / 3.384524 per 1000 rows。exp158 continuity 10.789163253 からは -0.123421935 改善したが、親の exp184 best Viterbi 10.560650325 からは +0.105090994 悪化した。

したがって exp203 は feature-only signal の確認としては有用だが、exp184 を更新しないため inference / submit には進めない。意図していた heatmap MDN path の選択候補化は `exp204_heatmap_mdn_topk_as_selector_candidates_on_exp158` で扱う。
