# 設計

## アプローチ

exp099 の train pseudo-tail cache を読み、raw train horizontal well から query geometry を復元する。GroupKFold の各 fold で validation well ごとに train-fold wells だけを neighbor source とし、標準化した well-level geometry feature 空間で近傍を選ぶ。source well の `true_delta_from_anchor` を `md_since` 軸で query well に補間し、distance-weighted mean を spatial prior とする。

評価は direct submit ではなく信号監査に限定する。`xy_only`、`xy_plus_azimuth`、`xy_plus_trajectory_shape`、`xy_plus_direction_and_typewell` を比較し、prior-only、`likpf_mean` / `pf_ancc` / `beam_mean` への clipped correction、base error と prior-base 差分の相関・符号一致率、distance bucket、by-well regression を見る。

## 実験範囲

- 対象実験: `exp114_spatial_neighbor_prior_signal_audit`
- Route: `ensemble`
- 親実験: `exp099_pf_multi_observation_likelihood_probe`
- 変更する変数: spatial neighbor 選択 feature set、same typewell group 制約、neighbor prior correction alpha / clip
- 固定する変数: exp099 fixed candidate surface、raw train pseudo-tail scoring rows、GroupKFold seed、PF/Beam candidate generation

## 再現性設計

- seed policy: `deterministic_groupkfold_fixed_neighbor_rules_no_model_rng`
- stochastic 処理の有無: この実験固有の stochastic 処理はなし。upstream exp099 / exp065 cache は SHA として記録する。
- PF/Beam / likelihood-PF / seed bagging の有無: rerun しない。保存済み exp099 candidate を読むだけ。
- 並列処理と乱数の関係: 並列処理なし、global RNG なし。GroupKFold well shuffle のみ fixed seed の local RNG。
- CPU/GPU runtime と deterministic flags: CPU notebook、GPU なし。
- train cache / test feature regeneration の SHA 記録方針: exp099 gzip は raw SHA と decompressed content SHA、schema SHA を記録する。OOF gzip も raw SHA と decompressed SHA を summary に記録する。
- model manifest / prediction / submission SHA 記録方針: モデルなし。OOF prediction SHA のみ記録し、submission は作らない。
- Kaggle package bootstrap 確認方針: `prepare-kaggle-notebooks --notebook train --run-on-push --strict` 後、生成 package の metadata と bootstrap support files を validate する。

## リスク

- リークリスク: valid well true TVT を neighbor source に入れると強く漏れる。実装は fold ごとに train wells のみを source にする。
- CV/LB 不一致リスク: train spatial neighbor structure が hidden test に対応する保証はない。global RMSE 改善だけでは submit しない。
- ランタイム/メモリリスク: 3.78M rows の cache と複数 candidate columns を保持するため memory は中程度。neighbor interpolation は well x top-k 単位なので PF rerun より軽い。
- 再現性リスク: upstream cache の再生成差分はこの実験では制御しない。入力 SHA を主証拠にする。
