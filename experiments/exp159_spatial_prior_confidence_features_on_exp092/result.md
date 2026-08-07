# exp159_spatial_prior_confidence_features_on_exp092 結果

## 仮説

spatial neighbor prior は直接補正や selector では危険があるが、exp092 が外れやすい high-disagreement / longtail regime を示す confidence feature としては有効な可能性がある。

## 状態

実装済み、未実行。Colab で full train する前提。

## 実装

- 親: `exp092_u_projection_correction_disagreement_fullrun`
- spatial prior 入力: `exp114_spatial_neighbor_prior_signal_audit` OOF prior cache
- 追加特徴:
  - `xy_only_k8` / `xy_plus_trajectory_shape_k8` prior delta
  - prior minus `likpf_mean` / `beam_mean` / `pf_ancc`
  - prior std / count / neighbor wells / distance / same-typewell share / mismatch
  - exp118 best gate proxy
  - near / longtail / high-disagreement interaction
- direct correction、hard selector、oracle label、true error rank は使わない。

## 結果

未実行。

## 次

Colab で実行し、`latest_done_summary.json` と軽量生成物を取得してから結果を記録する。
