# 要件

## 目的

`tvt_plus_z_beam_smoothness_penalty` を独立実験として実装する。Beam search の cost に、`TVT` 単体の平滑性だけでなく `U = TVT + Z - (T0 + Z0)`、`dU/dMD = dTVT/dMD + dZ/dMD`、必要に応じて `dU/dMD` の変化を penalty として入れ、train pseudo-tail 上で既存 PF/Beam/likPF 候補と比較する。

## 受け入れ条件

- 対象実験は `experiments/exp146_tvt_plus_z_beam_smoothness_penalty/`。
- `trajectory_aware_pf_transition_prior` とは分離し、exp142 に Beam smoothness variant を混ぜない。
- raw train horizontal/typewell から Beam search を再実行し、既存 exp072 `beam_mean` への posthoc 補正では済ませない。
- 生成候補は evaluation-zone true TVT を使わない。
- 比較対象として `likpf_mean`、exp072 `beam_mean`、`pf_ancc`、`pf_z` を同じ rows で評価する。
- candidate metrics、bucket metrics、by-well metrics、group metrics、Beam quality、candidate wide、summary JSON を保存する。

## 非目標

- 推論 port、提出、exp092/exp098 ML への feature 追加はこの実装では行わない。
- exp142 v2 の結果は混在実装として採用しない。
