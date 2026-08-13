# exp404_scale5_sigma_gr_likelihood_pf_ablation

## 状態

- ルート: `pf_beam`
- 状態: train-side scientific gate FAIL、global scale5 x1.3を閉鎖
- CV: control x1.0 `10.914522` / candidate x1.3 `11.174615`
- Public LB / Private LB: なし
- 作成日: 2026-07-26
- scientific parent: `exp400_all_well_1p3_sigma_gr_likelihood_pf`
- PF kernel parent: `exp072_exp063_full_replay_feature_cache`
- Kaggle kernel: version 4 COMPLETE / id_no `128628818`
- 実行URL:
  <https://www.kaggle.com/code/kentookumura/exp404-scale5-sigma-gr-likpf-ablation-train>

## 仮説

exp400では全well `gs×1.3`が算術seed平均`pf_mean`を悪化させたが、
likelihood temperature 5で低尤度seedを抑えれば、x1.0よりx1.3の方が
GRへの過信を弱めつつwrong alignment modeを除ける。

## 変更点

`scale_5`を固定し、次の2 variantだけを同じwell別seedで再生成する。

```text
control   = gs × 1.0 + scale_5
candidate = gs × 1.3 + scale_5
```

- particles 500、seeds 128、temperature 5
- x1.0/x1.3でcommon per-well SHA256 seed
- `gs`以外のexp072 likelihood-PF contractは固定
- `pf_mean`はexp072/exp400とのtechnical parity専用
- scale 3/8/12、HMM、Beam、ML、blendは作らない

## 検証方針

- Fold / Group: exp226 outer 5 folds / `well_id`
- Score rows: 全773 wells、3,783,989 unknown-suffix rows
- Primary: `RMSE(scale5_x1p0) - RMSE(scale5_x1p3)`
- Promotion: pooled `>=0.05 ft`改善、4/5 folds、GR observed/missing、
  high missing、1000+、hidden-like 2面、by-well p95、worstの全AND
- Leakage: 両variant predictionとSHAをfreeze後だけsuffix truthをjoin

## 実行量

- scientific variants: 2
- PF well-runs: 1,546
- seed-well trajectories: 197,888
- particle starts: 98,944,000
- model / booster / HMM / Beam: 0
- exp400実測からのCPU runtime見積り: 約5.83時間

## 実行入口

- 学習 notebook: `exp404_scale5_sigma_gr_likelihood_pf_ablation_train.ipynb`
- 推論 notebook: `exp404_scale5_sigma_gr_likelihood_pf_ablation_inference.ipynb`
- Kaggle 準備: `task prepare-kaggle-notebooks EXP=exp404_scale5_sigma_gr_likelihood_pf_ablation`
- notebook 実行: Kaggle kernel run を正とする。ローカル実行は `--allow-local` を付けた smoke debug のみに限定する。

2026-07-26の実行指示により、compact self-contained候補を正規Notebookへ
採用した。実装元は次の別名Jupytext source / Notebookとして保持する。

- `exp404_scale5_sigma_gr_likelihood_pf_ablation_compact_selfcontained_train.py`
- `exp404_scale5_sigma_gr_likelihood_pf_ablation_compact_selfcontained_train.ipynb`
- `exp404_scale5_sigma_gr_likelihood_pf_ablation_compact_selfcontained_inference.py`
- `exp404_scale5_sigma_gr_likelihood_pf_ablation_compact_selfcontained_inference.ipynb`

trainは11章・約2,000行でexp400 compactと同じ役割構成を持つ。inferenceは
submissionを生成しないfail-closed実装である。private CPU、GPUなし、
internet offのversion 4でlate readoutまで完了した。inferenceとsubmissionは
未承認のまま終了した。

## 所見

- technical gateとexp072/exp400 parityはすべてPASSした。
- pooled RMSEはx1.0 `10.914522`に対してx1.3 `11.174615`で、
  x1.3が`0.260093 ft`悪化した。
- nonworse foldは`1/5`。raw GR observedは`0.179632 ft`、missingは
  `0.431145 ft`、high-missingは`0.693932 ft`悪化した。
- hidden-like 2面は`0.068326 / 0.013773 ft`改善したが、事前固定した
  全AND gateを満たさない。
- by-well delta p95 `+4.826467 ft`、worst regression `+37.333851 ft`で、
  global scale5 x1.3はtailも悪化した。

## リスク / 注意

- scale 5はexp400結果後に注目したため、本実験はscale 5対meanのpromotion試験ではない。
- 公開Notebookはseedと後段pipelineが異なり、discussionのscoreを直接再現しない。
- `gs`差でresampling後のtrajectoryが分岐するが、これはtreatmentの一部である。
- version 1で生成・freezeしたpredictionをversion 4でSHA固定して再利用した。
- technical recovery中もscientific contract、prediction values、gateは変更していない。

## 実装検証

- parent exp072 Numba kernel fixture parity
- common per-well seed、x1.0/x1.3 scale、scale-5限定、mean parity
- truth-late freeze、paired metric、execution count、promotion gate
- 専用test: `12 passed`
- 専用 + notebook tests: `16 passed`
- Jupytext round-trip / py_compile / Ruff F821 / strict validate-exp: PASS

## 実行結果

- version 4 status: `KernelWorkerStatus.COMPLETE`
- runtime: `270.988 sec`（v1凍結予測のlate readout）
- prediction logical SHA:
  `5f4b6e715081b598b0a34607ad0c81339d0ecd5882ea3a45dd79f33123959a00`
- artifact manifest SHA:
  `131a65c36acafc8d3cac9bdc18b2b5e296ff9aceb93cbf2702b1a79e675b58f3`
- technical gate: PASS
- scientific gate: FAIL
- decision: `scale5_rejects_global_gs_x1p3_close_without_rescue`

同じOOFでtemperature / multiplier / clip / seed / particle / adaptive gate /
HMM / Beam / ML / blendを救済探索しない。inference / submissionは行わない。

## 表記

用語は `backlog/KAGGLE_DIRECTION.md` の表記方針と `docs/glossary.md` に合わせ、実験名や設定名を除いて日本語優先で記録する。
