# exp141_z_driven_pf_z_candidate_gate

## 状態

- ルート: `pf_beam`
- 状態: `completed_train_side_rejected_no_submit`
- CV: 11.594897672
- Public LB: 未提出
- Private LB: 未提出
- 作成日: 2026-06-27
- 親実験: `exp072_exp063_full_replay_feature_cache`

## 仮説

`pf_z` は単体では `likpf_mean` に負けるが、Z trajectory と同期する低頻度区間では候補として価値がある可能性がある。`likpf_mean` を default に固定し、target-free な Z-driven gate だけで `pf_z` を選ぶ。

## 変更点

- exp072 feature cache を固定入力にした posthoc audit を追加。
- `abs(dzdmd)`、Z slope alignment、候補差分、PF/Beam disagreement、path roughness、near-prefix guard を使う gate grid を実装。
- row / segment / well scope と switch-rate cap で 1-10% 程度の低頻度選択に制限。
- 新規学習、PF 再生成、submission 生成は行わない。

## 検証方針

- Fold: なし。保存済み train pseudo-tail cache の posthoc 評価。
- Group: `well`
- Metric: RMSE、MAE、within10、switch rate、by-well regression、bucket RMSE。
- Leakage Check: gate 条件は target-free columns のみ。`target_tvt` は scoring と oracle readout だけに使う。

## 実行入口

- 学習 notebook: `exp141_z_driven_pf_z_candidate_gate_train.ipynb`
- 推論 notebook: `exp141_z_driven_pf_z_candidate_gate_inference.ipynb`
- Kaggle 準備: `task prepare-kaggle-notebooks EXP=exp141_z_driven_pf_z_candidate_gate`
- notebook 実行: Kaggle kernel run を正とする。ローカル実行はユーザー明示の smoke debug のみに限定する。

## 結果

Kaggle train v1 は COMPLETE。best は baseline `likpf_mean` の RMSE 11.594897672 で、設定した `pf_z` low-frequency gate はすべて悪化した。

| variant | RMSE | delta vs `likpf_mean` |
| --- | ---: | ---: |
| `base_likpf_mean` | 11.594897672 | 0.000000000 |
| `seg_zq75_alignq60_diffq70_sr010_min32_clip20_a050` | 11.633719432 | +0.038821760 |
| `seg_zq80_alignq65_diffq65_sr005_min24_clip20_a075` | 11.680175213 | +0.085277540 |
| `row_zq90_alignq75_diffq75_sr001_clip25` | 11.714344701 | +0.119447029 |
| `single_pf_z` | 17.788171172 | +6.193273499 |

oracle `likpf_mean + pf_z` は RMSE 9.115200716、core PF/Beam oracle は RMSE 6.953036836 まで改善するため headroom はある。ただし今回の target-free gate では選択できず、最大 well regression と path step 増加が出るため inference port / submit はしない。

## 所見

### 良かった点

- direct replacement ではなく、switch-rate cap と segment guard 付きの低頻度 gate として検証できる。
- gate 条件は target-free columns に限定している。
- `ba48188d` / `fef8af96` のように `pf_z` が強い代表 well は確認でき、oracle headroom は残った。

### 悪かった点

- train-side posthoc audit なので、この段階では submit 候補ではない。
- ローカルの exp072 full cache は空ファイルしかなく、full audit は Kaggle input で実行する必要がある。
- global な target-free gate は `likpf_mean` を超えず、最良 gate でも RMSE +0.038822 悪化した。
- row-wise gate は不連続 step を増やし、well-level gate は最大 well regression が大きい。

### リスク / 注意

- `pf_z` は過去実験で `likpf_mean` の単体代替に負けているため、global RMSE だけで採用しない。
- 改善しても raw-test-compatible inference port と hidden-like stress readout が必要。
- 今回は改善がないため、raw-test inference port と submission は行わない。

## 次

1. `z_driven_pf_z_candidate_gate` backlog は完了 / 不採用として閉じる。
2. `pf_z` は hard switch ではなく、segment-level verifier / confidence feature / 小補正の材料として扱う。
