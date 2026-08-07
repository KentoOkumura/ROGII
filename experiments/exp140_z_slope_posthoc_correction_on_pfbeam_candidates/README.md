# exp140_z_slope_posthoc_correction_on_pfbeam_candidates

## 状態

- ルート: pf_beam
- 状態: completed_train_side_rejected_no_submit
- CV: 11.594897672217703 (`likpf_mean` baseline)
- Public LB: なし
- Private LB: なし
- Submit ID: なし
- 作成日: 2026-06-27
- 親実験: `exp072_exp063_full_replay_feature_cache`

## 仮説

exp083 の diagnostic plot では、一部 well で true TVT の大きな変化が `-Z` や `dZ/dMD` と同期して見えた。一方で exp100/104/106 の `pf_z` 系は `likpf_mean` の直接置換には届かなかった。

今回は PF/Beam を再生成せず、固定済み PF/Beam 候補に `-dZ/dMD` と candidate slope の gap から作る小さい累積補正を gate 付きでかけ、Z-driven 区間だけ救えるかを train-side pseudo-tail で監査する。

## 変更点

- exp072 deterministic full replay train cache を固定入力として読む。
- `pf_ancc`、`beam_mean`、`likpf_mean`、`sc_ens`、`hyb`、`pf_z` を候補として復元する。
- `target_slope = -dZ/dMD`、`base_slope = d(candidate)/dMD` の gap から clipped cumulative correction を作る。
- base candidate、alpha、clip、Z slope threshold、candidate disagreement、`pf_z` auxiliary mode を grid 比較する。
- true TVT は candidate RMSE / bucket / by-well / representative well metrics の scoring にだけ使う。
- PF/Beam 再実行、supervised selector、inference port、提出は行わない。

## 検証方針

- 検証面: exp072 train well pseudo-tail candidate cache
- Group: well
- 主比較: `likpf_mean`
- 追加確認: near `000_050`、`1000_plus`、`abs(dZ/dMD)` top quartile、representative wells (`91b301ce`, `ba48188d`, `fef8af96`, `1b1eba53`)、最大 well regression

## 実行入口

- 学習 notebook: `exp140_z_slope_posthoc_correction_on_pfbeam_candidates_train.ipynb`
- 推論 notebook: `exp140_z_slope_posthoc_correction_on_pfbeam_candidates_inference.ipynb`
- Kaggle 準備:

```bash
make prepare-kaggle-notebooks EXP=exp140_z_slope_posthoc_correction_on_pfbeam_candidates EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp140-z-slope-pfbeam-train --title 'exp140 z slope pfbeam train' --run-on-push --strict"
```

## 結果

Kaggle train v2 を完了した。`likpf_mean` が best overall のままで、best Z-slope variant `zsl_likpf_mean_a0p1_c10_z0p1_d5_pfz_agree` は RMSE 11.597150618、`likpf_mean` から +0.002252946 悪化した。

within10 も 0.772807479 から 0.772796115 にわずかに悪化し、max well regression は +0.204513805。したがって inference port / submit はしない。

## 所見

Z-driven に見える well は残るが、target-free な小さい slope 補正では global / longtail を安定して改善できなかった。Z 系情報は hard correction ではなく、confidence feature や segment verifier の補助材料に下げる。
