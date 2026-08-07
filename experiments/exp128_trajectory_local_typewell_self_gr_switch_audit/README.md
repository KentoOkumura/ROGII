# exp128_trajectory_local_typewell_self_gr_switch_audit

## 状態

- ルート: ensemble
- 状態: completed_train_side_rejected_no_submit
- CV: 11.594897672217703
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-06-25
- 親実験: exp099_pf_multi_observation_likelihood_probe

## 仮説

同一 trajectory 内でも、typewell GR による候補軌跡の観測コストが高い局所窓では、同じ horizontal well の既知 prefix にある GR motif への self-match が補助 prior になる可能性がある。ただし exp091 で self-GR 候補単体は大きく悪化しているため、hard switch は self-GR cost が typewell cost を十分に上回る局所窓に限定し、主目的は提出ではなく診断とする。

## 変更点

- exp099 の PF/Beam / likelihood-PF train-side candidate cache を入力にする。
- 各 well の評価区間 window で `typewell_cost` と visible prefix への `self_cost` を計算する。
- `self_gr_prefix_prior_tvt` と、`likpf_mean` / `pf_ancc` / `beam_mean` / `sc_ens` / `hyb` に対する局所 hard switch / soft blend 候補を生成する。
- switch 判断には真値を使わず、cost gap、self cost、prefix window coverage だけを使う。

## 検証方針

- Fold: exp099 OOF cache と同じ train-side 疑似 tail rows を使う。
- Group: well 単位。
- Stratification: なし。
- Leakage Check: self-GR source は同じ well の visible prefix `TVT_input` のみ。評価区間の true TVT は scoring 後にだけ使う。typewell cost は候補 TVT と typewell GR から作り、target TVT は参照しない。

## 実行入口

- 学習 notebook: `exp128_trajectory_local_typewell_self_gr_switch_audit_train.ipynb`
- 推論 notebook: `exp128_trajectory_local_typewell_self_gr_switch_audit_inference.ipynb`
- Kaggle 準備: `task prepare-kaggle-notebooks EXP=exp128_trajectory_local_typewell_self_gr_switch_audit`
- notebook 実行: Kaggle kernel run を正とする。ローカル実行は `--allow-local` を付けた smoke debug のみに限定する。

## 結果

| メトリック | 値 |
| --- | --- |
| CV | 11.594897672217703 |
| Public LB | - |
| Private LB | - |

## 所見

### 良かった点

- v2 は 3,783,989 rows / 773 wells で完了し、candidate / bucket / by-well / signal / window diagnostics を保存した。
- v1 の coverage bug を修正し、v2 では全候補が coverage 1.0 の公平な比較になった。

### 悪かった点

- best は baseline `likpf_mean` と同値で、local switch / blend は一度も発火しなかった。
- `local_cost_gap_typewell_minus_self` の平均は -0.742966 で、self-GR cost が typewell cost より悪い。
- `self_gr_prefix_prior_tvt` は worst-well で数千 ft 規模の悪化があり、直接候補として使えない。

### リスク / 注意

- exp091 では self-GR direct candidate が大きく悪化している。global RMSE が改善しても worst-well regression と hidden-like stress を確認するまで推論化しない。
- v2 の結果により、この実験から raw-test parity / inference port / submission へは進めない。

## 次

- self-GR path 直接利用は閉じる。
- self-GR 由来情報を使う場合は、`self_gr_multiscale_longtail_gate` のような補助 confidence に限定し、high-drift / PF-dense disagreement gate の一部として扱う。

## 表記

用語は `KAGGLE_DIRECTION.md` の表記方針と `docs/glossary.md` に合わせ、実験名や設定名を除いて日本語優先で記録する。
