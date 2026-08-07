# exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline

## 状態

- ルート: pf_beam
- 状態: submitted_v4_public_lb_57p063_failed_requirement
- 現在 selected: `known_tvt_fit_full`
- CV: 52.50742292458995
- Public LB: 57.063
- Private LB: -
- 直近 Submit ID: 54458212
- 作成日: 2026-07-06
- 親実験: `discussion711308_dz_dtvt_bpeak_cluster_baseline` backlog

## 仮説

discussion 711308 の `dTVT ~= a*dZ+b` と `b` peak cluster は、現行 ML anchor には届かなくても、formation level / offset の spatial coherence を no-ML baseline として説明できる可能性がある。

## 変更点

- v1 は `dTVT/dMD ~= a*dZ/dMD+b` の rate-fit と exact typewell / b-peak / XY nearest assignment を実装し、Public LB 41.214 で失敗した。
- v2 は discussion 本文に寄せて row-step `dTVT ~= a*dZ+b` に変更し、X/Y/Z + last-300 TVT/Z feature-nearest と prefix-holdout source selector を追加した。
- inference v2 は train-side best の `prefix_holdout_source_b_fixeda_h600` を selected とした。
- v3 は feature-nearest 近似ではなく、full X/Y/Z well geometry と last-300 TVT/XYZ shape samples で deterministic cluster を作り、cluster/local source の `a,b` を使う variant を追加した。
- v4 は source / cluster `a,b` を選ばず、各 query/test well 自身の known `TVT_input` 全体で `dTVT ~= a*dZ+b` を fit し、last known `TVT_input` から unknown suffix を累積予測する `known_tvt_fit_full` を selected にした。

## 結果

| version | selected | CV | Public LB | ref |
| --- | --- | ---: | ---: | --- |
| v1 | `exact_typewell_peak_xy_k8` | 81.7364272463997 | 41.214 | 54395246 |
| v2 | `prefix_holdout_source_b_fixeda_h600` | 35.41055512960111 | 34.908 | 54396544 |
| v3 | `discussion_fullxyz_cluster_holdout_ab_k24_h300` | 35.30041735041327 | 29.193 | 54408573 |
| v4 | `known_tvt_fit_full` | 52.50742292458995 | 57.063 | 54458212 |

v3 はディスカッション文により近い明示的 cluster 実装へ修正し、Public LB は v2 の 34.908 から 29.193 へ改善した。ただし要件の LB 約 12.8 にはまだ遠い。
v4 はユーザー指定の「test known TVT で fit して未知 test に transform する」直接経路として実行したが、CV 52.5074 / Public LB 57.063 で v3 より悪化した。

## 検証方針

- Fold: leave-one-well-out pseudo-tail audit
- Group: well
- Score rows: `TVT_input` missing suffix
- Leakage guard: validation target well は source pool から除外する。query assignment は known prefix、MD/X/Y/Z、typewell exact hash、last-300 TVT/Z summary だけを使い、target tail true TVT は使わない。
- Test inference: v4 selected は test known `TVT_input` のみで `a,b` を fit し、test tail true TVT は使わない。fit 不能時だけ train source full-fit `a,b` median に fallback する。

## 所見

- `b` peak は二峰性として検出できるが、target-free に test tail の offset を選ぶ signal としては弱い。
- visible prefix holdout で source `b` を選んでも、hidden tail の drift / level offset へ十分に転移しなかった。
- v4 の direct known-TVT fit は hidden tail への外挿が弱く、採用しない。

## 実行入口

- 学習 notebook: `exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline_train.ipynb`
- 推論 notebook: `exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline_inference.ipynb`
- notebook 実行: Kaggle kernel run を正とする。ローカル実行は `--allow-local` を付けた smoke debug のみに限定する。
