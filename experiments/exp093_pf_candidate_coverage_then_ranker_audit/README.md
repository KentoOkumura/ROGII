# exp093_pf_candidate_coverage_then_ranker_audit

## 状態

- ルート: pf_beam
- 状態: implemented_pending_kaggle_train
- CV: -
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-06-20
- 親実験: `exp091_self_gr_likelihood_pf_beam_probe`

## 仮説

PF/Beam/likelihood-PF の候補集合が真値近傍候補を十分に含んでいる bucket では、直接置換ではなく supervised candidate ranker / N-way classifier に落とせる可能性がある。一方で coverage 自体が低い bucket では、ranker ではなく候補生成側の失敗として扱う。

## 変更点

- exp072 deterministic full replay train cache を固定入力として読む。
- primary candidate set は `pf_ancc`、`beam_mean`、`likpf_mean`、`sc_ens`、`hyb`。
- ablation として exp091 由来の `self_gr_ens`、`self_gr_best`、`self_gr_sc8/15/25` を追加する。
- candidate set 別 oracle coverage、target-free rank score coverage、bucket 別 weak coverage、ranker readiness recommendation を保存する。

## 検証方針

- Fold: train-side pseudo-tail audit の既存 cache に従う
- Group: well
- Stratification: distance / tail rank / eval length / PF-Beam disagreement / likPF delta / PF seed std / exp056 / exp083 well context
- Leakage Check: true TVT は coverage と oracle headroom の scoring にだけ使い、候補生成や target-free rank score には使わない

## 実行入口

- 学習 notebook: `exp093_pf_candidate_coverage_then_ranker_audit_train.ipynb`
- 推論 notebook: `exp093_pf_candidate_coverage_then_ranker_audit_inference.ipynb`
- Kaggle 準備: `task prepare-kaggle-notebooks EXP=exp093_pf_candidate_coverage_then_ranker_audit EXTRA_ARGS="--notebook train --run-on-push --strict"`
- notebook 実行: Kaggle kernel run を正とする。ローカル実行は `--allow-local` を付けた smoke debug のみに限定する。

## 結果

未実行。Kaggle train 完了後に `result.md` と `metrics.json` を更新する。

## 所見

### 良かった点

- ranker 学習前に oracle coverage と生成側失敗 bucket を分けて読める構成にした。
- exp091 の self-GR 候補を直接置換ではなく candidate-set ablation として扱う。

### 悪かった点

- full train cache はローカルにないため、実測は Kaggle train notebook 待ち。

### リスク / 注意

- true TVT を oracle scoring に使うため、結果は診断限定であり、そのまま selector として使わない。
- candidate long table は大きくなるため、ローカル smoke は `EXPERIMENT_DEBUG=1` の row cap 前提にする。

## 次

1. `validate-exp` と Kaggle package 生成を通す。
2. Kaggle train notebook を実行する。
3. summary JSON の recommendation に従い、ranker 実験へ進むか候補生成失敗地図へ戻るかを決める。
