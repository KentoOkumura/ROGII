# exp400_all_well_1p3_sigma_gr_likelihood_pf

## 状態

- ルート: PF/Beam
- 状態: train-side gate FAIL・branch閉鎖
- CV: 12.221811
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-07-25
- 親実験: `exp072_exp063_full_replay_feature_cache`

## 仮説

exp072 likelihood-PFはknown prefixで推定したGR residual scaleを観測尤度に使う。
このscaleを全wellで一律1.3倍しGR evidenceの過信を弱めると、particleの
mode collapseや誤ったGR対応への追従を減らし、`likpf_mean`のunknown-suffix
RMSEを改善できる可能性がある。

## 変更点

変更は1点だけ。

```text
gs_base = clip(prefix zero-filled GR residual population std, 10, 60)
gs_candidate = 1.3 * gs_base
```

- 全773 wellsへ適用する。
- clip後に1回だけ掛け、再clipしない。有効範囲は`[13,78]`。
- 500 particles、128 stable seeds、scale 3/5/8/12、PF dynamics、
  likelihood、resampling、補間、aggregationはexp072のまま。
- primaryは `likpf_mean`。scale outputsは診断用で、best選択しない。
- 保存済みexp072 controlは再実行しない。

## Discussionとの関係

Discussion 728712で共有された `lik_pf` の `gs * 1.3` を直接検証する。
ただし、リンク先の現行Notebookには別のselector用PFもあり、そちらの`gs`はx1.0のまま。
そのため本実験は公開Notebook全体のscore再現ではなく、exp072と同型の
likelihood-PFへの1変数ablationである。

## 検証方針

- Fold: exp226の保存5 reporting folds
- Group: `well_id`
- Primary: saved exp072 `likpf_mean`比RMSE gain
- Leakage: horizontal suffix `TVT`を読まずcandidateとcontent SHAをfreezeし、
  その後だけtruth / fold / hidden-like roleをjoin
- Promotion: overall `>=0.05 ft`、4/5 folds、raw observed、
  missing/high-missing、1000+、hidden-like 2面、by-well p95/worst、
  fixed exp209-HMM 50:50を全AND判定
- Control: saved exp072 / exp209をload-only、parent PF・HMM再実行0

## 設計上の実行量

- scientific variant: 1
- candidate PF well-runs: 773
- 128 seeds/well、98,944 seed-well trajectories
- 500 particles/seed、49,472,000 particle starts
- prediction readouts: 5
- reporting folds: 5
- LightGBM config / trained fold / booster: 0 / 0 / 0
- control PF / HMM / Beam rerun: 0 / 0 / 0
- CPU、GPU/internetなし、runtime上限30,600秒

Kaggle private CPU version 1で計画どおり実行した。

## 実行入口

- 学習 notebook:
  `exp400_all_well_1p3_sigma_gr_likelihood_pf_train.ipynb`
  （正規Notebookとして採用・version 1実行済み）
- compact self-contained train候補:
  `exp400_all_well_1p3_sigma_gr_likelihood_pf_compact_selfcontained_train.ipynb`
- 推論 notebook:
  `exp400_all_well_1p3_sigma_gr_likelihood_pf_inference.ipynb`
  （template placeholderのまま）
- fail-closed compact inference候補:
  `exp400_all_well_1p3_sigma_gr_likelihood_pf_compact_selfcontained_inference.ipynb`
- Jupytext source / 専用test: 作成済み
- helper: なし
- Kaggle package: `kaggle/train`
- Kaggle output: `kaggle/output/train_v1`

## 結果

| メトリック | 値 |
| --- | --- |
| Candidate CV | 12.221811 |
| Saved exp072 control | 11.594894 |
| Improvement | -0.626917 ft |
| Fixed HMM 50:50 candidate / control | 10.659968 / 10.269693 |
| Public LB | - |
| Private LB | - |

## 所見

### 良い点

- Discussionの変更対象だったPF familyへ直接適用する。
- 既存の決定論的exp072 seed policyとsaved controlを使い、1変数だけを比較できる。
- 同一PF runから5 readoutsを保存でき、追加trajectoryコストがない。

### リスク / 注意

- Discussionにはscore、CV、version、active pathの詳細がない。
- x1.3でresampling分岐が変わるため、同じseedでもtrajectoryは大きく変わり得る。
- 固定exp072 cacheにはx1.0のscale 3/5/8/12列がないため、scale別readoutは
  x1.3 candidate-only診断になる。primary `likpf_mean`比較には影響しない。
- exp072 direct PFはCV 11.594898、Public LB 9.721で、現在の最良ML/ensemble anchorではない。
- exp398 HMMのx1.3に続き、PFのx1.3も悪化した。global scale multiplier
  familyを追加救済しない。

### 実行後

- technical gateはPASS。773 wells、98,944 seed-well、49,472,000 startsを
  fallbackなしで完走した。
- scientific gateはFAIL。改善は1/5 folds、305/773 wellsだけで、
  required stress scopeはすべて悪化した。
- candidate-only scale 3/5/8/12はprimary rescueへ使わない。
- prediction logical SHAは
  `009a1d73e187c4126a70231214f14fbe1ae44edee47d9a166818ab1bd928a3bf`。

## 次

全well一律x1.3 branchは救済なしで閉じる。inference / submissionへ進めない。
必要なら保存済みwell auditだけを使う0-PF failure-attributionを別設計にする。
