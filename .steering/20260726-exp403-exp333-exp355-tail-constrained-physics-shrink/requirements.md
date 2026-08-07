# exp403 要件

## 依頼

最終TVT予測を改善するため、保存済みexp263物理blendのK16成分をexp333、
exact-HMM成分をexp355へ同時置換し、tail制約付きcross-fit shrinkで
exp263へ戻す結論を検証可能な実験として設計する。

今回は以下だけを行う。

- `KAGGLE_DIRECTION.md`のバックログ追加
- steeringの要件・設計・タスクリスト確定
- `experiments/exp403_exp333_exp355_tail_constrained_physics_shrink/`
  のdesign-only scaffoldと設計記録作成

実装、Notebook採用、test、Kaggle package、push、run、inference、
submissionは行わない。

## 2026-07-26 実装承認

初回design-only依頼は完了済み。その後、ユーザーの
`exp403を実装してください`という指示により、凍結済み設計の実装だけを承認済みと
する。別名compact self-contained Jupytext train/inference候補と専用testを作る。
正規Notebook採用、Kaggle package / push / run、inference有効化、submissionは
引き続き別承認とする。

## 制約

- Route: `ensemble`
  - exp333の保存済みLightGBM segment補正と、exp226 / exp355 / exp209 /
    exp072 LikPFの物理候補が予測生成に本質的に寄与するため。
- 親: `exp263_last_anchor_better_candidate_confidence_pair_cache`
- 保存済みOOFをload-onlyで使用し、親候補、PF、HMM、LightGBMを再実行しない。
- 変更するのは、保存済み2成分の置換と、exp263へ戻すshrink係数だけとする。
- exp263の固定式、各source prediction、fold、truth、hidden-like roleは変更しない。
- exp226由来reporting foldとexp263 generation foldは独立ledgerとして扱い、
  fold labelの一致を要求しない。
- suffix truthは入力identity、4候補prediction、raw correction、formula、
  content SHAをfreezeした後だけ読む。
- λはouter-trainだけで固定候補集合から選び、outer-valid scoreで選び直さない。
- Public LBは設計、λ、gate、採用判断に使わない。
- `docs/06_reproducibility.md`に従い、gzipはdecompressed content SHAを
  主証拠にする。

## 検証対象

保存済み予測から次を再構成する。

```text
exp263 =
    0.50 * exp226_k16
  + 0.25 * likpf_mean
  + 0.25 * exp209_exact_hmm

full_replacement =
    0.50 * exp333_k16_segment_corrected
  + 0.25 * likpf_mean
  + 0.25 * exp355_k16_rate_prior_hmm

crossfit_shrink =
    exp263 + lambda_fold * (full_replacement - exp263)
```

## 受け入れ基準

- steering、config、README、SESSION_NOTES、result、metricsに未記入値が残らない。
- `experiment.route=ensemble`、`status=implementation_complete_not_run`である。
- implementationはtrue、canonical Notebook adoption / training / inference /
  submission / Kaggle run flagはすべてfalseである。
- 設計上の実行量が
  `1 policy / 9 calibration lambdas / 5 reporting folds /
  0 model / 0 booster / 0 PF / 0 HMM / 0 parent rerun`
  と明記されている。
- λ候補、選択順、tie-break、fallback、promotion gate、failure actionが一意である。
- exp226 reporting foldとexp263 generation foldの独立性が明記されている。
- 保存入力のSHA、行数、well数、prediction列、禁止列が固定されている。
- `make validate-exp EXP=exp403_exp333_exp355_tail_constrained_physics_shrink`
  がdesign-only scaffoldとしてPASSする。
- deterministic anchorとは呼ばず、将来実行時にinput / formula / prediction /
  gate content SHAとKaggle kernel versionを記録する設計になっている。
