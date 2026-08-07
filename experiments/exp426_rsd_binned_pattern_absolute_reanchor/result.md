# exp426_rsd_binned_pattern_absolute_reanchor 結果

## 状態

Stage A version 1はtechnical gate FAILでfail-close完了。scientific評価、
Stage B / C、inference、submissionには進まない。

## 仮説

RSD 0.5 ft bin mean＋Pearson pattern scoreは、pointwise GR比較よりabsolute datum
offsetを識別し、exp226の累積offsetとPFのfinite-support不足を改善できる。

## 設定

- exp226親:
  `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction`
- PF親:
  `exp404_scale5_sigma_gr_likelihood_pf_ablation`
- 検証:
  Stage A identifiability -> Stage B exp226 reanchor ->
  Stage C PF absolute-anchor proposal
- Fold / group: 5 / `well_id`
- seed: Stage A/B RNGなし、Stage C stable per-well SHA256
- parent control再実行: なし

## 実装差分

- exp226 final `tvt_pred`を基準に、固定13 offsetsを512-row blockで生成する。
- primaryはraw finite GRをRSD 0.5 ft bin meanへ集約した
  signed Fisher-Pearsonとする。
- raw pointwise Pearson、exp280互換raw Gaussian、stable permutationを
  同じcandidate bankのmatched controlとする。
- score / support / rank / top-3 / manifest / logical SHAとfixed probeを
  freezeしてからtruthとhidden-like roleを読む。
- technical FAIL時はtruthを読まずfail-closeする。

## 事前登録した判定

- Stage A:
  top-1 exact `>=0.25`、top-3 coverage `>=0.55`、direction `>=0.60`、
  raw Pearson / Gaussian / permutation差、blockwise replay、fold / scopeをAND判定。
- Stage B:
  exp226比`>=0.10 ft`、4/5 folds、persistent SSE`>=10%`削減、
  1000+ / hidden-like / by-well tail guardをAND判定。
- Stage C0:
  exp410 sentinelでsupport外率、episode SSE、truth-near seeds、recapture、
  uniform-anchor control差をAND判定。
- Stage C1:
  exp404比`>=0.10 ft`、4/5 folds、support / episode / scope / by-well guardを
  PF physical-model contribution条件とする。

## 結果

| Stage | 状態 | 結果 |
| --- | --- | --- |
| A | technical FAIL・完了 | supported blocks `25.593939%`、wells `89.262613%` |
| B | 未実装・停止 | Stage A technical FAIL |
| C0 / C1 | 未実装・停止 | Stage A technical FAIL |
| Public / Private LB | 対象外 | - / - |

### Stage A technical gate

- Kaggle:
  canonical private CPU version 1、id_no `128930757`
- inventory:
  3,783,989 rows / 773 wells / 5 folds / 7,787 blocks
- score bank:
  101,231 rows
- supported block fraction:
  `0.255939386 < 0.95`でFAIL
- supported well fraction:
  `0.892626132 < 0.98`でFAIL
- runtime:
  target-free freezeまで`160.789156 sec`、全処理`164.719113 sec`
- peak RSS:
  `0.803265 GB`
- fixed probe:
  well `000d7d20`、logical SHA一致
- truth / hidden-like role read:
  freeze前後とも`0 / 0`

inventory、canonical order、duplicate 0、finite score、固定offset順、
rank permutation、top-3 mask、runtime、memory、probe parityはPASSした。
support 2条件だけがFAILしたため、事前設計どおりtruthを読まずscientific評価を
skipした。

## 再現性

- deterministic anchor: false
- kernel:
  `kentookumura/exp426-rsd-binned-pattern-absolute-reanchor-train` version 1
- scientific contract SHA:
  `2fc2b35c9ea88aa3b9c35546e4979dbb335d9944a225c601b4810711d6c164ca`
- score content SHA:
  `463aa32bef9a1045469466e2cf5fd68e038258e75f11fc88153fd9ca7f8dd2fd`
- input manifest content SHA:
  `7933f0f2babaa382ee23ae64db096db0dcc775035fc399254e64e7b30fe7656b`
- fixed probe logical SHA:
  `b4fd1730c8c2c53a07a0e326bc762b52f1c4249b226e2546f30544b215069c59`
- model / submission SHA: 非該当
- output archive:
  CVや後段入力に使う実ファイル確認が不要なため取得していない。Kaggle logs /
  notebook cell outputを結果根拠とする。

## 解釈

RSD 0.5 ft bin mean scoreは、固定support条件の下で全block / wellへ十分な
coverageを持たなかった。これはoffset識別精度のnegative結果ではなく、
識別性を評価する前提となる観測supportのtechnical FAILである。

同じOOFでbin幅、block、offset、minimum points / paired bins、Type Well
extrapolation、score familyを緩める救済は禁止する。Stage AをHMM
residual-datum stateやPF proposalの根拠には使わない。

## 次

exp426をterminal closeする。Stage B / C、inference、submissionは行わず、
同じRSD-binned score familyの救済backlogも追加しない。
