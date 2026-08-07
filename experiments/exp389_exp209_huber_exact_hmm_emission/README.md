# exp389_exp209_huber_exact_hmm_emission

## 状態

- ルート: `pf_beam`
- 状態: train完了 / tail gate FAIL / 救済なしでterminal close
- CV: `11.852741129500146`
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-07-24
- 親実験: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- 優先度: 低・P4・CPU

## 仮説

exp209のabsolute-TVT exact HMMでは、Gaussian二乗損失が一部の大きなGR残差を
過大評価し、posteriorを誤modeへ固定している可能性がある。HMM本体とsigmaを変えず、
行別emissionだけをfixed Huber `delta=1.345`へ置換すれば、exp209 Gaussian
direct pathを安全に改善できるかを検証する。

## 変更点

- control:
  exp209 capped Gaussian `-0.5*min(z^2,600)`、saved RMSE
  `11.938287234887435`。
- candidate:
  Huber `delta=1.345`。quadratic center / linear tail、追加clipなし。
- 変更はrow emissionだけ。
- absolute TVT、grid、41 rate states、transition、prior、sigma、missing-GR、
  Type Well GR、momentum、likelihood weight、posterior meanをexp209のまま固定する。
- exp209 Gaussian controlはsaved predictionとSHAを使い、HMMを再実行しない。
- exp357のexp281 residual-offset HMM、`tvt_geop`、shift-rank Stage 0は使わない。

## 検証方針

- Fold: exp226保存identityからreporting 5 foldsのみを使う。
- Group: well単位、773 wells / 3,783,989 unknown-suffix rows。
- Primary:
  exp209比`>=0.05 ft`、4/5 folds、raw observed `>=0.05 ft`。
- Safety:
  missing/high-missing/1000+/hidden-like 2面非悪化、by-well p95 `<=0`、
  worst `<=+0.25 ft`。
- Secondary:
  saved LikPFとのfixed 50:50をGaussian fixed 50:50から非悪化。
- Leakage:
  candidate predictionとlogical SHAをfreezeしてからtruth/error/scopeをjoinする。
- 実行量:
  1 Huber variant / 773 HMM runs / model・trained fold・booster・PF・Beam・
  parent control rerun各0。

## 実行入口

- compact self-contained train実装:
  `exp389_exp209_huber_exact_hmm_emission_compact_selfcontained_train.py/.ipynb`
- fail-closed inference候補:
  `exp389_exp209_huber_exact_hmm_emission_compact_selfcontained_inference.py/.ipynb`
- compact trainを正規train notebookへ採用済み。正規inference notebookは
  template placeholderのままである。
- helperは作らず、train実装内にexp209固定HMM、Huber emission、SHA、
  truth-late join、readout、AND gateを自己完結実装した。
- 専用テスト9件、構文、Ruff F821/F401/F841、Jupytext round-tripをPASSした。
- Kaggle private CPU version 1を
  `kentookumura/exp389-exp209-huber-exact-hmm-emission-train`で完了した。
- inferenceとsubmissionは未承認で、無効のままである。

## 結果

| メトリック | 値 |
| --- | --- |
| CV | 11.852741130 |
| Public LB | - |
| Private LB | - |

- Gaussian control `11.938287235`から`+0.085546105 ft`改善。
- 5/5 folds、raw observed/missing、高missing、1000+、hidden-like 2面、
  fixed LikPF/HMM 50:50はいずれも改善。
- 411/773 wells改善、362/773 wells悪化。
- by-well delta p95 `+0.002234 ft`、worst well `00bbac68`
  `+1.750248 ft`で事前登録tail gateをFAIL。
- decision: `huber_exp209_failed_close_without_rescue`

## 所見

### 設計判断

- exp374は同じexp209親のStudent-t siblingとしてHMM固定契約だけを参照する。
- exp374の結果からHuber deltaやgateを調整しない。
- exp357は誤スコープ履歴として保持するが、そのRMSEは本実験の根拠に使わない。
- 0-HMM proxyを置かず、fixed Huber actual HMMだけを直接評価した。
- average/fold/scope改善は再現したが、少数wellの安全性を満たさないため採用しない。

### リスク / 注意

- robust tailがwrong stateへの罰も弱め、平均改善とwell-tail悪化が併存し得る。
- exp374は平均改善でもp95/worstをFAILしたため、tail gateをhard ANDにする。
- FAIL後のdelta/scale/clip/temperature/sigma/transition/grid/blend救済は禁止する。
- PASSしてもinference/submissionへ自動移行しない。

## 次

- 固定no-rescue契約に従い、delta/scale/clip/temperature/sigma/transition/grid/
  prior/blendの救済や再実行は行わない。
- inferenceとsubmissionを行わず、同familyの新規backlogも追加しない。

## 表記

用語は `KAGGLE_DIRECTION.md` の表記方針と `docs/glossary.md` に合わせ、実験名や設定名を除いて日本語優先で記録する。
