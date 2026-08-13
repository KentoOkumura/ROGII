# exp374_exp209_student_t_exact_hmm_emission

## 状態

- ルート: `pf_beam`
- 状態: terminal close（by-well tail gate FAIL、no rescue）
- CV / Public LB / Private LB: 11.720478702 / なし / なし
- 作成日: 2026-07-24
- 親実験: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- 優先度: 低・P4・CPU
- Kaggle kernel:
  `kentookumura/exp374-exp209-student-t-exact-hmm-emission-train`
  version 1、id_no `128436182`

## 仮説

exp209のabsolute-TV​T exact HMMでは、Gaussian二乗誤差が一部の大きなGR残差を
過大評価し、posteriorを誤ったmodeへ固定している可能性がある。HMM本体とsigmaを
変えず、行別emissionだけを固定`df=4` Student-tへ置換すれば、この影響を抑えて
direct path RMSEを改善できるかもしれない。

## 変更点

- Gaussian control:
  `-0.5 * min(z^2, 600)`
- Student-t candidate:
  `-0.5 * (df + 1) * log1p(z^2 / df)`、`df=4`
- 変更するのは上記emissionだけ。
- absolute TVT、grid、41 rate states、transition、prior、sigma、GR欠損処理、
  Type Well GR、momentum、likelihood weight、posterior meanはexp209のまま固定する。
- exp226の`tvt_geop`、residual-offset座標、exp342のshift-rank Stage 0は使わない。
- exp209 Gaussian controlは保存済み予測とSHAを使い、HMMを再実行しない。

## 検証方針

- 将来実行量: Student-t 1 variant、773 HMM well-runs、model/config/fold/boosterは0。
- primary control: exp209 Gaussian exact HMM
  `RMSE 11.938287234887435`。
- secondary control: fixed LikPF/HMM 50:50
  `RMSE 10.269696146642758`。
- candidate predictionとlogical content SHAをfreezeしてからtruthとscopeをjoinする。
- technical gateはSHA、control parity、773 wells / 3,783,989 rows、ID/order/fold、
  finite coverage、posterior normalization、truth-late joinをすべて要求する。
- scientific gateはdirect `>=0.05 ft`改善、4/5 folds改善、observed GR改善、
  missing/high-missing/1000+/hidden-like/tail/fixed blendの非悪化をANDで要求する。
- FAIL時は`student_t_exp209_failed_close_without_rescue`として閉じ、dfやscale、
  transition、grid、blend weightなどを同じOOFで調整しない。

## 実行入口

- `*_compact_selfcontained_train.py` / `.ipynb`:
  Student-t exact HMM、target-free prediction freeze、late truth/control join、
  scope metrics、fail-closed gateを実装した候補。
- `*_compact_selfcontained_inference.py` / `.ipynb`:
  train-side PASSと別承認まで必ず停止するfail-closed候補。
- 正規`*_train.ipynb`へcompact self-contained候補を採用済み。
- 正規inference Notebookは既存placeholderのまま維持する。
- Kaggle package/push/runは承認済み。inference、submissionは別承認が必要。

## 結果

| メトリック | 値 |
| --- | --- |
| Student-t direct CV | 11.720478702 |
| Gaussian direct control | 11.938287235 |
| direct改善 | +0.217808533 ft、4/5 folds |
| Student-t fixed 50:50 | 10.125385545 |
| by-well delta p95 | +0.982661 ft（FAIL） |
| worst-well delta | +35.015963 ft（FAIL） |
| Public LB | 未提出 |
| Private LB | 未提出 |

## 所見

### 設計上の判断

- exp342はexp281のresidual-offset HMMを親にした別仮説であり、本実験の親ではない。
- exp342の結果はnegative referenceとして扱うが、exp209 direct HMMの代用結果にはしない。
- 0-HMM proxyを先行gateにせず、承認後は固定Student-t 1件だけをfull HMMで直接評価する。
- 実装はexp346のcompact self-contained監査構成を参照したが、有限sigma変更は持ち込まず、
  exp209 zero-fill population stdを全rowで固定した。
- synthetic wellでは親exp209 `run_hmm2(emission="t", df=4)`とmean/std/loglikが一致し、
  exact forward-backward kernelも同一emission入力で親と一致した。

### リスク / 注意

- heavy-tail化は外れ値だけでなくwrong stateへの罰も弱めるため、RMSEやtailが悪化し得る。
- exp209系emission変更にはnegative結果が複数あり、現行P1/P2より優先しない。
- train実行は固定1 variant / 773 HMM well-runsだけとし、controlを再実行しない。
- overall、raw missing、高missing、1000+、hidden-like 2面、fixed blendは改善した。
- 343/773 wellsが悪化し、p95とworst-well gateをFAILしたため採用しない。

## 参照

- steering:
  `docs/legacy/steering/20260724-exp374-exp209-student-t-exact-hmm-emission/`
- 再現性:
  `docs/06_reproducibility.md`
- バックログ:
  `backlog/KAGGLE_DIRECTION.md`

## 次

`student_t_exp209_failed_close_without_rescue`として閉じる。df/scale/temperature/
grid/blend救済、inference、submission、同family backlog追加は行わない。
