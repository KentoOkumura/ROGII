# exp346_exp209_observed_only_finite_sigma_gr_hmm

## 状態

- ルート: `pf_beam`
- 状態: Kaggle CPU version 1完了・scientific gate FAIL・branch closed
- CV: candidate `13.295027` / exp209 control `11.938287` / 改善`-1.356739 ft`
- Public LB / Private LB / Submit: なし
- 作成日: 2026-07-22
- 親実験: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`

## 仮説

exp209の広いGR幅を全行で狭めると過信するが、補間されていないraw finite GR行だけならfinite-only幅へ狭めても有効な識別力を増やせる。raw missing・補間行をexp209幅のままにすれば、exp307で見られた全区間過信を避けられる可能性がある。

## 変更点

- raw finite行: known-prefix finite residualのpopulation stdを使用する。
- raw missing行: exp209 zero-fill stdを完全維持する。
- finite pair 20未満またはnonfinite時: well全体をexp209 scaleへno-op fallbackする。
- clip `[10,60]`、`a=1,b=0`、GR補間、Gaussian形状、HMM state/transition/prior/outputは固定する。
- 学習型confidence、threshold、係数gridは使わない。

## 検証方針

- Fold / Group: 保存済み5 folds、`well_id`。
- Score: `TVT_input`がNaNのunknown suffixだけ。
- 比較: 保存済みexp209 raw HMMとfixed LikPF 50:50。親controlは再実行しない。
- Gate: direct、4/5 folds、raw observed/missing、missing-fraction high、1000+、hidden-like 2面、p95、worst、fixed blendのAND。
- Leakage: raw mask、scale schedule、prediction content SHAをfreezeした後だけunknown-suffix truthをjoinする。

## 実行量

- 1 schedule audit、1 candidate、773 HMM well-runs。
- model / LightGBM config / trained fold / booster / PF / Beam / control再実行: すべて0。
- CPU、GPU/internet offでversion 1を`17,757.849 sec`で完了した。

## 所見

exp337ではfinite-only幅がknown-prefix forward NLLで最良だった一方、exp307では全行finite-only化がTVT RMSEを大きく悪化させた。本実験はこの2つを分ける最小変更として実行した。

technical gateは773/773 HMM runs、finite 100%、fallback 0%、raw missing emission parity差0、baseline parity、posterior正規化、runtimeを全PASSした。一方、directは`11.938287→13.295027`、改善1/5 folds。raw observed / missing、high-missing、1000+、hidden-like 2面、p95、worst、fixed LikPF 50:50をすべてFAILした。

finite sigma中央値はexp209幅の`0.369701`倍まで縮み、observed行のGR evidenceを過信した。raw missing行のemission自体は同一でも、全系列HMM posteriorを通じてmissing行へ悪影響が波及した。

## リスク / 注意

- raw finiteは「実測」であるだけでType Well対応が正しい保証はなく、依然として過信しうる。
- evaluation行の多くがraw finiteならexp307に近づくため、raw missing行とhigh-missing wellsのnon-regressionを必須にする。
- exp308を再開せず、exp209直系の独立実験として扱う。

## 次

decision `observed_only_finite_sigma_failed_close_without_rescue`に従いbranchを閉じる。sigma/confidence/emission/HMM/blend救済、raw-test inference、submissionは行わない。同familyの新規救済backlogは追加しない。
