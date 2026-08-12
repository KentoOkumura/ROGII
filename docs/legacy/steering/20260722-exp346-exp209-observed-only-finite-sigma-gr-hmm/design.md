# 設計

## 仮説

exp209の広いGR scaleを一律に縮めるのではなく、補間前raw GRがfiniteの行だけfinite-prefix stdへ狭めれば、欠損・補間行の過信防止を維持したまま、実測GRの識別力を強められる。

## アプローチ

exp209のknown-prefix scaleとexp307/337で確認したfinite-only scaleを同じwell内で2本計算し、raw GR missing maskだけで行ごとに切り替える。

```text
sigma_base,w = clip(std(fillna(GR_known, 0) - TW_GR(TVT_input)), 10, 60)
sigma_obs,w  = clip(std(GR_known[finite] - TW_GR(TVT_input)[finite]), 10, 60)

sigma_w,t = sigma_obs,w   if raw GR_w,t is finite
            sigma_base,w  if raw GR_w,t is missing
```

finite pairが20未満、または`std`がnonfiniteなら`sigma_obs,w = sigma_base,w`としてno-op fallbackする。HMMへ渡すGR値はexp209と同じlinear interpolation both directions + Type Well mean fallbackであり、raw maskは補間前にfreezeする。

Gaussian emissionはexp209と同じ`-0.5 * min(z^2, 600)`を使い、log-sigma項を新規追加しない。変更はz-score分母のrow scheduleだけである。raw missing行は値、scale、emissionがexp209 parityとなる。

## 実験範囲

- 対象実験: `exp346_exp209_observed_only_finite_sigma_gr_hmm`
- Route: `pf_beam`
- 科学的親: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- 変更する変数: raw finite evaluation行の`gr_sigma`をwell別finite-only population stdへ置換する。
- 固定する変数: raw missing行のexp209 scale、GR補間値、Type Well処理、`a=1,b=0`、Gaussian形状、41 rate states、`step=0.35`、`sig_r=0.002`、`sig_p=0.02`、momentum、prior、position floor、posterior mean。
- 比較基準: 保存済みexp209 raw HMM `11.9382872349 ft`、fixed LikPF 50:50 `10.2696961466 ft`。
- 失敗根拠: exp307の全行finite stdは`14.2097176758 ft`へ悪化したが、exp337ではfinite-only forward NLLが構造分散追加・zero-fillより両originで良かった。
- 実行量: 1 schedule audit、1 candidate、773 HMM well-runs。model/LightGBM config/trained fold/booster/PF/Beam/control再実行は0。

## 検証方法

1. raw well identity、exp209 saved HMM/LikPF、exp226 fold、exp115 hidden-likeのSHA・row・well・columnをHMM前にpreflightする。
2. unknown-suffix truthを読まず、raw finite mask、`sigma_base`、`sigma_obs`、fallback、row scheduleをfreezeし、schema SHAとdecompressed content SHAを保存する。
3. exp209 exact-HMMと同じstate/transition/priorでcandidate 1 variantだけを実行し、prediction content SHAをfreezeする。
4. その後だけexp226 truth/foldとhidden-like assignmentをjoinし、direct/fold/stress/tail/fixed blendを読む。
5. direct `>=0.05 ft`改善、4/5 folds、raw observed rows `>=0.05 ft`改善、raw missing rows non-regression、missing-fraction high bucket non-regression、1000+、hidden-like 2面、p95 non-regression、worst `<=+0.25 ft`、fixed LikPF blend non-regressionをAND gateにする。

raw missing fraction bucketは未知truthと独立に`[0,0.10) / [0.10,0.30) / [0.30,1.00]`へ固定する。gate FAIL時は枝を閉じ、同一結果上の救済を行わない。PASSしてもinference/submissionは同じexp内の別承認とする。

## 既存実験との差

- exp307: finite scaleを全evaluation行へ適用した。本実験はraw missing行をexp209のままにする。
- exp308: exp307 finite-MADを親にしてmissing行の尤度weightを下げる設計。本実験はexp209を直接親にし、observed行だけstd幅へ狭める。
- exp337: well全体へ構造分散を加えた。本実験は構造分散を使わない。
- exp341: exp281上でmissing行へ補間分散を加える条件付き設計。本実験はexp209上のobserved行だけを変更し、exp339/341に依存しない。

## 再現性設計

- seed policy: RNGなし。well ID、raw row、variant、集約順を固定する。
- stochastic処理、PF/Beam、likelihood-PF、seed bagging、GPU学習: なし。
- CPU、internet off、`outer_workers=2`、Numba threads 2を開始点とする。
- raw mask、scale audit、row schedule、prediction、metricsはschema SHAとdecompressed content SHAを記録する。
- 保存済みexp209/exp226/exp115入力のexpected SHAを実装前に固定する。
- model manifest/model SHA、submission SHAは非該当。candidateをdeterministic submission anchorとは扱わない。
- Kaggle package時はcanonical metadataとbootstrap内config/source SHAを照合する。

## リスク

- raw finiteは「補間されていない」ことしか保証せず、Type Wellと正しく対応する保証はない。observed行の過信が残る可能性が高い。
- evaluation行の多数がraw finiteならexp307のglobal narrowingへ近づき、path全体を悪化させる可能性がある。
- rowごとのscale切替がmissing境界でposteriorを急変させる可能性があるため、raw observed/missingとmissing-fraction bucketを別々にguardする。
- exp209 saved controlとのruntime/code差があるため、baseline prediction parityとSHAを先に確認する。

## 優先度

Late phaseの中・P2。0-HMMのexp339と0-boosterのexp340より後、同系の低優先robust-emission案より前とする。exp339/341のmissing分散枝とは独立に判定する。
