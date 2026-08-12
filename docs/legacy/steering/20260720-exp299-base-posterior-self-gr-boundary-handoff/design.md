# 設計

## 目的と位置付け

exp296ではcandidate state outside supportをexact 0にする実装自体はtechnical 12/12 PASSしたが、RMSEはexp223 `11.349943 -> 12.159749`、outside scope `+2.341425 ft`、worst-well `+39.687791 ft`で悪化した。unknown suffixのoutside 1,459,531 rowsはすべて`known_tvt_max`より上であり、insideだけに正boostを残すstate-wise maskが未来方向への相対障壁になった。

exp299はoutside self-GR禁止を撤回しない。self-GRをbase posteriorのinside条件付きstate順位付けへ限定し、境界接近時はrow全体をType Well exact HMMへ連続的にhandoffする。

## 固定する親実装

- exp223 `hmm_selfgr_boost_only_a070_c100`のraw self-GR surface生成までを固定する。
- HMMはstep `0.35`、41 rate states、rate span `0.10`、`sig_r=0.002`、`sig_p=0.02`、Gaussian Type Well emission、`lam=1.0`、`start_sig=0.75`、`r0_sig=0.01`、band pad `100`、momentum `0.998`、rate center zeroを固定する。
- self-GRはradius 12 rows、offset `[-12,-8,-4,0,4,8,12]`、top-k 5、anchor stride 3、max anchors 128、keep-last 32、sigma TVT 12、alpha 0.07、clip 1.0を固定する。
- visible-prefix finite `TVT_input`、horizontal `GR/MD/Z`、Type Well `TVT/GR`以外をgenerationに使わない。

## Pass A: base-only controller

各wellでexp223と同じgrid、Type Well emission、transitionを使い、self-GR contributionを一切加えずforward-backwardする。

```text
p0_t(s) = posterior from Type-Well-only exact HMM
mu0_t   = sum_s p0_t(s) * grid[s]
```

Pass A posterior、mean、grid、input manifestをSHA freezeしてからPass Bを作る。Pass Aはunknown-suffix truthやvariant predictionを読まない。保存済みexp209 exact HMM meanとrow identityを照合し、最大絶対差`<=1e-5 ft`を要求する。

## Supportとrow handoff

```text
L = min(finite visible-prefix TVT_input)
U = max(finite visible-prefix TVT_input)
S = {s | L <= grid[s] <= U}

m0_t = sum_{s in S} p0_t(s)
d0_t = max(0, min(mu0_t - L, U - mu0_t))
boundary_factor_t = clip(d0_t / 12.0, 0, 1)
g_t = m0_t * boundary_factor_t
```

- `mu0_t`がoutsideまたは境界上なら`d0_t=0`、`g_t=0`。
- boundaryから12 ft以上内側かつbase posterior massがsupport内なら`g_t`は1へ近づく。
- 12 ftは新規tuning値ではなく、exp223 raw surfaceの既存Gaussian sigmaをそのまま使う。
- `g_t`はPass Aからだけ計算し、Pass B posteriorで更新しない。

## Conditional self-GR contribution

exp223 raw boostを次で固定する。

```text
b_t(s) = clip(centered_self_gr_surface_t(s), 0, 1)
r_t(s) = 0.07 * quality_t * b_t(s)
```

outside candidateは常にexact 0とする。inside candidateでは、self-GRがinside/outsideの総尤度比を変えないよう、Pass A posterior条件付きで正規化する。

```text
if m0_t <= eps or g_t == 0:
    C_t(s) = 0 for every state
else:
    z_t = log(
        sum_{s in S} p0_t(s) * exp(g_t * r_t(s))
        / sum_{s in S} p0_t(s)
    )

    C_t(s) = g_t * r_t(s) - z_t  if s in S
             0                    otherwise

emission_exp299_t(s) = emission_typewell_t(s) + C_t(s)
```

これにより次を同時に満たす。

```text
C_t(s outside S) = 0
mu0_t outside or boundary => C_t(all states) = 0
sum_{s in S} p0_t(s) * exp(C_t(s)) = sum_{s in S} p0_t(s)
```

inside contributionはsignedになり得る。これはboost-onlyの強度を変えるためではなく、self-GRをsupport内の条件付きlikelihoodとして扱い、support総量へのpriorを作らないための必須normalizationである。

## Pass B: 1 scientific variant

variant名は`hmm_selfgr_base_posterior_conditional_handoff_a070_c100`とする。Pass Aで凍結した`p0/mu0/g`とconditional contributionをType Well emissionへ加え、同じHMMをもう一度forward-backwardする。

scientific variantは1本だが、HMM decodeはPass A / Pass Bの2回必要である。773 wellsに対して合計1,546 HMM well-runs、LightGBM config / trained fold / boosterは`0 / 0 / 0`、GPU 0、saved exp223/exp296 control再実行0とする。

## Leakageとfreeze順序

1. raw horizontalから`MD/Z/GR/TVT_input`、Type Wellから`TVT/GR`だけを読む。
2. Pass A base posterior/meanを生成しSHA freezeする。
3. exp223 raw self-GR surface、support、row gate、conditional contributionを生成しSHA freezeする。
4. Pass B predictionを生成しSHA freezeする。
5. ここまでunknown-suffix `TVT`、saved exp223 target、exp296 target/predictionを読まない。
6. freeze後にだけtruthとsaved controlsをrow identity one-to-one joinしてmetricsを作る。

reporting foldはstable SHA256 well hash modulo 5。exp115 hidden-like assignmentはmetricsだけに使う。

## Technical hard gate

- input wells / output wells `773 / 773`、row count `3,783,989`、finite coverage 1.0。
- Pass A HMM config parity、saved exp209 exact HMM row identity、mean max abs delta `<=1e-5 ft`。
- exp223 raw self-GR surface / quality / config parity before handoff。
- outside candidate contribution max abs `0.0`。
- base mean outsideまたは境界上のrowでall-state contribution max abs `0.0`。
- row gate finite、`0 <= g_t <= 1`、境界factorの定義一致。
- conditional support-mass relative error max `<=1e-6`。
- Pass A freeze前truth access 0、Pass B freeze前truth/control access 0。
- active scientific variant 1、Pass A / B HMM well-runs `773 / 773`、LightGBM config / trained fold / booster / parent-control retraining `0 / 0 / 0 / 0`。

## Performance hard gate

primary controlは保存済みexp223 `hmm_selfgr_boost_only_a070_c100`とする。exp296はnegative comparison、Pass Aはcontroller/parity comparisonとして併記する。

- pooled RMSE delta vs exp223 `<= -0.05 ft`。
- exp223を改善するreporting fold `>=4/5`。
- true TVT inside known range delta `<=+0.02 ft`。
- true TVT outside known range delta `<=-0.10 ft`。
- true TVTがupper boundaryから`0-12 / 12-24 / 24+ ft`外へ進んだ各scopeを報告し、0-12 ft delta `<=0.0 ft`。
- distance `1000_plus`、hidden-like spatial、hidden-like typewell-purged deltaは各`<=+0.02 ft`。
- by-well RMSE p95 delta `<=0.0 ft`、worst-well regression `<=+0.25 ft`。
- finite coverage 1.0、step-delta p99はexp223比非悪化。

全technical/performance gateをPASSした場合だけscientific supportとする。1条件でもFAILならhandoff式、12-ft fade、normalizer、alpha、clip、support、thresholdを救済せず閉じる。PASSでもexp209 blend 10.269696以下と別設計・承認なしにinferenceへ進めない。

## 実験範囲

- 対象実験: `exp299_base_posterior_self_gr_boundary_handoff`
- Route: `ensemble`
- 親実験: `exp223_joint_typewell_self_gr_hmm_likelihood_probe`
- 変更する変数: base-posterior row handoff、outside exact-zero、conditional support-mass normalizationを一体の1 handoff policyとして追加する。
- 固定する変数: exp223 raw self-GR surface、Type Well HMM、入力、score rows、reporting folds、saved control。
- 今回の範囲: 別名compact self-contained train候補、fail-closed inference候補、専用tests、config/記録更新。正規Notebook、Kaggle package、実行、実推論、submissionは変更・作成しない。

## 再現性設計

- seed policy: HMM/self-GR/handoffはRNGなし、reporting foldだけstable SHA256 well hash。
- stochastic処理: なし。
- PF/Beam / likelihood-PF / seed bagging: なし。
- parallel: 将来実装時はouter workers 2、Numba threads 2を固定し、Pass A/Bを同じwell worker内で順番に実行する。thread schedulingをgateへ使わない。
- runtime: Kaggle private CPU、GPU/internet無効。推定8-10時間、12時間上限に近づく場合もvariantを分割・削減せずfail-closeする。
- SHA: loose/package/bootstrap config/source、raw input、saved exp209/223/296、Pass A posterior、support、row gate、conditional contribution、Pass B prediction、schema、metricsを記録する。
- gzip: decompressed content SHAを主証拠、raw gzip SHAを補助にする。
- model/submission: trained model、submissionが存在しないことをmanifestへ記録する。
- deterministic anchor: false。train-side no-training diagnosticでありsubmission anchorではない。

## リスク

- controller error: base HMMが誤modeならhandoff timingも誤る。base posteriorを正解扱いせず、fold/scope/worst-wellで止める。
- double use: Type Well evidenceをPass A controllerとPass B base emissionに使うが、Pass Aはtarget-free deterministic controllerであり学習labelではない。循環を避けるためPass B posteriorをgateへ戻さない。
- conditional normalization: signed contributionがexp223 boost-onlyと異なる。support-mass parityとraw-surface parityを分けて記録する。
- boundary fade: 12 ftをposthoc変更すると原因分離不能になるため、exp223 sigma 1倍だけに固定する。
- runtime: 1,546 HMM decodesでCPU時間が重い。controlを再実行せず、scientific variant 1本だけに限定する。
- CV/LB: train positiveでもhidden testへ保証されない。train gatesと別承認なしにraw-test inferenceへ進めない。

## 結果

Kaggle private CPU version 2はPass A/B各773 wells、合計1,546 HMM well-runsを完了し、exp209 base parityはmax/mean abs `0/0 ft`でPASSした。candidate RMSEは`11.789577561`、saved exp223比`+0.439634615 ft`、改善0/5 folds、performance 2/11 PASS。technical唯一のFAILはrow gate maxの`2.9e-15`丸め超過で、performance棄却には影響しない。事前固定fail actionどおり数式救済を行わずbranchを閉じる。
