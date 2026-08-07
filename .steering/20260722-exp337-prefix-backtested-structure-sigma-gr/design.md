# 設計

## アプローチ

exp307ではzero-fill std中央値`38.6418`に対しfinite stdが`13.8957`、finite MADが`10.1367`へ縮み、raw HMMをそれぞれ`+2.271430 / +3.723054 ft`悪化させた。finite residualの散らばりを観測noise、時系列後半で増えるzero-center予測誤差をtypewell不一致・alignment誤差とみなし、両者を分散で加える。

known prefixのrow順を固定し、任意の利用可能prefix `P`に対して次を計算する。

```text
e_i = GR_horizontal,i - GR_typewell(TVT_input_i)
sigma_finite = std(e in first 60% finite pairs, ddof=0)
tau_structure^2 = max(0, mean(e^2 in last 40% finite pairs) - sigma_finite^2)
sigma_eff = clip(sqrt(sigma_finite^2 + tau_structure^2), 10, 60)
```

`mean(e^2)`はGaussian centerを`a=1,b=0`のtypewell GRへ固定したときのpredictive likelihood MLEに対応し、biasとmisregistrationをcenter補正せず不確実性へ入れる。late blockを含む全pair stdはStage 0のfinite-only comparatorにだけ使い、`tau_structure`との二重計上を避ける。50 finite pairs未満、early/late各20未満、nonfinite時は同じ利用可能prefixから計算したexp209 zero-fill scaleを返し、unsafeなfinite-only縮小を行わない。

Stage 0では全known-prefix rowの60%と80%をrolling originにし、その時点までの情報だけで3 scale（finite-only、exp209 zero-fill、structure-added）をfreezeする。直後20%のfinite residualに対する`log(sigma)+e^2/(2*sigma^2)`を評価し、構造scaleがfinite-onlyだけでなく旧zero-fillにも勝つかを確認する。Stage 0全gate PASSと別途承認後だけ、full known prefixで同じ式を1回計算し、exp209 exact-HMMへ`gr_sigma`だけを渡すStage 1を実装する。

## 実験範囲

- 対象実験: `exp337_prefix_backtested_structure_sigma_gr`
- Route: `pf_beam`
- 親実験: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- 失敗根拠: `exp307_finite_only_robust_sigma_gr`
- 変更する変数: well別`gr_sigma`をfinite stdとprefix-backtested structure varianceのquadratureへ置換する。
- 固定する変数: GR/typewell preprocessing、`a=1,b=0`、evaluation GR補間、Gaussian emission、`step=0.35`、41 rate states、`rate_span=0.10`、`sig_r=0.002`、`sig_p=0.02`、momentum、start/rate prior、posterior mean。
- Stage 0: 1 diagnostic、HMM/model/booster各0。
- Stage 1（条件付き）: 1 scientific variant、773 HMM well-runs、model/LightGBM/fold/PF/Beam/booster各0、親control再実行0。

## 検証方法

1. raw well identity、exp209 source/config/control prediction、exp307 scale/result、saved LikPF、fold、hidden-like assignmentの契約とSHAをpreflightする。
2. suffix truth/errorを読まず、known prefixだけから両originのscaleとforward evaluation residualをfreezeする。
3. rolling-origin/fold/coverage/fallback/clip/NLLのStage 0 summaryとcontent SHAを保存する。
4. Stage 0全gate PASS時も自動実装しない。別承認後だけfull-prefix scaleをfreezeして1 HMM variantを実行する。
5. Stage 1 prediction freeze後だけunknown suffix truth、5 folds、distance、hidden-like、saved LikPFをjoinする。
6. overall、fold、1000+、hidden-like 2面、by-well p95/worst、fixed 50:50 blendをAND gateで判定する。

## 生成物契約

- scientific/input/dependency contract JSON
- well×origin scale/NLL audit CSV.gz
- final full-prefix `sigma_finite` / `tau_structure` / `sigma_eff` / fallback / clip audit CSV.gz
- Stage 0 gate summary JSON
- Stage 1許可時のみprediction/posterior diagnostics CSV.gz、overall/fold/scope/by-well metrics、promotion gate JSON

## 再現性設計

- seed policy: RNGなし。well ID、raw row、origin `0.60,0.80`、scale policy順を固定する。
- stochastic処理: なし。
- PF/Beam / likelihood-PF / seed bagging: 新規生成なし。saved LikPFはStage 1 readout専用。
- 並列処理: 実装時はexp209採用値`outer_workers=2`、Numba threads `2`を開始点として固定し、変更時はprediction parityを要求する。
- runtime: Kaggle CPU、GPU/internet off。Stage 0は短時間audit、Stage 1は最大1 variant・8.5時間上限。
- SHA: raw/input/dependency/scientific contract、rolling-origin audit、full-prefix scale、prediction、metricsはCSV.gzのdecompressed content SHAを主証拠にする。
- model/submission SHA: 非該当。deterministic submission anchorとは扱わない。
- package: canonical kernel id/title、metadata、bootstrap config/source SHA、実行量をpush前に照合する。Stage 0実装は完了したが、package/push/runは未承認。

## リスク

- リークリスク: rolling originより後のknown-prefix blockをscale fitへ入れない。unknown suffix truth/error/oracleはscale/prediction freeze後だけjoinする。
- objective mismatch: prefix GR NLL改善がTVT RMSE改善を保証しないため、Stage 0は必要条件に限定し、Stage 1で厳しいRMSE/tail gateを維持する。
- 同一枝救済リスク: exp305/307の結果後にsplitやthresholdを調整しない。本案はユーザー指定の独立した構造分散仮説として1式だけを評価する。
- fallbackリスク: prefix不足wellはzero-fill controlへno-op fallbackし、finite-only小scaleを使わない。fallback率が10%を超えたらStage 0 FAILとする。
- runtime: exp307の2 variantsが約7.6時間だったため、Stage 1は1 variantだけに制限し、controlを再実行しない。
- CV/LB: train-side PASS後もinference/submissionを自動許可せず、raw-test portとsubmit-checkは別承認にする。

## 優先度

Lateフェーズでexp305/307が強いnegative evidenceを持つため`低-中・P2`とする。まず0-HMM Stage 0で安価に反証し、現行P1のML downstream候補より先に高コストHMMを走らせない。
