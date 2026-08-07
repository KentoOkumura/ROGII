# 設計

## アプローチ

exp280の13-shift log likelihoodをblock内で正規化し、予測値を変えない7 familyのalias confidenceへ圧縮する。block順依存familyはwell内の直前blockまたは直前3 blocksだけを使う。全feature、Q1/Q4、content SHAをfreezeした後だけexp264/exp226 truth-side block metricsをjoinする。

実装上は、marginだけ符号反転して「高いほどalias risk」の向きへ揃える。softmaxは
temperature 1、entropyは自然対数、weighted shift stdはpopulation stdとする。block順依存
2 familyの先頭値は0、3-block符号不整合は直近3 blockの非zero符号間pairwise disagreement
shareとする。circular controlはwellごとのtop1 shift列をSHA256由来の非zero offsetで回転後、
同じ2 familyを再計算する。

## 実験範囲

- 対象実験: `exp340_exp226_depth_alias_block_confidence_readout_on_exp264`
- Route: `ensemble`
- 親実験: `exp264_exp263_candidate_confidence_dual_selector`
- source: `exp280_exp226_shift_likelihood_separability_readout`
- 変更する変数: なし。target-free confidenceのattribution readoutだけ。
- 固定する変数: candidate bank、prediction、fold、H512 block、13 shifts、Gaussian score、family、quantile、gate。
- 実行量: 7 fixed readout families + 1 circular control、model/HMM/booster各0。

## 検証方法

1. exp280 score、exp264/exp226 OOF、fold/hidden-like SHAをpreflightする。
2. scoreから7 familyとfold別Q1/Q4をtruthなしでfreezeする。
3. feature content SHA確定後だけblock truth/errorをjoinする。
4. coverage、Q4-Q1、AUC、fold、1000+、hidden-like、control差を判定する。

`abs_error>=10 ft` AUCは、block riskをblock内の各rowへ繰り返した場合と厳密に等価な
tie-aware row-weighted AUCとして集計する。1000+はblock最小`md_since`が1000 ft以上、
hidden-likeはexp115のwell roleで固定する。

fold別guardはexp280が保存したexp226 GroupKFold foldを使う。exp264 Stage Dの
`outer_fold`は別のOOF生成契約なのでprovenanceとして監査するが、exp226 foldとの一致は
要求しない。truthはexp264保存`actual_tvt`を評価の正とし、exp226保存truthとはfloat32
serialization差を考慮した絶対誤差`1e-3 ft`以内のparityを要求する。

## 再現性設計

- RNGなし。block identity、shift order、well/block orderを固定する。
- circular controlはSHA256(well_id)から決めるdeterministic rotationを使う。
- CPU、GPU/internet off、modelなし。
- input/feature/quantile/readoutのdecompressed content SHAを記録する。
- prediction/model/submission SHAは新規生成しない。

## リスク

- same-OOF過適合: familyとgateを事前固定し、PASS後も介入ruleを本expから作らない。
- raw-test parity: exp226 pathとexp280 scoreがhidden testで再生成可能かは後続add-only exp前に別監査する。
- aliasの定義ずれ: 高誤差一般とalias-like failureを分けて両方読む。

## 優先度

Late phaseの低コスト`P1--P2`として先行し、Kaggle Stage 0を完了した。全familyが
scientific gateをFAILしたため、2026-07-23にterminal close。
