# 設計

## アプローチ

各well known prefixでtypewell residualのpositive ACF massを積分相関時間`tau`へ変換し、outer-train fold medianへlog shrinkする。Stage 0はtruthを使わずprefix window間の安定性だけを確認し、PASS後にexp281 Gaussian log emissionへwell-level `1/tau_eff`を掛ける。

旧exp320のgroup AR(1) whiteningとは異なり、Type Well群間transferやresidual innovation変換を
行わない。well自身のknown-prefix ACFとouter-train fold median fallbackだけを使うため、
exp311/313の失敗系列から独立している。

## 実験範囲

- 対象実験: `exp343_acf_effective_sample_likelihood_tempering_audit`
- Route: `pf_beam`
- 親実験: exp281。
- 履歴参照: exp320。旧実験は閉鎖履歴のまま変更しない。
- 変更する変数: well-level Gaussian log-likelihood weight `1/tau_eff`だけ。
- 固定する変数: residual、sigma、missing、exp226 shape、HMM state/transition/prior/output。
- 実行量: Stage 0は1 diagnostic / HMM 0。Stage 1は1 variant / 773 runs、control再実行0。
- 2026-07-23の実装範囲はStage 0だけ。Stage 1 decoderは未実装のままにする。

## 検証方法

1. raw/fold/exp281 control契約をpreflightする。
2. outer-train fold median、well tau、fallback、clipをtruthなしでfreezeする。
3. Stage 0 stability/coverage gateを判定する。
4. PASSと別承認時のみStage 1 predictionを生成し、late truthでRMSE/tail gateを判定する。

Stage 0のrhoはcontiguous finite run内のlag pairを連結したpairwise Pearsonとする。
full/last-512 stabilityは両windowがraw-evaluableなwellだけで測り、fallback prior共有による
見かけの相関を除外する。window別median tau、upper clip率、fold median比は悪い側をgateに使う。

## 再現性設計

- RNGなし。contiguous run、lag、fold、well順固定。
- CPU、GPU/internet off。Stage 1最大1 variant・8.5時間。
- residual/ACF/tau/contract/prediction/metricsのdecompressed content SHAを記録する。
- model/submission SHA非該当、inference/submission disabled。

## リスク

- likelihood underweight: exp232/305のnegativeを踏まえ`tau<=4`へcapし、Stage 0安定性を必須化する。
- ACF bias: missingをまたがず、短prefixはfold median fallbackにする。
- rescue risk: lag/k/capは結果後に変更しない。

## 優先度

高リスク`P3`。旧exp320の着眼点を独立化できるが、既存tempering negativeのため
Student-t Stage 0やP1/P2診断より後。Kaggle実行は別承認とする。
