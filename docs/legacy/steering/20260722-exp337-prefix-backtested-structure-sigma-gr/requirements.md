# 要件

## 依頼

- `prefix_backtested_structure_sigma_gr`を`exp337`として採番し、バックログ、steering、実験ディレクトリで設計を確定する。
- finite-only GR残差scaleへ、typewell不一致・alignment誤差を表す構造誤差を分散加算する。
- `tau_structure`はunknown suffix truthを使わず、known prefixの時系列分割とrolling-origin predictive likelihoodだけで校正する。
- 2026-07-22の追加依頼により、今回はStage 0のcompact self-contained Notebook候補、専用test、fail-closed inference候補までをimplementation-onlyで実装する。Kaggle package/push/run、Stage 1 HMM、inference、submissionは行わない。

## 仮説

exp307のfinite-only std/MADは観測残差の局所的な散らばりだけを測り、typewellの形状不一致やalignment mode不確実性を除きすぎた。既知prefix内の将来blockに対する予測誤差から構造分散を分離し、

```text
sigma_eff^2 = sigma_finite^2 + tau_structure^2
```

とすれば、欠損GRの0補完に依存せずにGR emissionの過信を抑えられる。

## 制約

- Route: `pf_beam`。
- 科学的親はGaussian GR emissionを持つ`exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`とし、`exp307_finite_only_robust_sigma_gr`は失敗根拠・比較対象として参照する。
- residualはfiniteなknown-prefix pairだけの`e_i = GR_horizontal,i - GR_typewell(TVT_input_i)`、affineは`a=1,b=0`に固定する。
- scale推定はknown-prefix row順、内部split `60% / 40%`、rolling origin `60% / 80%`、各forward evaluation block `20%`に固定する。
- 利用可能prefix内のearly blockで`σ_finite`のpopulation std、late blockで`tau_structure^2=max(0,mean(e_late^2)-sigma_finite^2)`を求める。suffix用最終scaleもfull known prefixを60/40分割し、同じ2項だけを分散加算する。late blockを含む全pair stdはfinite-only比較値にだけ使い、構造分散との二重計上を避ける。
- 利用可能prefixのfinite pairが50未満、early/lateが各20未満、またはnonfiniteの場合は、その時点のexp209 zero-fill scaleへfallbackしてno-opとする。
- 最終scaleは`[10,60]`へclipする。split、origin、pair閾値、fallback、clipを結果後に変更しない。
- Stage 0はHMMを実行せずknown-prefix forward Gaussian NLLだけを監査し、全gate PASS時だけStage 1の1 HMM variantを別承認対象にする。
- evaluation GR補間、typewell前処理、Gaussian center、state grid、transition、prior、posterior mean、LikPF blend weightはexp209互換のまま固定する。
- MAD、affine補正、mean centering、temperature、Student-t/mixture、row-wise sigma、missing-GR downweight、transition noise、blend gridを混ぜない。
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。

## 受け入れ基準

- Stage 0は両rolling originで評価可能well率90%以上、fallback率10%以下を満たす。
- 構造scaleのforward Gaussian NLLがfinite-only scaleより両originのpooledと各4/5 foldsで改善する。
- exp209 zero-fill scaleに対しても両originのpooled NLLを1 finite residualあたり0.005以上改善し、各4/5 foldsで改善する。
- final known-prefix auditで`tau_structure`中央値5.0 GR units以上、`sigma_eff`の下限clip率10%以下を満たす。満たさなければHMMを実装・実行せず閉じる。
- Stage 1を後日実施する場合は、保存済みexp209 raw HMM比RMSE 0.05 ft以上、4/5 folds改善、1000+・hidden-like 2面・by-well p95非悪化、worst regression `<=+0.25 ft`をすべて要求する。
- saved LikPFとの固定50:50 blendをexp209基準から悪化させない。
- Stage 0またはStage 1の1 gateでもFAILした場合は、split/threshold/scale/likelihood/HMM/blend救済、inference、submissionへ進まない。
- RNGなし、raw/input/scientific contract、scale audit、rolling-origin audit、predictionのdecompressed content SHA方針が文書とconfigで一致する。
- 本実験はdeterministic submission anchorではない。model/submission SHAは非該当で、Kaggle kernel versionとprediction/content SHAは実行時のみ記録する。

## 次のアクション

別途Kaggle実行承認が得られた場合だけStage 0のknown-prefix rolling-origin auditを実行する。Stage 1はStage 0全gate PASS後も自動で実装・実行しない。
