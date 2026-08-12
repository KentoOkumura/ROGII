# 設計

## アプローチ

exp072の各粒子`(p,r)`に、sampleしないreliability probability
`alpha=(P(normal),P(weak))`を付ける。transitionは
`[[511/512,1/512],[1/128,127/128]]`、初期`[0.8,0.2]`。
normalはexp072 GR sigma、weakは同じ予測GR平均でsigma 4倍。各行で2状態forward updateを行い、
周辺尤度をparticle weightへ使う。resampling時はancestorのalphaもcopyする。

Stage 0の前半はknown prefixの128行historyから64行held-out GRを予測し、base Gaussian比NLL gainを
測る。後半は保存済みexp072 likpf path上で512行block / stride256のweak posteriorをfreezeし、
truth join後にbad-block AUCを読む。両方を通さない限りPFは実装しない。

Stage 0実装時の残余規約は次で固定する。各wellのfinite `TVT_input` の最終連続192行を
使い、先頭128行をhistory、後続64行をheld-outとする。GRはexp072と同じboth-direction
interpolation後Type Well平均fallbackを使う。sigmaもexp072と同じ全known prefixの
Type Well GR残差（missing raw GRは0埋め）から作り、`[10,60]` clipを使う。
history上でq posteriorを更新し、そのposteriorをheld-outへ持ち越す。baseはsigma 1倍の
正規化済みGaussian、marginalizedはsigma 1倍/4倍の正規化済みGaussian mixtureの逐次予測
密度とし、pooled held-out NLL gain率を読む。

suffixでは保存済みexp072 cacheの`last_known_tvt + likpf_mean_d`だけをpathとして読む。
sigmaはexp072と同じ全known prefixのType Well GR残差標準偏差（missing GRは0埋め）、
suffix GRはboth-direction interpolation後Type Well平均fallbackとする。512行blockは
stride 256の全startを使い、短い末尾blockも保持する。blockごとにqを`[0.8,0.2]`へ戻し、
posterior mean weak scoreをfreezeする。negative controlはwell内block scoreのSHA256由来
非ゼロcircular shift、single-block wellはidentityとする。

Stage 1は500 particles × 128 seeds × 773 wellsの1 treatment。保存済みexp072 controlを使う。
robust likelihood既存負結果を踏まえ、失敗時のsigma/transition/temperature/outlier mixture救済は禁止。

## 実験範囲

- 対象: `exp368_marginalized_reliability_pf`
- Route: `pf_beam`
- 親: `exp072_exp063_full_replay_feature_cache`
- 変更: sticky reliability recursionとmarginal likelihoodだけ。
- 固定: particle state/dynamics、500 particles、128 seeds、ESS/resampling、mean aggregation。
- Stage 0 gate: known-prefix NLL gain`>=1%`、bad-block AUC`>=0.60`、circular差`>=0.02`、
  4/5 folds、hidden-like AUC`>=0.55`、weak mass`[0.02,0.50]`。
- Stage 1 gate: exp072比`>=0.05 ft`、4/5 folds、1000+/hidden-like/p95回帰`<=0.02 ft`、
  worst`<=0.25 ft`。

## 再現性設計

- seed: `SHA256(experiment|well|family|seed_index)`からlocal RNG。
- global RNGとthread schedule依存は禁止。
- stochastic成分はexp072と同じ。q recursionは決定論的。
- CPU single worker、GPU off、上限30,600秒。
- raw train/testを別生成し、q diagnosticsとpredictionのcontent SHAを保存する。
- gzipはdecompressed SHAを使い、suffix truthはfreeze後にjoinする。
- 保存済みexp072 cacheは`id / well / last_known_tvt / likpf_mean_d`だけをparseし、
  cache内`target`は読まない。

## リスク

- leakage: bad-block errorをq updateへ混ぜる危険。freeze境界で防ぐ。
- CV/LB不一致: anomaly頻度差。
- runtime: per-particle 2-state recursionの追加。
- reproducibility: alpha copy順とresampling ancestor順を固定する。
- science: weak likelihoodがwrong modeを延命し、exp232/233同様に悪化し得る。
