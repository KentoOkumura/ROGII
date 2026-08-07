# exp498_geometry_mean_reversion_tail_regime_physics_readout 結果

## 状態

Kaggle private CPU version 2完了。technical PASS / physics-regime FAIL、terminal close。

## 仮説

弱いknown-prefix GR拘束、geometry面とGR-corrected exp226面の大きな不一致、materialな
early offsetが同時にあるwellへ、exp490のtail悪化が集中する。

## 実行

- kernel: `kentookumura/exp498-geometry-mean-reversion-tail-regime-train`
- Kaggle id_no / version: `129328553` / `2`
- runtime: private CPU、internet off、`76.685304 sec`、peak RSS `0.360542 GiB`
- scope: 3,783,989 rows / 773 wells / truth-late 5 folds
- scientific readout / well aggregation / fold readout: `1 / 773 / 5`
- 新規HMM / prediction / model / trained fold / booster / PF / Beam / GPU: すべて0
- inference / submission: 0 / 0

version 1はtruth-late joinの列suffix参照不整合で技術FAILした。固定科学契約を変えず
`fold_manifest` / `fold_outcome`へ修正し、14 testsをPASSしてversion 2を実行した。

## Technical gate

全項目PASS。

- 固定input SHA一致、prediction 3,783,989 rows、well identity 773件を確認した。
- horizontal suffix truth readは0、feature freeze前のoutcome readは0。
- fixed bucket assignmentとfeature finite coverageは完全。
- new prediction / HMM / model / PF / Beam / GPUは0。
- feature content SHA: `c1d31113e7247ada9be0d6fd1e183808f7fbc04af256612481363ee900e0f5ad`
- feature contract SHA: `92d1e78a197a9726640ea049891a6081e30784b98f3faa0b8ab113af8eb2416c`

## Physics-regime gate

primary `weak_gr_geometry_conflict`は0 / 773 wellsで、6項目all-ANDを全てFAILした。

| 指標 | 結果 |
| --- | ---: |
| weak observation | 359 wells |
| geometry disagreement `>=10 ft` | 0 wells |
| geometry disagreement最大 | 5.337991 ft |
| early abs offset `>=5 ft` | 1 well |
| primary 3条件同時成立 | 0 wells |
| supported folds | 0 / 5 |
| catastrophic tail | 51 wells |
| catastrophic capture | 0.0% |

regimeが空のためharmful rate ratioとmean delta差は未定義で、coverage/fold support、
pooled harmful rate、fold harmful direction、pooled mean delta、fold mean delta、
bounded-coverage catastrophic captureはすべてFAILした。

## 補足 readout

全773 wellsを占めるcomplementではharmful 211 wells（27.2962%）、catastrophic 51 wells
（6.5977%）、mean / median candidate-minus-parent RMSEは
`-0.769496 / -0.057105 ft`だった。persistent episode 638件のcandidate SSEは
parent比`41.409965%`減であり、exp490のpooled improvement自体は再現した。
これは空のprimary regimeを救済しない。

## 再現性

- output: `kaggle/output/train_v2`
- input manifest SHA: `450e2cfba697cbe700338b3b5e430bed04bf01e72c659ec0ab81121ab4c93ae2`
- feature contract file SHA: `ccdc3d5d5233546a350385f6f5f8b5d1ac351488bb3c62acd50af20df37d55c1`
- target-free feature file SHA: `c1d31113e7247ada9be0d6fd1e183808f7fbc04af256612481363ee900e0f5ad`
- by-fold file SHA: `eaa7cea3b0a6cd0164495ddf85573f7e28c219787b7ded9da78f2fa0394357fb`
- bucket summary file SHA: `c1a0e5ebb69ac95dfd4af3fda6e421caeb99d02b03971aefffb4b1c91039a406`
- summary file SHA: `485964967acf3a6cc913eaeba6ebaa458384d37090bb0f71d1a16e1f298c1df7`
- metrics file SHA: `74a02ecbf176ed68b76f906c6069fd133857ce4aa6a219309d790124d4b3fee9`

## 判断

`terminate_mean_reversion_tail_regime_cause_tracking`。

事前登録どおり、geometry thresholdやearly-offset thresholdを緩和せず、secondary bucket、
interaction、same-OOF gateで救済しない。exp490のterminal fail-closeを維持し、
観測可能な不確実性で復元力を弱める後続式は設計しない。inference / submissionも行わない。
