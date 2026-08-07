# タスクリスト

## 現在の停止点

Kaggle CPU train version 2を完了。固定判定`INCONCLUSIVE_COVERAGE`かつ主要科学guard不支持のため、救済なしでbranchを閉じた。inference、submissionは行わない。

## 完了

- [x] exp263、exp280、exp281、exp133、exp177の証拠を確認する。
- [x] Routeを`pf_beam`、親をexp263、shrink先をexp226、likelihood parity親をexp280に固定する。
- [x] GR弱区間をouter-train低margin AND 高entropyで定義する。
- [x] exp226 admissibility、GR観測率、near vetoを含む発火条件を固定する。
- [x] `alpha=0.25`、`clip=10 ft`のbounded soft shrink式を固定する。
- [x] stable well内circular-shift negative controlを固定する。
- [x] truth late-join、SHA、technical/scientific guard、停止条件を固定する。
- [x] backlog、steering、exp322 design-only記録、experiment summaryを整合させる。

## 実装完了

- [x] compact self-contained Jupytext train sourceを別名で作る。
- [x] exp263 cache / exp226 OOF / raw/typewell / hidden-like resolverとSHA hard guardを実装する。
- [x] exp280 parity emissionと固定13-shift block scorerを実装する。
- [x] outer-train quantile、real/control gate、bounded shrinkを実装する。
- [x] target-free freezeとlate-truth readout APIを分離する。
- [x] unit testで式、tie、fold境界、near veto、coverage、circular shift、禁止列を確認する。
- [x] Jupytext変換、構文、ruff、strict `validate-exp`を確認する。

## Kaggle実行前の別確認

- [x] 実行量を`1 candidate / 1 control / 5 strata / 0 config / 0 trained fold / 0 booster / 0 parent rerun`として再確認する。
- [x] private CPU、GPU/TPU/internet off、kernel source、bootstrap内configを確認する。
- [x] ユーザーからKaggle CPU package/push/runの明示承認を得る。

## 実行後

- [x] input、schema、content、prediction SHAを記録する。
- [x] technical coverageとexp263 parityを確認する。
- [x] overall/fold/activated subset/scope/by-well/controlの全固定guardを判定する。
- [x] PASS条件を満たさなかったためinference設計更新へ進まない。
- [x] `INCONCLUSIVE_COVERAGE`として救済gridなしでbranchを閉じ、backlogを更新する。

## ブロック中

- なし。exp322は完了・不採用。
