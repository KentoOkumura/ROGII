# 要件

## 依頼

exp337の結果を受け、不確実なGR行の幅を広げるのではなく、確実と事前定義できる行だけGR emission幅を狭める後続実験を設計する。バックログ、steering、実験ディレクトリ、設定、記録を作成するが、科学コードとNotebookは実装しない。

## 制約

- Routeは`pf_beam`、科学的親はexact-HMM parity確認済みの`exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`とする。
- 「確実な行」は補間前のraw horizontal `GR`がfiniteの行だけとし、学習型confidence、truth由来confidence、threshold探索を使わない。
- raw finite行だけknown-prefix finite residualのpopulation stdを使い、raw missing行はexp209 zero-fill stdを完全維持する。
- finite pair 20未満またはscale nonfinite時は、well全体をexp209 scaleへno-op fallbackする。
- GR値の補間、Type Well処理、Gaussian形状、state grammar、transition、prior、posterior meanはexp209から変更しない。
- exp307の全行finite-only化、exp308のfinite-MAD親からのmissing downweight、exp341のmissing variance加算とは別実験として扱う。
- 保存済みexp209 HMM/LikPFを比較基準に使い、親controlを再実行しない。
- 予定実行量は1 variant、773 HMM well-runs、LightGBM/model/fold/booster/PF/Beam/control再実行はすべて0とする。
- 実装、Notebook編集、Kaggle package/push/run、inference、submissionは別承認まで行わない。
- 再現性は`docs/06_reproducibility.md`に従い、raw mask、scale schedule、prediction、metricsのcontent SHAを記録する設計にする。

## 受け入れ基準

- 親、単一変更、raw mask境界、scale式、fallback、clip、固定HMM契約が`config.yaml`と`design.md`で一致している。
- unknown-suffix truthはraw mask・scale schedule・prediction content SHAのfreeze後だけjoinする。
- direct RMSE、5 folds、raw observed/missing rows、missing-fraction bucket、1000+、hidden-like 2面、by-well p95/worst、fixed LikPF 50:50を固定AND gateに含める。
- 同一結果上のsigma multiplier、MAD、threshold、confidence、emission、HMM、blend救済を禁止する。
- deterministic submission anchorとは扱わず、gzip生成物はdecompressed content SHAを主証拠にする。
- design-onlyとして未記入値を解消し、strict experiment validationとtemplate validationを通す。
