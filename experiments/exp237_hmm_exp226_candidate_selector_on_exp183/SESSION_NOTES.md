# exp237_hmm_exp226_candidate_selector_on_exp183 セッションノート

## 2026-07-12 実装

### 狙い

`hmm_exp226_candidate_selector_on_exp183` backlogを実装する。exp183の8候補とcluster/prior confidence surfaceを維持し、exp209 exact-HMM blend、exp223 self-GR HMM、exp226 K16 geometryを候補 pathとして追加する。

### 実装内容

- steering: `docs/legacy/steering/20260712-exp237-hmm-exp226-candidate-selector-on-exp183/`
- 親: `exp183_cluster_outlier_prior_confidence_addonly_on_exp158_selector`
- Route: `ensemble`。PF/Beam候補、HMM候補、geometry候補とLightGBM selectorが本質的に寄与するため。
- exp209 / exp223 / exp226 のOOFを`id`、`well`、`row`、target、last-known TVTで照合して結合する。
- exp209から`blend_likpf_hmm_w500`、exp223からpure self-GR HMM、exp226からK16 prediction、`gr_delta`、geometry disagreementを作る。
- candidate oracle readout（候補別RMSE、unique-best rate、残差相関、8候補oracleとの差）をranker学習前に保存する。
- 3 head比較は行わない。過去exp101/157/183で最良だった`lgb_candidate_error_ranker`のみを5 foldsで学習する。
- Viterbiはexp183 best parameter 1本に固定し、追加tuningを行わない。

### Kaggle train push 前のコスト確認

- active selector variant: 1
- LightGBM configs: 1 (`lgb_candidate_error_ranker`)
- folds: 5
- planned boosters: 5
- fixed Viterbi variants: 1
- parent / control retraining: なし
- GPU: disabled
- raw-test inference / submit: なし

### 再現性

- upstream PF/Beam、exact HMM、self-GR HMM、K16 geometryは固定済みgroup-safe OOFを入力にする。再生成しない。
- exp237の確率性はLightGBM seedとcandidate-longのfold別local RNG subsampleだけ。seed=42からfold別に固定する。
- OOF source、feature schema、model manifest、OOF predictionはSHAを記録する。gzipはdecompressed SHAを主証拠にする。
- deterministic submission anchorではない。

### 静的検証

- `.venv/bin/python -m py_compile`: PASS
- `.venv/bin/ruff check --select F821`: PASS
- `make validate-exp EXP=exp237_hmm_exp226_candidate_selector_on_exp183`: PASS
- Jupytext `--to ipynb --test`（train / inference）: PASS
- local OOF sample contract: exp223の100行とexp226の100行をbase cacheへID結合し、coverage=100%、well一致、target / TVT差がCSV丸め許容値`1e-3`以内であることを確認。exp223の保存順はbase cacheと異なるため、実装は順序を使わずID/well joinに限定する。
- train Kaggle package生成: PASS。canonical id/title、CPU/internet off、7 kernel sources、bootstrap内exp237 config、5 boosters設定を確認。
- 上記はpush前の実装時点の確認。後続のKaggle v1実行・output取得は次節に記録する。

## 2026-07-12 Kaggle CPU train v1

### Pushと存在確認

```bash
make push-kaggle-train EXP=exp237_hmm_exp226_candidate_selector_on_exp183
```

- kernel: `kentookumura/exp237-hmm-exp226-candidate-selector-exp183-train`
- URL: https://www.kaggle.com/code/kentookumura/exp237-hmm-exp226-candidate-selector-exp183-train
- version: 1
- push: 成功
- runtime: CPU (`enable_gpu=false`)、internet off
- 取得metadataの `id_no`: `126812357`
- kernel sources: exp065 / exp099 / exp109 / exp114 / exp209 / exp223 / exp226 の7本を確認。
- 初回状態: `KernelWorkerStatus.RUNNING`

### 完了確認

- status: KernelWorkerStatus.COMPLETE
- summary status: completed_train_side_audit
- runtime: 3,051.086 sec
- rows / wells: 3,783,989 / 773
- 5 boosters と fixed Viterbi 1本を完走。追加のgrid、parent/control再学習、inference、submitは行っていない。
- exp209 / exp223 / exp226 source contractはそれぞれ joined rows 3,783,989、wells 773、missing rows 0。

### Candidate readout

- 11候補 oracle RMSE: 2.883509552（base8 oracle 4.564605115から -1.681095563）。
- added candidate single RMSE: exp226 9.427109674、exp209 blend 10.269696511、exp223 self-GR HMM 11.349942883。
- exp223 unique-best 22.2748%、exp226 14.7338%、exp209 7.4737%。candidate coverage自体は支持された。

### Selector結果とguard

- row-wise error ranker: RMSE 8.545227678、MAE 4.998193016、within10 0.850156013。
- fixed Viterbi: RMSE 8.545093286、MAE 4.991818709、within10 0.849474721、switch 7,640 / 2.019033 per 1,000 rows。
- fixed Viterbi のdelta: exp183 10.601481774から -2.056388488、exp226単体 9.427109597から -0.882016311、likPF 11.594897672から -3.049804386。
- exp183同一Viterbiとのdistance bucket比較は 000_050 が +0.167573（0.508182 -> 0.675755）で悪化、他5 bucketは改善。
- hidden-like は spatial valid 12.593127 -> 8.637572、typewell-purged valid 12.479252 -> 8.598143で改善。
- worst RMSE は fb03ae90 の 57.642328。well別最大回帰は 70925e23 の +25.639006（6.588074 -> 32.227080）。fb03ae90ではgeometry / dense選択行のMAEが約60 ftであり、new-candidate mis-selectionの安全性リスクが残る。
- よってglobal改善は支持するがnear / worst-well guardは不通過。rank-slot add-only、raw-test inference、submitへは進まない。

### 取得生成物とSHA

- output: /tmp/kaggle-output/exp237_hmm_exp226_candidate_selector_on_exp183/train_v1/artifacts/
- metrics: f48a5906a6a7698cc6d88fef2db182bb8a4f634f118b8a9720a01fb379eb6486
- OOF gzip: c5d94361c2582f3f2e419ff70e8f87c1e4d3613b4cc21981e11f009f956d66c9; decompressed: 588451ff8cc46da47461f8acf9f2541d3ac6e989fc6e828bd542a5cb213394e9
- feature schema: bbdd443317b5926efe0acf20e075978b8ee774d5504291973b9ad66ea07ef20e
- model manifest: bcdc24f98696c8bd4fe19059d6726debc6f5b4514bac02611b36737e4539ddb6
- source decompressed SHA: exp209 8e2f42367b7b8b28e73094eae642c57c75dc8a7ebcfbc3826b0f2067b37f7ae5; exp223 0eb48b5516276b0ab7b2191a52a39ebb89d9997363cc7839ede519c7863baa0c; exp226 709eb726cc30da523f017ed0dbd0371967b88a91ddcf25578eb9356f28e4c609。

## 次アクション

## 2026-07-12 ユーザー承認済み raw-test inference

- ユーザーはnear / worst-well guard不通過を記録済みのまま、raw-test inferenceへの進行を明示承認した。
- scopeは同一exp237内のfixed Viterbi artifact生成だけ。Kaggle competition submit、rank-slot ML化、booster再学習、Viterbi再tuningは含めない。
- raw-test source: exp073 CPU deterministic base cache、exp226 K16 inference output、raw test再生成のexp209 exact HMM / exp223 self-GR HMM、raw test GRで再計算するexp099 multi-observation scorer、exp237 train v1の5 saved boosters。
- train v1はfold imputation medianを保存していない。saved boosterは再学習せず、model schemaごとにraw-test medianを再計算し、all-missing列は0にする。
- exp109 / exp114 OOF-only cluster/prior confidenceにはraw-test parity sourceがない。該当long featureはraw-test median / all-missing時0に補完し、rawtest summaryに列名と件数を出す。これは既知のfeature parity gapであり、submission判断には使わない。
- inference notebook / helper / configを更新し、py_compile、F821、Jupytext test、strict validate-expを実行する。Kaggle CPU inference push前にkernel source、bootstrap、source contractを確認する。
- planned inference workload: raw test HMM generation 2 paths（exp209 exact / exp223 self-GR）、exp099 scorer 1 pass、saved ranker boosters 5本の予測、raw-test model-schema median計算。LightGBM booster再学習 0、parent/control再学習 0、Viterbi variant 1、competition submit 0。

### Kaggle inference v1 push

- 初回canonical候補 `kentookumura/exp237-hmm-exp226-candidate-selector-exp183-inference` は id/title 一致、5 sourceまで縮減後も `SaveKernel` 400。slug length 53でtrain slug 49より長かった。
- 同じ実験内でsuffixを `infer` に縮め、canonical id/titleを `kentookumura/exp237-hmm-exp226-candidate-selector-exp183-infer` / `exp237 hmm exp226 candidate selector exp183 infer` に統一した。
- `make push-kaggle-infer EXP=exp237_hmm_exp226_candidate_selector_on_exp183`: v1 push成功。
- URL: https://www.kaggle.com/code/kentookumura/exp237-hmm-exp226-candidate-selector-exp183-infer
- metadata確認: CPU、internet off、competition source 1、kernel source 5。追加trainなし、competition submitなし。
- 初回状態: `KernelWorkerStatus.RUNNING`。

### Kaggle inference v1 failure and v2 fix

- v1 failed after HMM candidate regeneration and exp099 multi-observation scoring, before saved booster prediction.
- root cause: `rawtest_inference.py` の `_test_candidate_features()` は `(frame, candidate_values, dummy_labels)` の3値を返すが、呼び出し側が4値としてunpackしていた。Kaggle traceback: `ValueError: not enough values to unpack (expected 4, got 3)`。
- fix: candidate specsを別に作り、3値だけを受け取るように修正。candidate値、Viterbi parameter、source contract、imputation policy、booster数は変更していない。
- py_compile / ruff / Jupytext test / strict validate-exp: PASS。same canonical infer kernelへv2として再pushする。
- v2: `Kernel version 2 successfully pushed`。kernel ID / source 5本 / CPU / internet off / no-submit はv1と同一。ユーザー指定に従い、agent側の継続監視はしない。

## 2026-07-12 Kaggle CPU raw-test inference v2 完了

- kernel: `kentookumura/exp237-hmm-exp226-candidate-selector-exp183-infer` v2（CPU、internet off、competition submitなし）。
- summary status: `rawtest_inference_completed_not_submitted`、runtime 213.659 sec、14,151 rows / 3 wells。
- exp209 exact-HMM、exp223 self-GR HMM、exp099 multi-observation scorerをraw testで生成し、exp226 K16とtrain v1の保存済み5 boostersを使用した。source contractとmodel manifestを完了ログで確認した。
- `submission.csv` は `id,tvt` の14,151行で、ID重複0、欠損0、finite TVT、selected predictionの`selected_tvt`と完全一致した。submission SHAは `8e188be763a761965b9cfa1f3b26991b8093b447e5198ee01864209fb4d1c2a0`。
- selected prediction gzip SHA: `a5e1fa8becbc682d7376c21fef29d9e0ab4c0ee62b27cecda81808e973aff55f`。candidate feature gzip SHA: `052e7db70bd715a2543b9f185a2e6d44bd815b11e75a090e1310e238862f108a`。decompressed prediction content SHA: `01034afdbc7671dfcd93128f704e87b77cc78c9538a28f7030c8e7987a210207`。
- raw-test feature coverageはrow feature 149本、unavailable long feature 320本。exp109/114 OOF-only cluster/prior featureをraw-test schema median（all-missingは0）で補完した。さらにselectorは全14,151行で`pf_ancc`を選び、predicted errorも一定だった。
- したがって生成物の整合性は確認できたが、train-side near / worst-well guard不通過とfeature parity gapの両方により提出候補にはしない。rank-slot ML化も保留を維持する。
