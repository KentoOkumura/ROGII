# exp248_candidate_perturbation_augmentation_for_likelihood_ranker セッションノート

## 目的

`candidate_perturbation_augmentation_for_likelihood_ranker` backlogを実験化する。
exp237の固定11候補へ正解非依存のcandidate-set perturbationを加え、learned within10 likelihoodとcandidate absolute-error rankerの教師例を増やす。

## 現在の状態

- Route: `ensemble`
- 状態: Kaggle CPU train version 1完了、augmentation guard failed / 不採用
- CV / LB: augmented fixed Viterbi 8.728086071、original-only control 8.421415097 / なし
- inference / submit: disabled
- 親: `exp237_hmm_exp226_candidate_selector_on_exp183`
- ML参照anchor: exp218 CV 8.475793752
- ensemble Public LB anchor: exp082 7.601

## 固定した設計

- candidate bankはexp237のPF/Beam/dense/HMM/geometry 11候補を固定する。
- exp218は参照anchorに留め、候補追加や最終LightGBM再学習を初回probeへ混ぜない。
- active variantsは`original_only`と`perturbation_augmented`。
- 両variantでwithin10 classifierとL1 expected-error regressorを同一5 outer foldsで学習する。
- augmented variantは、clean viewを必ず残し、sampled outer-train rowごとに最大1つのsingle-transform viewだけを追加する。
- transformはfixed shift、common datum shift、half-cosine low-frequency drift、candidate dropout、family dropout、target-free top dropout、candidate spread scaleの7種類。transformを合成しない。
- outer-validは常にclean original 11候補だけ。augmented candidateは教師例専用で、direct prediction / Viterbi state / blend / replacementに使わない。
- candidate値変更後はminus-last、candidate pair/mean/std/range、multi-observation score/MAE/NCC、availability contextを再計算する。
- exp237 fixed Viterbi ruleはparameter gridなしで各variantのclean OOF expected-error surfaceへ適用する。

## 実行前コストガード

- active variants: 2 (`original_only`, `perturbation_augmented`)
- LightGBM objectives/configs: 2 per variant、合計4
- folds: 5
- total boosters: 2 variants x 2 configs x 5 folds = 20
- original-only control retraining: あり。augmentationだけの因果比較に必要なsame-fold controlであり、Kaggle CPU実行とする。
- parent candidate/control retraining: なし
- exp218 final model retraining: なし
- PF/HMM/Beam/dense/geometry regeneration: なし
- GPU: なし
- inference / submit: なし

Kaggle train pushはこの20 CPU boostersの実行契約を再確認してから行う。

## リークガード

- augmentation割当、方向、amplitude、dropout対象、sample採否はtarget、true error、oracle rank、hidden-like roleを参照しない。
- outer split後、train_idxだけへaugmentationを生成する。
- validation likelihood/error featureはclean original候補から再生成する。
- model feature schemaから`target`、`true_tvt`、`abs_error`、`oracle`、`is_oracle`をassertionで除外する。
- target-free top dropoutはmulti-observation scoreだけを使い、oracle topを使わない。
- augmented viewが偶然正解方向へ改善したかどうかでsampleを採否しない。
- exp115 roleはsubgroup metricだけに使う。

## 再現性メモ

- `docs/06_reproducibility.md`を2026-07-14に確認した。
- seed policy: experiment/fold/variantをSHA256へ入れたlocal RNG。Python `hash()`とglobal RNGを使わない。
- sampled input rowはsortし、single-thread augmentation生成順を固定する。
- PF/HMM等の新規乱数生成なし。保存済みfold-safe OOF candidateを固定入力にする。
- Kaggle CPU、GPU false、internet false。deterministic submission anchorとは扱わない。
- input file/decompressed SHA、augmentation inventory decompressed SHA、feature schema、20 model SHA、OOF prediction decompressed SHAをsummaryへ保存する。
- inference/submission SHAは対象外。

## コマンドログ

### 2026-07-14 steering / scaffold

```bash
make new-steering EXP=exp248_candidate_perturbation_augmentation_for_likelihood_ranker
make new-exp EXP=exp248_candidate_perturbation_augmentation_for_likelihood_ranker SOURCE=experiments/exp237_hmm_exp226_candidate_selector_on_exp183
```

- `.steering/20260714-exp248-candidate-perturbation-augmentation-for-likelihood-ranker/`へ要件、設計、tasklistを記録した。
- exp237の固定candidate/context/Viterbi実装を参照元として継承した。

### 2026-07-14 実装

- `config.yaml`をexp248用に更新した。
- deterministic augmentation inventory、raw horizontal GRを使うcandidate observation再計算、candidate-long feature/label構築を実装した。
- original-only / augmentedのbinary likelihood・expected-error GroupKFold学習、clean OOF score、fixed Viterbi、candidate calibration、topK、margin、bucket、hidden-like、by-well、worst-well guardを実装した。
- heavy LightGBM fold学習とcandidate feature生成は補助moduleへ置き、notebook上にcost/input/augmentation/fold/metrics/SHA orchestrationを展開した。
- inference notebookはtrain-side-only guardで停止する。

### 2026-07-14 静的・合成契約検証

```bash
.venv/bin/python -m py_compile experiments/exp248_candidate_perturbation_augmentation_for_likelihood_ranker/*.py
.venv/bin/ruff check experiments/exp248_candidate_perturbation_augmentation_for_likelihood_ranker/candidate_perturbation_augmentation_for_likelihood_ranker.py experiments/exp248_candidate_perturbation_augmentation_for_likelihood_ranker/exp248_candidate_perturbation_augmentation_for_likelihood_ranker_train.py experiments/exp248_candidate_perturbation_augmentation_for_likelihood_ranker/exp248_candidate_perturbation_augmentation_for_likelihood_ranker_inference.py
PYTHONPATH=experiments/exp248_candidate_perturbation_augmentation_for_likelihood_ranker .venv/bin/python -c "from candidate_perturbation_augmentation_for_likelihood_ranker import synthetic_augmentation_contract_test; print(synthetic_augmentation_contract_test())"
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp248_candidate_perturbation_augmentation_for_likelihood_ranker/exp248_candidate_perturbation_augmentation_for_likelihood_ranker_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp248_candidate_perturbation_augmentation_for_likelihood_ranker/exp248_candidate_perturbation_augmentation_for_likelihood_ranker_inference.py
```

- py_compile: pass。
- Ruff: pass。
- Jupytext convert/test: train / inferenceともpass。
- `make validate-exp ... STRICT=1`: pass。
- `make validate-template`: pass。
- synthetic contract: 同seedでcandidate values/availabilityが一致し、全7transformをcoverage、最低available候補数6。
- 実データ20 base rowsの関数単位smoke test: 220 candidate rows、97 model features、finite label、protected feature混入なし。
- notebook sourceに`__file__`なし。
- 親exp237にcompact self-contained版はない。親train 205行/5章に対し、exp248 trainは256行/7章で、cost、input、augmentation、fold学習、metrics、feature importance、SHAを上位セルへ展開した。重いengineは補助moduleに残す。
- 初回notebook実行はKaggleを正とし、ローカルnotebook/full trainは実行していない。

### 2026-07-14 Kaggle package検証

```bash
make prepare-kaggle-notebooks EXP=exp248_candidate_perturbation_augmentation_for_likelihood_ranker EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp248-candidate-perturbation-augmentation-for-likelihood-ranker-train --title 'exp248 candidate perturbation augmentation for likelihood ranker train' --run-on-push --strict"
```

- canonical kernel ID / title、`enable_gpu=false`、`enable_internet=false`、`run_on_push=true`を確認した。
- package内configのactive variants 2、LightGBM configs 4、folds 5、total boosters 20を確認した。
- `parent_retraining=false`、`control_retraining=true`を確認した。
- package内Python sourceのpy_compile / Ruffをpassした。
- 最終sourceとpackage内augmentation engineのSHA256一致を確認した。
- 最終sourceでの実データsmoke再実行は、ローカルにexp072大容量補助cacheがないためinput resolveで停止した。Kaggle packageは同artifactを含むupstream kernel sourceを持つ。コード起因の失敗ではなく、ローカルfull-input実行は正としない。
- この時点ではpackage作成までで停止し、Kaggle pushは行っていなかった。

## 次のアクション

1. augmentation mix / amplitude / transform gridの事後探索を行わず、このaugmentation branchを閉じる。
2. original-only dual-objective controlを再利用する場合は、raw-test-safe featureだけに限定した別監査を先に行う。
3. inference / competition submitは行わない。

## 2026-07-14 Kaggle train push v1

- ユーザーが20 CPU boostersの実行を明示承認した。
- push対象: 2 variants、2 objectives / variant、5 folds、合計20 boosters。`control_retraining=true`、`parent_retraining=false`、GPU / internet / inference / submitなし。
- 初回ID/title: `kentookumura/exp248-candidate-perturbation-augmentation-for-likelihood-ranker-train` / `exp248 candidate perturbation augmentation for likelihood ranker train`。
- 結果: Kaggle `SaveKernel`が詳細なしのHTTP 400を返し、学習は開始しなかった。
- 調査: IDとtitle由来slugは一致していたが70文字だった。失敗後のpullは403、`kaggle kernels list --mine --search exp248`は`Not found`で、Kaggle側に利用可能なkernelは作成されていない。
- 復旧: 同じexp248、同じ科学設定のまま、意味を保った42文字のID/title `kentookumura/exp248-candidate-perturbation-ranker-train` / `exp248 candidate perturbation ranker train`へ短縮して再package / pushする。
- 再package検証: slug/title一致、slug長42、CPU、internet off、20 boosters、parent retrainingなし、source/package augmentation engine SHA一致、py_compile / Ruff pass。
- 再push: `Kernel version 1 successfully pushed`。URL: `https://www.kaggle.com/code/kentookumura/exp248-candidate-perturbation-ranker-train`。
- 存在確認: 同じIDのpullに成功し、Kaggle `id_no=127067118`、CPU (`machine_shape=None`)、7 upstream kernel sourcesを確認した。
- 初期状態: `KernelWorkerStatus.RUNNING`。実行中logsが空でも再pushやslug変更を行わず、同じversion 1を監視する。

## 2026-07-15 Kaggle train version 1完了

```bash
kaggle kernels status kentookumura/exp248-candidate-perturbation-ranker-train
kaggle kernels logs kentookumura/exp248-candidate-perturbation-ranker-train
kaggle kernels files kentookumura/exp248-candidate-perturbation-ranker-train --page-size 200 --format json
kaggle kernels output kentookumura/exp248-candidate-perturbation-ranker-train -p /tmp/kaggle-output/exp248-candidate-perturbation-ranker-train/train_v1 --file-pattern 'exp248_.*(summary\.json|metrics\.csv|candidate_metrics\.csv|bucket_metrics\.csv|subgroup_metrics\.csv|by_well\.csv|model_manifest\.json|topk_coverage\.csv|calibration\.csv|margin_calibration\.csv)$' --page-size 200
```

- Kaggle status: `KernelWorkerStatus.COMPLETE`。
- summary status: `completed_train_side_guard_failed`。
- runtime: 9,687.118秒（約2時間41分27秒）。
- rows / wells / candidates: 3,783,989 / 773 / 11。
- features: base 148 / candidate-long 297。
- models: 20、model SHA 20本すべてunique。augmentation inventory 300,000 rows。
- full output archiveは取得せず、logsに表示されなかったmetrics / guard / SHAの記録に必要な小さいCSV/JSONだけを`/tmp/kaggle-output/exp248-candidate-perturbation-ranker-train/train_v1/`へ取得した。

### Clean OOF結果

| mode | original-only RMSE | augmented RMSE | delta |
| --- | ---: | ---: | ---: |
| probability row-wise | 8.500237521 | 8.855196975 | +0.354959454 |
| expected-error row-wise | 8.493972922 | 8.778269697 | +0.284296774 |
| expected-error fixed Viterbi | 8.421415097 | 8.728086071 | +0.306670974 |

- fixed Viterbi fold 0..4 delta: `+0.301214 / +0.336758 / +0.293061 / +0.482689 / +0.103018`。5/5 foldsで悪化。fold値は保存済みby-well row数から、実装と同じnon-shuffled sklearn GroupKFold割当を復元してrow-weight集計した。
- candidate AUC: `0.925006783 -> 0.923061812`。
- candidate logloss: `0.326743985 -> 0.331255410`（`+0.004511425`）。
- candidate Brier: `0.103219669 -> 0.104912398`。
- expected-error MAE: `4.537501009 -> 4.610538118`。
- top1 within10 coverageはprobabilityで`0.855006 -> 0.847102`、predicted-errorで`0.850955 -> 0.847088`へ悪化。

### Guard / by-well

- `selected_rmse_nonworse=false`
- `candidate_logloss_nonworse=false`
- `1000_plus_nonworse=false`: fixed Viterbi RMSE `+0.352222315`
- `hidden_like_nonworse=false`: exp115 spatial `+0.328867071`、typewell-purged `+0.302609121`
- `worst_well_regression_bounded=false`: `389ae58f`で`+15.575245723`
- well: 360改善 / 413悪化。
- `adoption_supported=false`。

### Anchor比較と解釈

- original-only fixed Viterbi `8.421415097`はexp237 fixed Viterbi `8.545093286`から`-0.123678189`、exp218 `8.475793752`から`-0.054378655`。train-sideではpositiveだが、exp237由来のOOF-only contextを含むためraw-test-safe anchorへ昇格しない。
- augmented variantはclean 660,000 long rows/foldへ約619,000 perturbed rowsを足し、synthetic viewがcleanとほぼ同じ学習量になった。最大80 ftのshift/drift、dropout、spreadを一括混合したことがclean calibrationを崩した可能性が高い。ただしtransform個別attributionはないため、原因は推論として扱う。
- 事前契約どおりaugmentation gridを事後探索せず、不採用。inference / submissionなし。

### 再現性SHA

- augmentation inventory decompressed: `80a0e6c7e394f24f2c0a21e6a1447c3bc071b89e3a9852473c620273c208799a`
- OOF prediction decompressed: `aecbecbec5199e6d6559c5cb7c5797b07afedff0c4709ff50e193299e25a82dc`
- feature schema: `46819aaebc98e61a172567b20a1c93bb7816cc6343d8989a429e6dbd69a507ba`
- model manifest: `28e0dbfa9b52f5cad8b0d35f73b7aed97514b61350fdbc93c65093336dccc3ca`
- metrics: `a279a1dcc571b0e68c86c2397fa39f1cc53f50ac26c8129dd7dfdb92c74de64d`
- by-well: `6ee1149067704a7c3b3ff209d1dd139ce380d0e087358545167eab818a86258b`
- downloaded summary: `c82f4db1e6250781f8c8ccef4af45588fd735e6d5ec4d06cb455f415d956ed59`
- kernel log: `1b781cf6328f0ab5cb83133923d6764a24341318a5bd3fcfe3d48591b96956e4`
- 取得したmetrics / candidate metrics / calibration / topK / margin / by-well / bucket / subgroup / model manifestのSHAはsummary記録値とすべて一致した。
