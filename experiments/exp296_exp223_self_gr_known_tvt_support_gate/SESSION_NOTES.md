# exp296_exp223_self_gr_known_tvt_support_gate セッションノート

## 目的

exp223 descriptor-motif self-GRを完全固定し、candidate stateがvisible-prefix known TVT range外の場合だけself-GR boostを0にする単一差分を、実装前に反証可能な契約へ固定する。

## 現在の状態

- Route: `ensemble`
- 状態: Kaggle CPU version 3完了・performance guard FAIL・branch closed
- CV / LB: `12.159749140` / 未提出
- compact implementation / canonical Notebook / Kaggle CPU run: あり / あり / 完了
- 実行count: `1 variant / 773 HMM well-runs / 0 model / 0 booster`
- runtime: `16,667.265 sec = 4.630 hours`

## 2026-07-19 設計確定

### 作成コマンド

```bash
make new-steering EXP=exp296_exp223_self_gr_known_tvt_support_gate
make new-exp EXP=exp296_exp223_self_gr_known_tvt_support_gate
```

### 根拠

- exp223はdescriptor motif boostでexp072 `likpf_mean` 11.594897668を11.349950650へ改善した。
- 一方、exp223はcandidate stateのknown TVT supportをhard gateせず、worst-well regressionが`+46.954683 ft`だった。
- exp225はknown range外をneutralにしたが、descriptor motifをstate-known `TVT_input -> GR`曲線へ同時変更し、RMSE 14.212954500へ悪化した。
- exp225はsupport mask単独の評価ではないため、exp223 motifを固定したisolated ablationが未検証である。

### 固定した設計

- exp223 `hmm_selfgr_boost_only_a070_c100`のHMM、Type Well emission、descriptor、anchor、top-k、surface、quality、alpha/clip/modeを固定する。
- finite visible-prefix `TVT_input`からwell-level `[known_tvt_min, known_tvt_max]`を作る。
- exp223 full-grid centering/scaling/positive clip後にinclusive support maskを掛ける。
- support内boostはexact exp223 parity、support外contributionはexact 0。
- base HMM/posteriorをknown rangeへ拘束せず、final predictionをgateに使わない。
- padding、nearest-distance、hole-aware、soft weight、inside recentering、alpha/clip/window/top-k gridは禁止する。
- controlはsaved exp223 decompressed SHA `0eb48b5516276b0ab7b2191a52a39ebb89d9997363cc7839ede519c7863baa0c`を再利用し、再実行しない。
- performance gateはpooled `-0.05 ft`、4/5 folds、outside scope `-0.10 ft`、inside/1000+/hidden-like/p95非悪化、worst `+0.25 ft`以内を全必須とする。
- FAIL時は救済gridなしで閉じる。PASSでもexp209 blend 10.269696以下と別承認なしにinferenceへ進めない。

### 初回設計ターンで変更していないもの

- scaffold train/inference Notebookと`settings.py`。
- 実験ロジック、Jupytext source、helper、tests（後続の実装承認前時点）。
- Kaggle package、kernel metadata、bootstrap、train、output。
- inference、submission、submission log。

### 設計検証

- `metrics.json`を`python -m json.tool`でparseし、`config.yaml`を`yaml.safe_load`したうえでstatus、planned variant 1、planned HMM well-runs 773、implementation falseをassertした。
- 初回のstrict experiment validationはREADMEの必須`## 所見`節不足だけでFAILした。数値結果なし・exp225はmask単独比較でないという所見を追記し、`make validate-exp EXP=exp296_exp223_self_gr_known_tvt_support_gate`を再実行してPASSした。
- `make validate-template`は`project.yml validation passed (template)`でPASSした。
- `make update-summary`で`experiment_summary.md`を更新し、292 experiments、exp223からexp296へのlineage、status `design_locked_not_implemented`を確認した。
- `kaggle-review-exp`同梱reviewerを再実行し、steering、backlog、README、SESSION_NOTES、config、metrics、result、support contractのcore evidence categoryがすべて揃っていることを確認した。
- scaffold train/inference Notebookと`settings.py`は生成時の実験名置換以外templateと一致した。実験ロジックは未実装である。
- design/experiment docsを未記入値監査し、template utility内の未記入値検出用sentinel定義を除いて未記入項目がないことを確認した。

## 2026-07-19 compact実装

### 承認範囲

ユーザーから実装指示を受け、別名compact self-contained train source/Notebook候補とcontract testsまでを実装した。正規train Notebook採用、Kaggle package/push、HMM実行、inference、submissionは承認範囲外のため行っていない。

### 実装内容

- `exp296_exp223_self_gr_known_tvt_support_gate_compact_selfcontained_train.py`へ、exp223のprefix stats、GR descriptor、anchor selection、full-grid surface、quality、exact forward-backward HMMを統合した。
- exp223の`center -> scale -> positive clip`後にだけinclusive candidate-state maskを適用し、support外列をdtype-preserving assignmentでexact `0.0`にした。support内はcopy後にbitwise parityをassertする。
- no-finite-known-TVTではmaskをall-falseにし、self-GRだけneutralにする。通常データ外のfail-safeとしてbase Type Well HMMはType Well TVT中央値start、sigma 30、rate 0でfiniteに保つ。
- final predictionはknown rangeへclipせず、posterior outside-support massをrow diagnosticとして保存する。
- generation horizontal loaderは`MD/Z/GR/TVT_input`だけを`usecols`で読む。prediction、support manifest、schema、decoder manifestを書いてSHA freezeした後にだけraw unknown-suffix `TVT`とsaved exp223 control `target`を読む。
- saved exp223 controlはdecompressed SHA `0eb48b5516276b0ab7b2191a52a39ebb89d9997363cc7839ede519c7863baa0c`を必須照合し、row identity one-to-one後にreadoutする。ローカル保存物の実SHAと必要列を確認済み。
- overall、stable SHA256 5 folds、distance、true-TVT inside/outside、hidden-like 2面、by-well、step-delta metricsとtechnical/performance hard gateを実装した。
- raw input、external input、support mask、prediction、schema、decoder、metricsのSHA manifestを実装した。model/submissionは生成しない。
- configを`implementation_complete_not_run`へ更新し、`implementation=true`、`run_variant=false`、`kaggle_cpu_push_approved=false`、正規Notebook未採用を固定した。
- 親exp223のdescriptor実装にある旧pandas `fillna(method=...)`は現環境で実行不能だったため、意味が同じ`.bfill().ffill()`へ互換置換した。親ファイル自体は変更していない。

### Notebook構成比較

親exp223にはcompact self-contained sourceがなく、正規trainは190行のorchestrationと`exact_hmm_smoother.py` 1,211行、comparison helper 488行へ分離されていた。exp296は実行時helper importなしの1本へ統合し、Imports/contract、runtime/path/SHA、parent helpers、support gate、exact HMM、truth-free generation、truth-late metrics、full orchestration、execution switchの10章で構成した。正規scaffold Notebookは上書きしていない。

### 検証コマンドと結果

```bash
.venv/bin/python -m py_compile experiments/exp296_exp223_self_gr_known_tvt_support_gate/exp296_exp223_self_gr_known_tvt_support_gate_compact_selfcontained_train.py
.venv/bin/ruff check experiments/exp296_exp223_self_gr_known_tvt_support_gate/exp296_exp223_self_gr_known_tvt_support_gate_compact_selfcontained_train.py experiments/exp296_exp223_self_gr_known_tvt_support_gate/tests/test_exp296_exp223_self_gr_known_tvt_support_gate.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp296_exp223_self_gr_known_tvt_support_gate/exp296_exp223_self_gr_known_tvt_support_gate_compact_selfcontained_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp296_exp223_self_gr_known_tvt_support_gate/exp296_exp223_self_gr_known_tvt_support_gate_compact_selfcontained_train.py
.venv/bin/pytest -q experiments/exp296_exp223_self_gr_known_tvt_support_gate/tests/test_exp296_exp223_self_gr_known_tvt_support_gate.py
.venv/bin/pytest -q
make validate-exp EXP=exp296_exp223_self_gr_known_tvt_support_gate
make validate-template
.venv/bin/python .agents/skills/kaggle-review-exp/scripts/review_exp_docs.py exp296 --root .
make update-summary
```

- full Ruff: PASS。
- Jupytext conversion / round-trip: PASS。
- 専用contract tests: `12 passed`。
- repository tests: `329 passed, 1 skipped in 145.36s`。
- strict experiment validation: PASS。
- project template validation: PASS。
- reviewer: steering、backlog、README、SESSION_NOTES、config、metrics、result、support contractの対象10文書すべてで各core evidence categoryがPASS。
- experiment summary: `293 experiment(s)`へ更新し、exp296を`implementation_complete_not_run`として反映。
- ローカル実データ/HMM run: 0。
- Kaggle package/push/run: 0。

## 再現性メモ

- seed policy: HMM/self-GR/support gateはno RNG、reporting foldはstable SHA256 well hash。
- stochastic components: なし。
- CPU/GPU runtime: Kaggle CPU、outer workers 2、Numba threads 2で16,667.265秒。GPUなし。
- deterministic anchor: false。train-side no-training auditでありsubmission anchorではない。
- parent control: exp223 feature decompressed SHA `0eb48b55...aa0c`を必須照合する。
- support/prediction/metrics SHA: Kaggle output summaryと選択取得artifactで記録・照合済み。
- model manifest: 学習modelなし。
- submission SHA: 対象外。
- Kaggle kernel id / version: `kentookumura/exp296-exp223-self-gr-known-tvt-support-gate-train` / version 3 COMPLETE（id_no `127897387`）。

## 次のアクション

完了済み。strict support gateを棄却し、救済variant、inference、submissionへ進めない。

## 2026-07-19 Kaggle CPU実行承認

ユーザーの「実行」を、正規train Notebookへのcompact候補採用とKaggle CPU pushの明示承認として受けた。承認対象は新variant `hmm_selfgr_boost_only_a070_c100_known_tvt_support_gate` 1本だけである。実行前countはactive variant 1、HMM well-runs 773、LightGBM config / trained fold / booster `0 / 0 / 0`、親exp223 control再実行0。inference、submission、control再実行は承認対象外のまま無効とする。

Kaggle runtimeはCPU、private、internet off、GPU off、outer workers 2、Numba threads 2とする。保存済みexp223 controlはKaggle inputから読み、decompressed SHAとrow identityを照合する。実行中は別slugへ切り替えず、同じcanonical kernel idを監視する。

初回local package監査で、`runtime.kaggle.planned_kernel_sources`はprepareスクリプトのmetadata入力ではなく、生成された`kernel_sources`が空になることを確認した。push前に`runtime.kaggle.train_kernel_sources`へexp223 saved controlとexp115 hidden-like kernelを明示し、同じcanonical packageを再生成する。実験variant、HMM、metric契約は変更していない。

再生成後のmetadataはcanonical id `kentookumura/exp296-exp223-self-gr-known-tvt-support-gate-train`、一致するtitle、private、CPU、internet/GPU/TPU off、run-on-push trueを満たした。kernel sourcesはexp223 saved controlとexp115 hidden-likeの2件を確認した。bootstrap ZIP内と正ファイル・package copyのbyte一致を確認し、config SHAは`05111dd61b59498d51b02affec67c1d6d88c3dd8bec40fd25cfd7cac11c864a6`、compact source SHAは`1156c5fb141651df68f30045dd0a5f68931d8b0c563c6acf799582a6eb926cfb`。push notebook SHAは`363756366e486b25ad7be2cc588b166220b68cd7a3b44fc69ea9ee7685045974`、metadata SHAは`5a1062e57798e582840b67dd593788b4834472a0aadcdc0c418d072e1aa9ce2a`である。

Kaggle CPU version 1（id_no `127897387`）のpush自体は成功したが、約18秒でERRORになった。bootstrap、Numba、実行count表示まではPASSし、HMM well-run / boosterは0。原因はresolverが`/kaggle/input/competitions/.../train`を明示候補に含めず、fallbackの先頭にあった3-well public test directoryを選んだためで、`raw input well count mismatch: 3 != 773`となった。saved control、truth、metricには未到達である。

version 2では同じcanonical kernel idを使う。resolverへcompetition train pathを追加し、全候補のhorizontal file inventoryがexpected 773と一致するdirectoryだけを採用するfail-closeへ変更した。public testをskipしてexpected inventoryを選ぶcontract testを追加し、科学的variant、support gate、HMM、control、performance gateは変更していない。

version 2 packageもmetadata/bootstrap監査を再実行してPASSした。config SHA `dab65aa4af602f93962a59d3a93fba7943c365daf1016299dc7c520550823710`、compact source SHA `0d25d0b4cac0ced71b4fc75552135c0dbfea3f84a23d8d17bb3baef6d0975370`、push notebook SHA `1a006d7f440214095f65e4455246057b6c75eaba1ed7c6561c346dc7ff8ac089`、metadata SHA `5a1062e57798e582840b67dd593788b4834472a0aadcdc0c418d072e1aa9ce2a`である。専用contract testsは`13 passed`、Ruff、py_compile、Jupytext、strict validationを再度PASSした。

Kaggle CPU version 2は773-well inventoryを通過し、先頭4 wellsのdispatchまで進んだが、最初のHMM呼び出し前に`TypeError: run_hmm2_known_tvt_support_gate() got an unexpected keyword argument 'source'`でERRORになった。`model.hmm.source`はlineage監査用metadataであり、runtime scientific argumentではない。version 3ではruntime許可キーを明示し、`source`だけを除外、未知キーはfail-closeする。HMM完了well 0、saved control/truth/metric未到達、booster 0であり、科学条件は変更していない。

version 3 package監査はPASS。config SHA `f9a60b4db3864470283048184694b06576afad3770dc07c34cc9ee7d31619166`、compact source SHA `2fb814ef92daf8b323bde6292bcd45b9ee7c2827255bd8acdc95aad77563d056`、push notebook SHA `27a79f6a5d24018c1904f888cc86cbe1c07dc797bc9e57c5868482314d2ca346`、metadata SHA `5a1062e57798e582840b67dd593788b4834472a0aadcdc0c418d072e1aa9ce2a`。専用contract testsは`14 passed`である。

Kaggle CPU version 3を同じcanonical kernel idへpushし、初期statusは`RUNNING`。version 2の失敗時点を越えた後も複数回`RUNNING`を確認した。ローカルrepository full testsは`331 passed, 1 skipped in 28.72s`。version 3完了までは数値結果やhard gate判定を記録しない。

## 2026-07-20 Kaggle CPU version 3完了

### 実行結果

canonical kernel `kentookumura/exp296-exp223-self-gr-known-tvt-support-gate-train` version 3は`COMPLETE`となり、実験statusは`completed_train_side_guard_failed`だった。3,783,989 rows / 773 wells、新variant 1本 / 773 HMM well-runs、LightGBM config / trained fold / booster `0 / 0 / 0`、親control再実行0を16,667.265秒（4.630時間）で完走した。inferenceとsubmissionはfalseのままである。

overallはsaved exp223 control RMSE `11.349942946`に対しstrict gate `12.159749140`、delta `+0.809806194 ft`。MAEは`6.471268657 -> 7.041758436`、within10は`0.794840577 -> 0.767777602`だった。configに事前記録したparent RMSE `11.349950650`との差は約`7.7e-6 ft`で、同一truth-late join上の再集計値を判定に用いた。

fold deltaは`+0.965195 / +0.530868 / +0.637723 / -0.258307 / +2.289724 ft`で、改善1/5 folds。true-TVT-inside-known-rangeは`-0.571802 ft`改善したが、outside-known-rangeは`+2.341425 ft`悪化した。1000+、hidden-like spatial、hidden-like typewell-purgedもそれぞれ`+0.897491 / +1.110813 / +1.118634 ft`悪化した。

by-wellは302改善 / 471悪化 / 0同値、p95 delta `+1.728087 ft`。worst `2364716c`はRMSE `4.661378 -> 44.349169`、delta `+39.687791 ft`。best `028d7b28`は`-19.815744 ft`だった。

### Hard gate判定

technical 12/12はPASSした。input wells 773、finite coverage 1.0、saved control row identityとdecompressed SHA exact、outside contribution max abs 0、inside boost delta max abs 0、base/self-GR config parity、truth-before-freeze 0、control再学習0、LightGBM/fold/booster 0を確認した。

performanceはinside-range deltaとstep p99非悪化の2/10だけPASSし、pooled、改善fold数、outside-range、1000+、hidden-like 2面、by-well p95、worst-wellの8項目をFAILした。総合判定はFAILである。

### Artifact監査

AGENTS.mdの取得最小化方針に従い、Kaggle output archive全体は取得せず、metrics/manifest/schemaに一致する小規模artifactだけを`kaggle kernels output --file-pattern`で取得した。summaryにSHAが記録された13ファイルはローカルbyte SHAと全件一致した。大容量prediction freezeとOOF readoutは取得せず、次の記録SHAを正とする。

- saved control decompressed: `0eb48b5516276b0ab7b2191a52a39ebb89d9997363cc7839ede519c7863baa0c`
- prediction decompressed / raw gzip: `e87f1c64a870991b65f310891b316e2854f6c717947df923d60ab2f73c5ac99a` / `e8aabd98ed0d7b675b2d8f20d793129b02ada3c7a571a8706a72099b2bb07261`
- OOF decompressed / raw gzip: `bd2db24c2598cf2d2d1490a765a338a8e90dd6179048ca1650efed1d90d916a1` / `0d8a7a651fee4032ebe8b026457f5e2d9a44b4f449e1a2c73f2f751fcbd1f3bf`
- support manifest: `b537eb37a81155da031f46b8472a848e4bfed257e845a745b0f4890e8383b209`
- downloaded summary: `74701c7642c86d3e9b019f46b11491c12d00eacec74b7a3a42b013fa13ffc4fc`
- downloaded kernel log: `8efb5e7a8065e0caafaa648fb41d50784d0110ee880805943a63f77031c581a0`

取得先は`kaggle/output/train_v3/`である。

### 結論とbacklog反映

range内rowの改善よりrange外rowの悪化が大きく、「candidate stateがprefix known TVT range外ならself-GR evidenceを参照しない」というhard conditionは支持されない。same-well motifはrange外でも有益な場合があり、strict zeroは大きなmode errorを増やした。

事前登録どおりpadding、nearest/hole-aware/soft gate、alpha/clip/window/top-k/threshold救済を行わずbranchを閉じる。inference、submissionも行わない。`backlog/KAGGLE_DIRECTION.md`からexp296未着手行を削除し、既存`self_gr_quality_addonly_features_on_exp092`へtarget-free quality / posterior outside-support mass / known-range overlapのfeature-only診断として証拠を統合した。これはexp296の直接救済ではない。

### 完了時validation

```bash
.venv/bin/python -m json.tool experiments/exp296_exp223_self_gr_known_tvt_support_gate/metrics.json
make update-summary
make validate-exp EXP=exp296_exp223_self_gr_known_tvt_support_gate
make validate-template
.venv/bin/python .agents/skills/kaggle-review-exp/scripts/review_exp_docs.py exp296 --root .
```

- metrics JSON parse / config completion contract: PASS。
- experiment summary: `294 experiment(s)`へ更新し、status `completed_train_side_guard_failed_closed`、CV `12.15974913969598`を反映。
- strict experiment validation: PASS。
- project template validation: PASS。
- reviewer: core evidence categoryは対象文書群にすべて存在。
- コードはKaggle実行前の検証後に変更していないため、専用`14 passed`、repository `331 passed, 1 skipped`を最終code test結果とする。
