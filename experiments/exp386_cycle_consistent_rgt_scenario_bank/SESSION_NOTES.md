# exp386_cycle_consistent_rgt_scenario_bank セッションノート

## 目的

6 地層を絶対面として個別補間せず、outer-train 上の順序付き RGT 対応グラフから、対象井戸ごとに物理的に異なる複数の TVT scenario を決定論的に生成できるか検証する。

## 現在の状態

- Route: `pf_beam`
- 状態: Kaggle CPU version 1完了・Stage 0 FAIL_CLOSE
- CV / LB: なし
- 設計承認: あり
- 実装: 承認済み・完了
- 正規notebook採用 / package / push / run: 承認済み
- inference / submission: 無効

## 2026-07-24 Stage 0完了確認

ユーザー連絡:

```text
完了しました
```

Kaggle evidence:

- canonical kernel: `kentookumura/exp386-cycle-consistent-rgt-scenario-bank-train`
- version: 1
- id_no: `128478384`
- status: `KernelWorkerStatus.COMPLETE`
- runtime: `2411.033028923 sec`
- 実行量: 1 variant / 5 graph solves / 16 target-well path solves /
  model・booster・HMM・PF・Beam各0 / parent control再生成0
- Kaggle files一覧で予定した13生成物とbootstrap support filesの存在を確認した。
  実ファイルを必要とする後続がないため、output archiveは取得していない。

Stage 0:

| check | 値 | 判定 |
| --- | ---: | --- |
| RGT source coverage | 0.9898471305934208 | PASS |
| graph query coverage | 0.0 | FAIL |
| scenario-bank well coverage | 0.0 | FAIL |
| scenario count p05 | 0.0 | FAIL |
| finite-path coverage | 0.0 | FAIL |
| cycle residual p95 | 2.363303287461948 | FAIL |
| target GR reads | 0 | PASS |
| valid Formation reads | 0 | PASS |
| valid suffix truth reads | 0 | PASS |
| source-valid overlap | 0 | PASS |
| projected runtime | 2867.2464886752423 sec | PASS |
| projected peak RSS | 1.1459312438964844 GB | PASS |

補足:

- preflightは100,721 rows / 16 wells / folds 0〜4を含む。
- graph構築は`2353.120898594 sec`、16対象井戸のpath処理は`10.6416681 sec`。
- target pathとreference-GR templateは空で、logical SHAはどちらも
  `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945`。
- RGT nodes、graph edges、cycle manifest、role-read ledgerのlogical SHAは保存された。
- logsでは全routeの棄却段階を分解していないため、scenario 0の詳細原因は断定しない。
  cycle residual非整合とscenario bank空は、それぞれ独立した停止理由として扱う。
- Stage 0の全AND gateが不成立なので、条件付き承認されたfull runは実行しない。
- edge / stretch / scenario count / diversityの救済、Stage 1/2、exp387、inference、
  submissionは実行せず閉じる。

完了後の記録検証:

- exp386専用contract test: `11 passed`
- exp386 / exp387 strict experiment validation: PASS
- 更新testのRuff: PASS
- exp386 / exp387 `review_exp_docs.py`: core evidence categoriesあり
- 全repository test: `918 passed / 6 skipped / 3 failed`。
  FAILは未変更の既存状態であるexp296のstatus/run flag期待2件と、
  exp384の実行承認flag期待1件。exp386/exp387更新による新規FAILはない。

## 2026-07-24 実行承認セッション

ユーザー指示:

```text
実行してください
```

承認範囲:

- compact self-contained trainとfail-closed inferenceを正規Notebookへ採用する。
- Kaggle private CPU / internet off packageを作成し、canonical train kernelへpushする。
- まず5 foldを含む16-well Stage 0 resource preflightを実行する。
- preflightの全AND gateがPASSした場合だけ`current_mode=full_run`へ切り替えて再pushする。
- inference、current-test、submissionは実行しない。

push前固定実行量:

- scientific variant: 1
- graph fold solve: 5
- target-well path solve: full run 773、preflight target subset 16
- scenario: 8〜32 / well
- fitted model / model config / trained fold / LightGBM booster: `0 / 0 / 0 / 0`
- HMM / PF / Beam: `0 / 0 / 0`
- parent exp226 control再生成: 0。保存済みOOFだけを読む。
- accelerator: Kaggle CPU、internet: off

canonical kernel:

- ID: `kentookumura/exp386-cycle-consistent-rgt-scenario-bank-train`
- title: `exp386 cycle consistent rgt scenario bank train`
- parent source: `kentookumura/exp226-k16-kappa-repro-train`

実行済み:

```bash
.venv/bin/python .agents/skills/kaggle-platform/shared/check_all_credentials.py
make prepare-kaggle-notebooks EXP=exp386_cycle_consistent_rgt_scenario_bank \
  EXTRA_ARGS="--notebook train \
  --kernel-id kentookumura/exp386-cycle-consistent-rgt-scenario-bank-train \
  --title 'exp386 cycle consistent rgt scenario bank train' \
  --run-on-push --strict --no-src"
```

push前監査:

- OAuthとlegacy API credentialはKaggle CLI認証に使用可能。
- canonical IDはpush前pullで取得不可、既知の親kernel statusは`COMPLETE`で、認証は正常。
- 親kernel filesに必須の
  `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction_train_oof_predictions.csv.gz`
  が存在する。
- metadata: private / CPU / TPU off / internet off / run-on-push。
- competition source: `rogii-wellbore-geology-prediction`
- kernel source: `kentookumura/exp226-k16-kappa-repro-train`
- canonical train/inferenceとcompact生成Notebookはそれぞれbyte一致。
- package埋め込みZIP監査: PASS。
  - ZIP SHA256: `fb6118dfee604ed3f6c2c09a13854ee00d41afd01f2961755588faaafbd687b4`
  - embedded/config SHA256: `1b0fbc560a2658bff2ee33560ff7bccab6ce3cc773df18e5f14a90e7007bfa6c`
  - packaged Notebook SHA256: `cfef9ec7012c8fbb991f7b5d1be0779e003f59043c4d5ac19ed32cf4cdcbbb85`
  - metadata SHA256: `1386bde0e2e0263a4a773536df29a201463428ac744cae8cdacdf66e6359ec82`
- embedded configでmode=`stage0_resource_preflight`、16 wells、1 variant、5 graph solves、
  full target-well solves 773、model・HMM・PF・Beam・booster各0、parent再生成0を確認。
- 専用contract test `11 passed`、Ruff、py_compile、Jupytext round-trip、
  strict experiment validationを再実行して全PASS。
- canonical version 1をpushし、id_no `128478384`をpull-backで確認した。
- pull-back metadataでprivate / CPU / TPU off / internet off、competition source、
  exp226 kernel sourceがpackageと一致した。Kaggle statusは
  `KernelWorkerStatus.RUNNING`。開始直後のlogsは空で、即時ERRORは確認されなかった。
- 2026-07-24 14:15:42 UTC時点も`KernelWorkerStatus.RUNNING`。full runは未開始。

## 2026-07-24 実装セッション

ユーザー指示:

```text
exp386を実装してください
```

実装内容:

- 正規notebook scaffoldを上書きせず、Jupytext percent形式の
  `*_compact_selfcontained_train.py` と生成Notebookを追加した。
- fixed formation orderから連続RGTを作り、64 ft window / 32 ft strideのnodeを生成する。
- outer-train wellだけで近傍24 unique well graphを作り、well pair最大4対応node、
  LOO q05--q95 stretch、Huber cycle solve、stable fundamental cycle basisを実装した。
- virtual start/end付きsource-well graphのdeterministic k-shortest simple routeを最大128本列挙し、
  prefix hard anchor、8〜32 scenario、0.5 ft RMS diversity、4成分prior costを実装した。
- target scenario入力は`MD/X/Y/Z/TVT_input`だけに限定し、target `GR/Formation/TVT`を
  target-free readerでfail-closed拒否する。
- scenario control pathと対応source nodeから、exp387用reference-GR templateを保存する。
- target-free logical SHA freeze、Stage 0、512-row rolling-origin Stage 1、
  truth-late H512 scenario oracle Stage 2、resource projection、artifact/SHA manifestを実装した。
- inferenceはfail-closed候補だけを追加し、実際のtest再生成やsubmissionは実装範囲外とした。

実行済み:

```bash
.venv/bin/python -m py_compile experiments/exp386_cycle_consistent_rgt_scenario_bank/exp386_cycle_consistent_rgt_scenario_bank_compact_selfcontained_train.py
.venv/bin/python -m py_compile experiments/exp386_cycle_consistent_rgt_scenario_bank/exp386_cycle_consistent_rgt_scenario_bank_compact_selfcontained_inference.py
.venv/bin/ruff check experiments/exp386_cycle_consistent_rgt_scenario_bank/exp386_cycle_consistent_rgt_scenario_bank_compact_selfcontained_train.py experiments/exp386_cycle_consistent_rgt_scenario_bank/exp386_cycle_consistent_rgt_scenario_bank_compact_selfcontained_inference.py experiments/exp386_cycle_consistent_rgt_scenario_bank/tests/test_exp386_cycle_consistent_rgt_scenario_bank.py
.venv/bin/pytest -q experiments/exp386_cycle_consistent_rgt_scenario_bank/tests/test_exp386_cycle_consistent_rgt_scenario_bank.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp386_cycle_consistent_rgt_scenario_bank/exp386_cycle_consistent_rgt_scenario_bank_compact_selfcontained_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp386_cycle_consistent_rgt_scenario_bank/exp386_cycle_consistent_rgt_scenario_bank_compact_selfcontained_inference.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp386_cycle_consistent_rgt_scenario_bank/exp386_cycle_consistent_rgt_scenario_bank_compact_selfcontained_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp386_cycle_consistent_rgt_scenario_bank/exp386_cycle_consistent_rgt_scenario_bank_compact_selfcontained_inference.py
```

検証:

- 専用contract test: `11 passed`
- py_compile: train / inference / settings PASS
- Ruff: PASS
- Jupytext train / inference生成・round-trip: PASS
- trainは10章、同一exp helper importなし、`__file__`依存なし
- 独立familyのため親compactはない。最も近い実装参照のexp383 compact trainは
  10章 / 2,448行、exp386 compact trainは10章 / 2,718行で、RGT、graph/cycle、
  scenario、Stage 0〜2をNotebookセル内で追える構成にした。
- 正規`*_train.ipynb` / `*_inference.ipynb`は未変更
- Kaggle package/push/run、ローカルNotebook実行、current-test、inference、submissionは未実施
- 全repository test: `907 passed / 6 skipped / 3 failed`。FAILは未変更の既存状態である
  exp296の完了後status/run flag期待2件と、exp384の実行承認flag期待1件。exp386専用11件と
  scaffold/validatorはPASSした。

## 2026-07-24 設計セッション

実行済み:

```bash
make new-steering EXP=exp386_cycle_consistent_rgt_scenario_bank
make new-exp EXP=exp386_cycle_consistent_rgt_scenario_bank
```

設計確定内容:

- exp383〜385 の拡張ではない topology-first RGT の独立系統とする。
- target GR は exp386 では禁止し、exp387 の尤度評価まで保留する。
- outer-train の TVT と6地層面を RGT に変換し、サイクル整合グラフを構築する。
- 対象井戸は軌跡と既知 prefix のみで 8〜32 scenario を生成する。
- 予測値は出さず、固定 scenario bank、graph-cost prior、outer-train 参照 GR template を保存する。
- Stage 0 / rolling-origin prefix / truth-late oracle の順で検証する。

## 予定計算量

- variant: 1
- outer fold graph solve: 5
- target-well path solve: 773
- fitted model / LightGBM / HMM / PF / Beam: 0
- exp226 control 再生成: なし

## 再現性メモ

- seed policy: RNG を使わず、fold・well・node・edge・cycle・path の不変キーで安定順序化
- stochastic components: なし
- logical SHA: fold manifest、RGT node/edge、cycle basis、scenario bank、参照 GR template、prediction に記録予定
- deterministic anchor: 未確立。将来の同一設定 rerun で content SHA 一致を確認してから確立
- 実装時点ではKaggle SHAは未生成だった。version 1でtarget-free logical SHAを保存済み。

## 次のアクション

1. exp386はStage 0 FAIL_CLOSEとして終了し、full runへ切り替えない。
2. exp387は必要なscenario bankが空のため未実装で閉じる。
3. 再訪する場合は同じ設定のrejection funnelとedge residual成分を測る
   0-prediction診断だけを別実験として事前設計し、独立承認を得る。
