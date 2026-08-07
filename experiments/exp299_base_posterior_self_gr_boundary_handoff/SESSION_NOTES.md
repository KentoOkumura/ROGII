# exp299_base_posterior_self_gr_boundary_handoff セッションノート

## 目的

range外candidateではself-GRをexact 0にする要件を維持しながら、exp296で生じたknown-max境界の相対priorをbase-only posterior handoffとconditional normalizationで除去する1 variantを、実装前に反証可能な契約へ固定する。

## 現在の状態

- Route: `ensemble`
- 状態: Kaggle private CPU version 2完了、`completed_train_side_guard_failed`でbranch close
- CV / LB: `11.789577561` / 未提出
- implementation / active variant / v2 executed HMM well-run / model / booster: `1 / 1 / 1,546 / 0 / 0`
- 承認run: 1 scientific variant / Pass A+B 1,546 HMM well-runs / LightGBM config・trained fold・booster `0/0/0` / control再実行0 / CPU

## 2026-07-20 実行承認とpush前count

- ユーザーの「実行してください」を、別名compact train候補の正規train Notebook採用と、同じ固定契約によるKaggle private CPU train 1回の明示承認として記録した。
- scientific variant: `1`（`hmm_selfgr_base_posterior_conditional_handoff_a070_c100`）。
- LightGBM config / trained fold / booster: `0 / 0 / 0`。
- HMM: Pass A base-only `773` well-runs + Pass B handoff `773` well-runs = 合計`1,546` well-runs。
- saved exp209 / exp223 / exp296の参照artifactを使い、parent/control再学習は`0`。
- Kaggle private CPU、GPU/internet無効、outer workers 2、Numba threads 2。推定8-10時間。
- inference、raw-test生成、submissionは承認範囲外のため`false`を維持する。

## 2026-07-20 Kaggle CPU train version 1開始

- 最初のpackageはtitle/slugが52文字で、Kaggle `SaveKernel`がHTTP 400を返した。`kaggle kernels list --mine --search exp299`は`Not found`で、kernel/versionは作成されていない。
- 科学契約とNotebook本体を変えず、Kaggle名だけを47文字の`kentookumura/exp299-base-post-self-gr-boundary-handoff-train`へ短縮した。
- private / CPU / GPU false / TPU false / internet false / run-on-push trueでversion 1をpushした。
- Kaggle pull-backでid_no `127957958`、version `1`、competition source `rogii-wellbore-geology-prediction`、kernel sources exp115/exp209/exp223を確認した。
- 2026-07-20T10:55:04+09:00に開始し、10:59:05+09:00の再確認でもstatusは`RUNNING`。2回の初期`kaggle kernels logs`は空で、即時failureは確認されていない。長時間runの初期は空ログになり得るため再pushしない。
- URL: https://www.kaggle.com/code/kentookumura/exp299-base-post-self-gr-boundary-handoff-train
- run source SHA256: `8cccf41cfa9b95d4fc4c0181a40efef2c4f71f157aae459b97e2b3ea749491f8`。
- run config SHA256: `ba57fe479d523ab4d537cdf537971166740e898120b752461c0d9662d047c7f7`。
- canonical train Notebook SHA256: `c50a0f689a11243614d430a76db7bb2598dffd39cc132a96293727fdf66abeee`。
- package metadata SHA256: `24ba05fa601422fbfa4bfc68b35af39d33b54601c05c688c1696bc666786a1af`。
- pull-back Notebook SHA256: `a3151f04ccf60504aa1d1750aada99cf7cbdfca3cd1cc2a37a61f6a25305caef`。
- 1回分のpush承認は消費済みとし、version 2以降のrepushは新しいユーザー承認なしに行わない。

## 2026-07-20 Kaggle CPU train version 1失敗とlocal修正

### 一次ログ

- status: `ERROR`。log最終timestampは`40053.639842005 sec`（約11.126時間）。
- Pass A base-only / Pass B handoffは各773 wells、合計1,546 HMM well-runsを完了した。
- 3,783,989 rows / 773 wellsのordered id/well identityはexact一致した。
- 失敗箇所はgeneration freeze後、unknown-suffix truthとsaved exp223 controlを読む前のsaved exp209 base parity guard。
- parityはmax abs `0.00048437499935971573 ft`、mean abs `0.00023841843883720324 ft`で、固定atol `1e-5 ft`を超えてfail-closeした。
- `kaggle kernels files`は空で、failure前に`/kaggle/working/artifacts`へ書いたfreeze生成物はKaggle outputとして取得できない。CV/performance readoutは存在しない。

### 原因

- exp209は`_numeric_frame()`で`hmm_mean_tvt`をfloat32へcastしてからCSVへ保存する。
- pandasはfloat32を短い十進表記（例`11236.02`）でCSVへ書く。これをexp299が既定float64で読むと、元のbinary float32値`11236.01953125`との差が`0.00046875 ft`生じる。
- exp299はcandidate側だけをfloat32へ戻し、exp209 reference側をCSV parse後のfloat64のまま比較していた。観測された最大差はこのfloat32 CSV round-trip差と一致する。
- HMM、handoff、入力identity、row coverageの不一致ではなく、parity comparisonのserialization contract bug。

### 最小修正

- candidateとexp209 referenceの両方をfloat32へ正規化してからfloat64で差分を計算する。
- atolは`1e-5 ft`のまま維持し、float32 ULPへ緩和しない。
- 約12,000 ftの実値でraw CSV decimal差が`>1e-5`、float32復元後のparity差がexact 0になる回帰testへ更新した。
- 専用tests `12 passed`、py_compile、Ruff F821、Jupytext round-tripはPASS。
- 修正source SHA256: `d033f3e4e1aa959f830b6a8bf3f698afdd223ec990a101da3a529fe7ec1df66f`。
- 修正正規train Notebook SHA256: `3133d91c166f3ea326cbe4447001a1cf3524cd9607f1dced5a9e62b8b901b27c`。
- version 2 package/pushは未実施。再実行は同じ1 variant / 1,546 HMM well-runs / 0 booster / control再実行0で約11.1時間を要するため、明示承認を待つ。

## 2026-07-20 Kaggle CPU train version 2再実行承認

- ユーザーの「実行してください」を、同じcanonical slugへのversion 2 private CPU push 1回の明示承認として記録した。
- scientific variant: `1`（v1と同じhandoff policy）。
- HMM: Pass A `773` + Pass B `773` = 合計`1,546` well-runs。
- LightGBM config / trained fold / booster: `0 / 0 / 0`。
- exp209 / exp223 / exp296 control再実行: `0`。GPU / inference / submission: `0 / 0 / 0`。
- v1からのコード差分はexp209 CSV referenceもfloat32へ戻して比較するparity修正だけ。atol `1e-5 ft`、HMM、handoff、freeze順序は不変。
- CPU所要時間はv1実測を基に11-12時間と見積もる。

## 2026-07-20 Kaggle CPU train version 2開始

- 同じcanonical kernel `kentookumura/exp299-base-post-self-gr-boundary-handoff-train`へversion 2をpushした。新slugは作成していない。
- Kaggle pull-backでid_no `127957958`、private、CPU、GPU/TPU/internet false、competition source 1件、exp115/exp209/exp223 kernel sourcesを再確認した。
- 2026-07-20T22:22:08+09:00時点でstatus `RUNNING`。初期CLI logsは空で、実行中は空ログになる既知挙動のため再pushしない。
- 2026-07-20T22:27:35+09:00に`kaggle kernels status`で再確認し、引き続き`RUNNING`。
- source SHA256: `d033f3e4e1aa959f830b6a8bf3f698afdd223ec990a101da3a529fe7ec1df66f`。
- pushed config SHA256: `ebaad8ee9f96076938468da6f10a1c6207238e31c800ed69b7dd1de2fadab6c4`。
- canonical train Notebook SHA256: `3133d91c166f3ea326cbe4447001a1cf3524cd9607f1dced5a9e62b8b901b27c`。
- package metadata SHA256: `24ba05fa601422fbfa4bfc68b35af39d33b54601c05c688c1696bc666786a1af`。
- pull-back Notebook SHA256: `752822d34ccc2987eddb7cc0aa5153a90985a71bcad7b348b24411e676c7a2d3`。
- version 2の1回分の承認は消費済み。version 3以降のpushは新しいユーザー承認なしに行わない。

## 2026-07-21 Kaggle CPU train version 2完了

- 2026-07-21T07:34:42+09:00に`kaggle kernels status`で`COMPLETE`を確認し、通常`kaggle kernels logs`で完了ログを取得した。Kaggle summary生成時刻は2026-07-21T04:36:40+09:00。
- 最終train-side status: `completed_train_side_guard_failed`。elapsed `22,481.454033613205 sec`（約6.245時間）。
- Pass A/Bは各773 wells、合計1,546 HMM well-runs。3,783,989 rows / 773 wells、prediction finite coverage 1.0。
- version 1で失敗したexp209 parityは、float32保存契約へ両側を正規化した結果、ordered id/well exact、max/mean abs `0.0 / 0.0 ft`でPASS。reference decompressed SHA `8e2f4236...7ae5`もexact一致した。
- saved exp223 control RMSE `11.349942946009358`に対し、candidate RMSE `11.7895775608145`、delta `+0.43963461480514177 ft`、改善fold `0/5`。
- exp296 RMSE `12.15974913969598`からは`-0.37017157888148 ft`回復したが、exp223を上回れずpromotion FAIL。
- performance gateは2/11 PASS。outside / inside known-range deltaは`+0.478309948 / +0.415018842 ft`、upper-boundary 0--12は`+0.971109945 ft`、1000+は`+0.508852469 ft`、by-well p95 / worstは`+1.454561921 / +35.990274405 ft`。hidden-like 2面だけ`-0.014390249 / -0.013615362 ft`改善した。
- technical gateは24/25 PASS。唯一のFAILは`row_gate_max=1.0000000000000029`が上限1.0を`2.8866e-15`だけ超えた浮動小数丸め。exp209 parity、row identity、input SHA、outside contribution exact 0、boundary neutral exact 0、conditional mass error `2.454881e-10 <= 1e-6`、truth/control-before-freeze 0、run countはPASSした。
- この微小technical超過を許容してもperformance 9/11 FAILは変わらず、事前固定fail actionどおりhandoff/fade/normalizer/alpha/clip/support/threshold救済なしでbranchを閉じる。
- prediction decompressed SHA: `6d354abc32df1989ed2a74da16f7e2dbbf7a99e2110a8ec216dad7ad2611a28e`。
- OOF readout decompressed SHA: `2738281aa92f5d54c8c0f5172b9e3b262945d055c8b6db5ea5b4c9af3cac7266`。
- summary SHA: `c5e98734355fbec17b3fccb0e45cfa84034f1e0a6203ee8dd2b6b0e8df1efeae`。
- 完了ログにCV、全gate、生成物path、SHAが揃っているため、Kaggle output archiveは取得していない。
- inference / submissionは`false`のまま。version 3、実推論、提出へ進めず、本結果だけを根拠とする救済backlogも追加しない。

## 2026-07-20 設計確定

### 作成コマンド

```bash
make new-steering EXP=exp299_base_posterior_self_gr_boundary_handoff
make new-exp EXP=exp299_base_posterior_self_gr_boundary_handoff
```

### 根拠

- exp223はself-GR weak boostでexp072 `likpf_mean`から`-0.244947 ft`改善したが、worst-well `+46.954683 ft`のriskがあった。
- exp296はoutside exact-zero / inside exact-parityを実装しtechnical 12/12 PASSしたが、exp223比overall `+0.809806 ft`、outside `+2.341425 ft`、worst `+39.687791 ft`で悪化した。
- unknown suffixのoutside 1,459,531 rowsは全て`known_tvt_max`より上、below-minは0だった。state-wise maskは実質的に未来方向の上側half-gridだけからself-GRを除去した。
- exp296のinside scopeは`-0.571802 ft`改善しており、self-GRを安全なinside条件付きsignalとして使う仮説は完全には否定されていない。

### 固定した設計

- Pass A Type Well-only exact HMM posteriorをtarget-free controllerとしてSHA freezeする。
- support外candidate contributionは全row exact 0。
- base meanがoutside/boundaryならrow全体exact 0。
- base meanのboundary距離をexp223 sigma 12 ftでfadeし、base inside posterior massを掛けてrow weightを作る。
- support内ではbase posterior重み付きconditional normalizerを引き、inside/outside総massを保持する。
- Pass B posterior/final predictionをgateへ戻さない。
- 1 scientific variant、2 HMM passes、1,546 well-runs、0 boosterへ固定する。
- pooled exp223比`-0.05 ft`、4/5 folds、outside `-0.10 ft`、inside/1000+/hidden-like/p95非悪化、worst `+0.25 ft`以内を全必須とする。
- FAIL時はhandoff/fade/normalizer/alpha/clip/support/threshold救済なしで閉じる。

### 今回変更していないもの

- template train/inference Notebookと`settings.py`。
- 実験ロジック、Jupytext source、helper、tests。
- Kaggle package、kernel metadata、bootstrap、train、output。
- inference、submission、submission log。

## 再現性メモ

- `docs/06_reproducibility.md`を確認済み。
- seed policy: HMM/self-GR/handoffはno RNG、reporting foldはstable SHA256 well hash。
- stochastic components: なし。
- runtime: Kaggle private CPU、outer workers 2、Numba threads 2、Pass A/B同一worker内順次実行。v1実測約11.1時間、v2見積11-12時間。GPU/internetなし。
- deterministic anchor: false。train-side no-training diagnostic。
- required SHA: source/config/bootstrap、raw input、saved exp209/223/296、Pass A posterior、support、row gate、conditional contribution、Pass B prediction、schema、metrics。
- gzipはdecompressed content SHAを主証拠にする。
- model/submission SHA: 対象外。trained model/submissionなしをmanifestへ記録する。

## 2026-07-20 compact実装

### 実装範囲

- `exp299_base_posterior_self_gr_boundary_handoff_compact_selfcontained_train.py`をJupytext percent形式で作成し、対応する別名`.ipynb`候補へ変換した。
- Pass A Type Well-only exact HMM posterior/meanを計算し、posterior・mean・grid・raw input identityをself-GR handoff前にSHA化する。
- exp223 raw self-GR surfaceを固定し、inclusive visible-prefix support、12-ft boundary fade、base inside posterior massによるrow gate、support-mass-preserving conditional normalizerを実装した。
- Pass BはPass A posteriorだけをcontrollerとして使い、variant posterior/final prediction/truth/controlをgateへ戻さない。
- per-well manifestへPass A posterior/mean、support、row gate、conditional contribution、Pass B prediction SHAとtechnical diagnosticsを保存する経路を実装した。
- generation freeze後にtarget-freeなsaved exp209 `hmm_mean_tvt`だけを読み、float32保存契約へ合わせてordered id/well parity `<=1e-5 ft`を検証する。
- unknown-suffix truthとsaved exp223 performance controlはgeneration freezeとexp209 parityの後にだけ読み、overall/fold/inside-outside/upper-boundary 0-12・12-24・24+/distance/hidden-like/by-well/step metricsを作る。
- `exp299_base_posterior_self_gr_boundary_handoff_compact_selfcontained_inference.py`はraw testを読まず常にfail-closeする候補として追加した。
- 正規train/inference Notebookと`settings.py`は変更していない。Kaggle package/outputも作成していない。

### 実装時count

- scientific variant: 1。
- future Pass A / Pass B: `773 / 773` wells、合計`1,546` HMM well-runs。
- LightGBM config / trained fold / booster: `0 / 0 / 0`。
- exp209/exp223/exp296 control retraining: 0。
- GPU / inference / submission: `0 / 0 / 0`。

### 検証コマンドと結果

```bash
.venv/bin/python -m py_compile experiments/exp299_base_posterior_self_gr_boundary_handoff/exp299_base_posterior_self_gr_boundary_handoff_compact_selfcontained_train.py experiments/exp299_base_posterior_self_gr_boundary_handoff/exp299_base_posterior_self_gr_boundary_handoff_compact_selfcontained_inference.py tests/test_exp299_base_posterior_self_gr_boundary_handoff.py
.venv/bin/ruff check experiments/exp299_base_posterior_self_gr_boundary_handoff/exp299_base_posterior_self_gr_boundary_handoff_compact_selfcontained_train.py experiments/exp299_base_posterior_self_gr_boundary_handoff/exp299_base_posterior_self_gr_boundary_handoff_compact_selfcontained_inference.py tests/test_exp299_base_posterior_self_gr_boundary_handoff.py --select F821
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp299_base_posterior_self_gr_boundary_handoff/exp299_base_posterior_self_gr_boundary_handoff_compact_selfcontained_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp299_base_posterior_self_gr_boundary_handoff/exp299_base_posterior_self_gr_boundary_handoff_compact_selfcontained_inference.py
.venv/bin/pytest -q tests/test_exp299_base_posterior_self_gr_boundary_handoff.py
.venv/bin/pytest -q
make validate-exp EXP=exp299_base_posterior_self_gr_boundary_handoff
make validate-template
```

- py_compile / Ruff F821 / Jupytext round-trip: PASS。
- 専用tests: `12 passed`。
- repository tests: `354 passed, 1 skipped, 2 failed`。2 failureは今回未変更のexp296で、完了後statusが`kaggle_cpu_` prefixではないことと、閉鎖後`run_variant=false`なのにtestがpush approval guardまで進むことを期待する既知不一致。exp298記録時にも同じ2件が存在する。
- strict experiment validation / project template validation: PASS。
- Kaggle/ローカルHMM実行は行っていない。

### 親compactとの構成比較

- 構成参照元はexp296 compact self-contained train `2,260`行・10章。exp223にはcompact sourceがないため、exp296がexp223 HMM/self-GRを自己完結化した最短の親構成である。
- exp299 train候補は`2,758`行・10章を維持し、Pass A/Pass B、exp209 parity、upper-boundary readout、SHA manifestを追加した。薄いhelper呼び出しNotebookではない。

## 実行権限

- design/scaffold: 2026-07-20のユーザー指示で承認済み。
- implementation: 2026-07-20のユーザー指示で承認済み・完了。
- canonical train Notebook adoption: 2026-07-20のユーザー指示で承認済み。
- Kaggle CPU push: 2026-07-20のユーザー指示による1回承認をversion 1で消費済み。1 variant / 1,546 HMM well-runs / 0 booster / control再実行0をpush前に再確認した。
- inference/submission: 未承認。

## 次のアクション

version 2の完了を待ち、logs/cell outputからexp209 parity、technical/performance gate、生成物pathを確認する。version 3へのrepush、inference、submissionへは進まない。

## 設計validation

```bash
.venv/bin/python -m json.tool experiments/exp299_base_posterior_self_gr_boundary_handoff/metrics.json
make validate-exp EXP=exp299_base_posterior_self_gr_boundary_handoff
make validate-template
.venv/bin/python .agents/skills/kaggle-review-exp/scripts/review_exp_docs.py exp299 --root .
make update-summary
```

- metrics JSON parse / config design contract: PASS。
- strict experiment validation: PASS。
- project template validation: PASS。
- reviewer: core evidence categoryは対象文書群にすべて存在。
- 設計時点のexperiment summaryは`295 experiment(s)`、exp299 status `design_locked_not_implemented`だった。実装後は同時点で存在する`296 experiment(s)`へ再生成し、status `implemented_waiting_for_notebook_adoption`、parent exp223を反映した。
- train/inference Notebookはtemplate scaffoldで出力なし。`settings.py`以外の`.py`、compact source、helper、tests、Kaggle package/outputは作成していない。
