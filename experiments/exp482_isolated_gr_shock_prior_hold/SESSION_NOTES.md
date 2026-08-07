# exp482_isolated_gr_shock_prior_hold セッションノート

## 目的

exp440のbroad ambiguity holdを救済せず、raw GR単発shockとpast/future
message一致を同時に満たす場合だけ、current observationをrow-localに除外する
独立仮説を反証可能な1 candidateとして固定する。

## 現在の状態

- Route: `pf_beam`
- 状態: `stage_a0_eligibility_failed_closed`
- 優先度: 低・P3
- 親: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- CV / LB: Stage A1前に停止したためなし / なし
- implementation: 承認済み・compact実装済み
- canonical train Notebook採用 / Kaggle package / Stage A0/A1 run: 承認済み
- Stage 1 / inference / submission: 未承認
- train Notebook: compact self-contained実装を正規Notebookへ採用済み
- inference Notebook: generic template scaffoldのまま・実行禁止

## 設計根拠

- exp440: full OOF candidate `12.992063`、parent `11.938287`、
  `+1.053776 ft`悪化、positive fold`1/5`、ambiguous-row SSEも悪化。
- exp408: current emissionの新規wrong反転は`9/807,710 rows`で、
  persistent errorの主因はemission前のforward hysteresis。
- exp358: missing-distance downweightはoverall`+0.074283 ft`悪化、
  0/5 foldsで、missingness単独はtriggerにならない。
- exp363: weak reliability mass`0.589441`で広すぎた。
- exp389: Huber emissionはoverall`0.085546 ft`、5/5 folds改善したが
  well-tail FAIL。観測外れ値耐性には平均signalがあるがglobal適用は危険。

## 固定した差分

- raw GR shock:
  currentを除く`±5`行、左右finite各3以上、
  `max(1.4826*MAD,1.0)`に対するrobust z`>=4.5`、
  左右median差`<=2 scale`、`±2`行cluster抑制。
- past/future agreement:
  predictive meanとleave-one-out mean差`<=1.05 ft`、
  両std`<=6.0 ft`。
- current conflict:
  predictive→provisional mean shift`>=1.05 ft`、
  saved parent→leave-one-out output差`>=0.35 ft`。
- active rowだけ`normalize(predictive * beta)`のTVT meanへ置換。
- 親filtered state、backward message、次行以降のpredictionは不変。
- candidate 1本。threshold/window/output weight gridなし。

## 実装

- canonical trainのJupytext source:
  `exp482_isolated_gr_shock_prior_hold_compact_selfcontained_train.py`
- fail-closed inference候補:
  `exp482_isolated_gr_shock_prior_hold_compact_selfcontained_inference.py`
- 専用test:
  `tests/test_exp482_isolated_gr_shock_prior_hold.py`
- `alpha_t - emission_t + beta_t`を正規化してcurrent observationだけを除いた
  row-local LOO posteriorを計算する。親forward stateとbeta計算は変更しない。
- raw-only censusは773 wellsをwell ID順に走査し、isolated row座標とwell summaryを
  freezeする。supportはshock count降順、suffix rows降順、well ID昇順で32 wells。
- control matchingの`standardized_l1`は全773-well censusの2変数について
  population mean / population std（`ddof=0`）で標準化し、support選択順に
  without-replacement greedy matching、distance / well ID順でtie-breakする。
- 正規`*_train.ipynb`は上記sourceから生成した。正規`*_inference.ipynb`は
  generic scaffoldのまま変更していない。

## 実行量契約

### Stage A0

- raw-only census: 773 wells / 3,783,989 suffix rows。
- HMM / model / booster / PF / Beam / GPU: 0。
- target-free fixed64:
  support32 + zero-shock matched control32。

### Stage A1

- scientific candidate: 1。
- unchanged exp209 message HMM replay: 64 wells。
- candidate state-modifying HMM: 0。
- saved exp209 prediction rerun: 0。
- LightGBM config / trained fold / booster / fitted model:
  `0 / 0 / 0 / 0`。
- PF / Beam / GPU: `0 / 0 / 0`。

### Stage 1

- Stage A0/A1全PASSと別承認時のみ。
- unchanged exp209 message HMM replay: 773 wells。
- candidate state-modifying HMM、saved parent prediction rerun、
  model / booster / PF / Beam / GPU: すべて0。

## コマンドログ

```bash
make new-steering EXP=exp450_isolated_gr_shock_prior_hold
make new-exp EXP=exp450_isolated_gr_shock_prior_hold
```

- 2026-07-30: steeringを先に作成し、その後templateからexperiment scaffoldを作成。
- 同時進行の別実験がexp450--458とexp480--481を使用したため、内容を変えず
  現在の最大番号の次にあたるexp482へ再採番した。
- 親コードやNotebookはコピーしていない。
- config、README、SESSION_NOTES、result、metricsだけをdesign-onlyへ更新。
- design-only確定時点ではcandidateコード、専用test、Jupytext source、
  Kaggle packageは作成していなかった。

### 2026-07-30 実装

```bash
.venv/bin/python -m py_compile \
  experiments/exp482_isolated_gr_shock_prior_hold/exp482_isolated_gr_shock_prior_hold_compact_selfcontained_train.py \
  experiments/exp482_isolated_gr_shock_prior_hold/exp482_isolated_gr_shock_prior_hold_compact_selfcontained_inference.py
.venv/bin/ruff check \
  experiments/exp482_isolated_gr_shock_prior_hold/exp482_isolated_gr_shock_prior_hold_compact_selfcontained_train.py \
  experiments/exp482_isolated_gr_shock_prior_hold/exp482_isolated_gr_shock_prior_hold_compact_selfcontained_inference.py \
  tests/test_exp482_isolated_gr_shock_prior_hold.py --select F821,E9
.venv/bin/pytest -q tests/test_exp482_isolated_gr_shock_prior_hold.py
.venv/bin/pytest -q \
  tests/test_exp408_hmm_message_rate_basin_audit.py \
  tests/test_exp440_ambiguity_gated_predictive_prior_hmm.py \
  tests/test_exp482_isolated_gr_shock_prior_hold.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test \
  experiments/exp482_isolated_gr_shock_prior_hold/exp482_isolated_gr_shock_prior_hold_compact_selfcontained_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test \
  experiments/exp482_isolated_gr_shock_prior_hold/exp482_isolated_gr_shock_prior_hold_compact_selfcontained_inference.py
make validate-exp EXP=exp482_isolated_gr_shock_prior_hold
```

- ユーザーの「exp482を実装してください」をimplementation承認として記録。
- `task` commandは環境に存在しなかったため、同等の
  `make validate-exp EXP=exp482_isolated_gr_shock_prior_hold`を使用。
- `py_compile`、Ruff F821/E9、Jupytext round-trip、strict experiment
  validationはPASS。
- 専用pytestは`14 passed`。
- exp408 / exp440 / exp482関連回帰pytestは`37 passed`。
- 親exp209 train Jupytext sourceは174行・6章、exp482候補は2269行・10章。
  exp482はraw census、manifest、message replay、LOO readout、truth-late、
  gate、生成物保存をNotebookセル上で追える構成にした。
- 数値testではparent posteriorを独立exp209実装、各rowのLOO posteriorを
  current emissionだけ0にした独立exp209 rerunと比較し、absolute
  tolerance`5e-7`で一致した。設計gateのsaved-parent parity上限`1e-5 ft`
  より十分小さい実装検証であり、科学的性能証拠ではない。
- Kaggle package、push、Stage A0/A1 run、ローカルNotebook実行は行っていない。

### 2026-07-30 Stage A0/A1実行承認とpre-push契約

- ユーザーの「実行してください」により、canonical train Notebook採用、
  Kaggle package、private CPUでのStage A0/A1 runを承認済みとした。
- canonical train Notebook:
  `exp482_isolated_gr_shock_prior_hold_train.ipynb`
- canonical train Notebook SHA256:
  `a498818082b9e29240fac8a57b06f9646433f3f0a434d18dfe51fcca1fdcfc72`
- planned kernel:
  `kentookumura/exp482-isolated-gr-shock-prior-hold-train`
- planned title: `exp482 isolated gr shock prior hold train`
- internet / GPU: `false / false`
- Kaggle kernel source:
  `kentookumura/exp209-joint-exact-parity-train`,
  `kentookumura/exp226-k16-kappa-repro-train`
- scientific candidate: 1。
- Stage A0 raw-only census: 773 wells / 3,783,989 suffix rows。
- Stage A1 unchanged exp209 message HMM replay: 64 wells。
- candidate state-modifying HMM: 0。
- saved exp209 prediction rerun: 0。
- LightGBM config / trained fold / booster / fitted model:
  `0 / 0 / 0 / 0`。
- PF / Beam / GPU: `0 / 0 / 0`。
- 親controlは保存済みexp209 predictionを参照し、再学習・再生成しない。
- Stage 1最大773 HMM replay、inference、submissionは今回の承認範囲外。
- strict package:
  `experiments/exp482_isolated_gr_shock_prior_hold/kaggle/train`
- packaged Notebook SHA256:
  `3a3d8c06867d01d2368723896eb6521a6dfcfb2117f5f7352682c280efe6c85c`
- kernel metadata SHA256:
  `bc75840cc935fd777111af14f6ccaf575bcbb1bcf74823f8f6cf08c33d48903a`
- package metadataはprivate、CPU、internet disabled、run-on-push、
  exp209/exp226 kernel sourceを確認済み。
- `2026-07-30 12:05:48 UTC`: kernel v1をpush。
- Kaggle `id_no`: `129168015`。
- push直後にpullしたmetadataでslug、title、private、CPU、internet disabled、
  exp209/exp226 kernel sourceを再確認した。
- push直後のstatus: `KernelWorkerStatus.RUNNING`。

### 2026-07-30 Kaggle Stage A0結果

- kernel v1 / `id_no=129168015`は`2026-07-30 12:10:10 UTC`に
  `KernelWorkerStatus.COMPLETE`。
- Kaggle log elapsed: `263.99671437 sec`。
- raw-only census: 773 wells。
- raw-shock rows artifact: 21,027 rows。
- isolated raw-shock rows: 17,047。
- support wells: 763。
- zero-shock control wells: 10。
- eligibility:
  minimum isolated rows PASS、minimum support wells PASS、
  minimum zero-shock controls `10 < 32` FAIL。
- status: `stage_a0_eligibility_failed_closed`。
- raw census SHA256:
  `fdbb653e13bdd6132ffbe08d129fc44a744ed72b81fdc4d41ae04aa0848202cb`
- raw-shock rows decompressed SHA256:
  `1615aa3504eba71a90dc5c36f782ba79ea34162f3bd2876043b132f703332116`
- raw-shock rows raw gzip SHA256:
  `b8236c4d31ffb5c4659bf091f851b5badab9d048e8358a25fa30877658ac40e2`
- actual HMM replay / candidate state HMM / saved parent rerun:
  `0 / 0 / 0`。
- actual LightGBM config / booster / PF / Beam / GPU:
  `0 / 0 / 0 / 0 / 0`。
- fixed64 manifest、message、trigger、candidate prediction、truth/fold joinは
  eligibility FAIL後に作られていない。
- raw detectorは763/773 wellsで発火し、zero-shock well controlを32作る
  事前設計が成立しなかった。性能評価前のsupport failureである。
- no-rescue契約に従い、threshold/window/control定義を変更せず、再run、
  Stage A1、Stage 1、inference、submissionを行わない。
- 結果記録後の専用pytestは`14 passed`、exp408 / exp440 / exp482回帰は
  `37 passed`、Ruff F821/E9、format check、config YAML / metrics JSON parse、
  strict experiment validationはすべてPASS。
- Kaggleログだけでeligibility結果、実行量、artifact SHAを確定できたため、
  output archiveはダウンロードしていない。

## 再現性メモ

- `docs/06_reproducibility.md`: 確認済み。
- seed policy: RNGなし、stable well/row/state/message/manifest順。
- stochastic components: なし。
- CPU/GPU runtime: 将来実行時もCPU-only、GPU 0、internet disabled。
- freeze順:
  raw census → fixed64 manifest → HMM message → trigger →
  candidate prediction → SHA readback → truth/fold/role/error join。
- input / census / manifest / message / trigger / prediction / metrics SHA:
  実行時に記録。
- gzip: decompressed content SHAを主証拠にする。
- model / submission SHA: 対象外。
- rerun check: 初回成功runをdeterministic anchorとしない。

## Assumption

hidden inferenceでもunknown suffix全体のraw GRを観測でき、親exp209と同様の
future-message利用が可能である。成立しなければcausal版へ変更せず実装前に閉じる。

## 次のアクション

1. exp482をterminal closeとして記録する。
2. raw shockを再訪する場合は別実験の0-HMM/0-prediction preflightで、
   TVT/errorを読まないsensor-specificity証拠を先に固定する。
3. threshold/window/control定義の救済、再run、Stage A1、Stage 1、
   inference、submissionは行わない。
