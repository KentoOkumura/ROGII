# exp343 セッションノート

## 2026-07-22 設計確定

- 目的: lag-1相関0.724に対する対処を、robust emissionとは独立に検証する。
- Stage 0規模: tau安定性readout 1、HMM run 0、booster 0。
- Stage 1規模予約: 全gate通過と再承認時のみvariant 1、fold 5、773 well HMM run。
- leakage guard: raw finite known-prefix residualの連続runのみを使い、suffix TVTを参照しない。
- long-tail guard: coverage、fallback、full/last512一致、upper clip率、fold中央値比を固定した。
- 既知リスク: 過去の一律tempering失敗を踏まえ、tauの安定性が確認できなければHMMへ進めない。

## 未実施

2026-07-22時点ではコード実装、notebook実行、Kaggle push、成果物生成を行っていなかった。

## 2026-07-23 Stage 0 実装

- ユーザーの`exp343を実装してください`を、Stage 0実装とscaffold placeholderの
  正規train/inference notebook置換の承認として記録した。
- compact self-contained Jupytext train source（10章、約1,100行）とfail-closed
  inference sourceを実装し、compact notebookと正規notebookへ変換した。
- 科学的親exp281 train source（10章、1,526行）と比較し、runtime/path/SHA、
  exp226 group-safe fold、raw/typewell known-prefix処理、setup/orchestrationを維持した。
  HMM kernel、suffix truth join、prediction diagnosticsはStage 0対象外のため持ち込んでいない。
- exp226 OOFはdecompressed SHA
  `709eb726cc30da523f017ed0dbd0371967b88a91ddcf25578eb9356f28e4c609`
  をhard guardし、`well_id` / `fold`だけを読む。horizontal `TVT`、`tvt_pred`、
  `tvt_geop`、`gr_delta`、`tvt_true`、`error`、`abs_error`はStage 0で読まない。
- full known-prefixとlast-512 known-prefix raw rowsの2 windowを固定した。GR欠損を
  finite化せず、raw row indexが連続するrun内だけでlag 1--20 pairを作る。
- rhoは各lagの全contiguous-run pairを連結したpairwise Pearsonと明示した。
  finite residual 128未満、各lag pair 20未満、またはrho非finiteはfallbackとする。
- outer-valid foldを除くevaluable wellのraw tau中央値をwindow別priorとし、
  `alpha=n/(n+200)`のlog shrink後に`[1,4]`へclipする。
- stability Spearmanとabsolute log ratioは両windowがraw-evaluableなwellだけで計算する。
  median tau、upper clip率、fold median比はfull/last-512の悪い側を固定gateへ入れる。
- Stage 0 FAIL後のlag/support/clip/temperature/downsampling救済はコードとgateで禁止した。
- Stage 1 decoder、HMM、prediction、inference、submissionは実装していない。

## 実行コスト契約

- Stage 0 diagnostic variant: 1
- reporting fold: 5（trained fold 0）
- HMM well-run / model config / booster: `0 / 0 / 0`
- parent/control再学習・再生成: 0
- Kaggle CPU、GPU/TPU/internet off
- Stage 1予約: 全gate PASSかつ別承認時だけ 1 variant / 773 HMM well-runs

## 実装・検証

```bash
.venv/bin/python -m py_compile <exp343 compact train/inference.py> <exp343 test.py>
.venv/bin/ruff check <exp343 sources and test> --select F821,F401,F841,E722,E501
.venv/bin/pytest -q tests/test_exp343_acf_effective_sample_likelihood_tempering_audit.py
# 7 passed
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test <compact source.py>
make validate-exp EXP=exp343_acf_effective_sample_likelihood_tempering_audit
# strict validation passed
```

## 再現性メモ

- ACF/tauはRNGなし。well、window、run、lag、fold順を固定する。
- gzipはmtime=0で保存し、raw/decompressed/content SHAを分離する。
- Stage 0実行時はknown residual、lag ACF、tau schedule、fold prior、stability、
  fold metrics、scientific contract、gate、input/well manifest、summaryを保存する。
- Kaggle version 1 push時のimplementation SHA（完了後のstatus/test記録更新前）:
  - config: `a4fcb7432c4b8b84cd5f6074af724a72470ac3f61395999160bc1d0e92b0c534`
  - train source: `12ca5434f7fd9ab099d3db7118ed0c64eca2099cef7f17dc8e66d61c620f9989`
  - inference source: `442dde21e9b40bcb9eca6137f1c2c5385611e03bcbe40c893fc3ab37f46a139d`
  - test: `05709fcc8dda90834572cf3f56a533560cff444cbcb12b021d8e2848235b294f`
- compact/正規Notebookはtrain 21 cells、inference 7 cellsでcell source一致を確認した。
- fixed-input diagnosticであり、prediction/submission deterministic anchorではない。

## Repository full test

```bash
.venv/bin/pytest -q
# 683 passed, 5 skipped, 2 failed
```

- 失敗2件はいずれも既存exp296の完了後configと旧test期待の不一致。
  `experiment.status=completed_train_side_guard_failed_closed`に対し旧testが
  `kaggle_cpu_*`を要求し、`execution.run_variant=false`に対し旧testがpush approval
  guardを先に期待している。exp343専用test 7件はPASSしており、この実装ではexp296を変更しない。

## Stage 0後も未実施

- Stage 1実装・773 HMM well-runs
- inference / submission
- lag / support / clip / temperature / downsampling rescue

## exp320着眼点との関係

- 旧exp320は閉鎖履歴のまま維持し、exp343をType Well群非依存の独立後継として扱う。
- exp311/313/320のgroup prior、group AR(1)、group-label transferは入力しない。
- exp343自身もStage 0固定gate FAILで閉じ、旧branchをreopenしない。

## 次のアクション

固定gate FAILのため同一契約の再実行やStage 1へ進まない。
同familyの救済実験も追加せず、独立候補の優先順位を維持する。

## 2026-07-23 Stage 0 Kaggle CPU実行承認

- ユーザーの`実行してください`をStage 0 Kaggle CPU package/push/runの承認として記録した。
- 実行対象はdeterministic diagnostic 1、5 reporting folds。
- HMM well-run / model config / trained fold / booster / 親control再実行は
  `0 / 0 / 0 / 0 / 0`。
- `execution.run_stage_0=true`、`kaggle_push_approved=true`、
  `train_run_on_push=true`へ更新した。
- GPU/TPU/internetはoff。Stage 1、inference、submissionは未承認のまま。

## 2026-07-23 Stage 0 Kaggle push v1準備

- 初回packageはid
  `kentookumura/exp343-acf-effective-sample-likelihood-tempering-audit-train`、
  title `exp343 acf effective sample likelihood tempering audit train`、
  private CPU、internet/GPU/TPU off、run-on-push trueで生成した。
- local / loose package / bootstrap ZIP内configのbyte一致、exp226 kernel source、
  competition sourceを確認した。config SHAは
  `a4fcb7432c4b8b84cd5f6074af724a72470ac3f61395999160bc1d0e92b0c534`。
- 初回pushはKaggle `SaveKernel 400 Bad Request`で科学実行前に停止した。
  60文字のid末尾slug/titleがKaggle kernel slug上限を超えた可能性を確認した。
  同kernelのpullは403で、Kaggle側に利用可能なnotebookが作成されていない。
- 科学式・gate・入力を変えず、既存exp342/350と同じ短縮canonical方針で
  `kentookumura/exp343-acf-effective-sample-tempering-train` /
  `exp343 acf effective sample tempering train`へ再prepareする。
- 短縮canonical packageはmetadata SHA
  `5d7f5323a75694d545c488bb389bc8dbeedb960ea4441c2be12b7da07e7df914`、
  push notebook SHA
  `90afc0b99bf0368b63ff40f5782613f2eada352dacb02ec5506bb96a0c21de45`。
  bootstrap ZIPは18 support filesで、local/loose/ZIP内configのbyte一致を再確認した。
- 短縮canonicalへのpushは成功し、Kaggle private CPU version 1、
  id_no `128358348`として実行を開始した。Kaggle pull metadataでもinternet/GPU/TPU off、
  exp226 kernel source、competition sourceを確認した。
- 初回status確認は`KernelWorkerStatus.RUNNING`。CLI logsは実行中仕様どおり空であり、
  再pushやslug変更は行わず同じkernel idを監視する。

## 2026-07-23 Stage 0 Kaggle CPU version 1完了

- kernel `kentookumura/exp343-acf-effective-sample-tempering-train`、
  id_no `128358348`、version 1が`KernelWorkerStatus.COMPLETE`になった。
- 実行時間は`273.66704466799996 sec`。private Kaggle CPU、
  internet/GPU/TPU offをpull metadataでも再確認した。
- 実行量はdiagnostic 1 / reporting folds 5。
  HMM well-run / model config / trained fold / booster / 親control再実行は
  `0 / 0 / 0 / 0 / 0`。
- expected/actual wellsは`773 / 773`でPASSした。
- joint-evaluableは`295 / 773 = 0.3816300129366106`で、固定下限0.90をFAILした。
- fallbackは`478 / 773 = 0.6183699870633894`で、固定上限0.10をFAILした。
- full/last-512 Spearmanは`tau_eff`がほぼ定数4となりundefined、固定下限0.70をFAILした。
- median absolute log ratioは0.0、pooled median tauはfull/tailとも4.0、
  fold median tau ratioは両windowとも1.0でPASSしたが、clip由来の値である。
- stable foldは`0 / 5`。fold別joint-evaluable率は`0.357143--0.402597`、
  fallback率は`0.597403--0.642857`だった。
- upper-clip率はfull `0.9974126778783958`、tail `1.0`で固定上限0.25をFAILした。
- outer-train fold raw tau中央値はfull
  `9.7714364--10.0399628`、tail`24.2582863--25.1728468`だった。
  `[1,4]` clipがwell差を潰しており、Spearman undefinedとlog ratio 0.0を
  安定性の証拠とは解釈しない。
- decisionは`stage_0_failed_close_without_rescue`。
  `stage_1_eligible=false`で、Stage 1、prediction、inference、submissionは
  未実装・未実施のままbranchを閉じた。

## version 1成果物とSHA

- scientific contract content SHA:
  `59faf92ce0130ec025fb5d244dfa35170aa338b56e77e87baba593fa071177a0`
- known-prefix residual content SHA:
  `58b9b269ab2374270463b000edb2a671197e5514dbb1dbc664cabb941cc12cea`
- ACF lag content SHA:
  `2383489b6d6feefeda71c2ae4a35122a001145a7d175c06b2a78622eecd87512`
- tau schedule content SHA:
  `42391a82f4228ecae79755050b1b589732ba8e7d319ca5f7d159aeeb8b3d7022`
- fold priors / stability / fold metrics / well manifest content SHA:
  `3aad8f6ff19db33e6b318fd6c1184d69ea16725e7f8c45ea3794e57df938e200` /
  `1498f587c511bbfa35aba80c8358fa42ff56979b8ba42d8ce3c3a19e45475940` /
  `9b77ba5df4fe697833037edab3119a2f6770db81a3f529e1d75b41c00acfd307` /
  `8612ef9b445d3bbc8384aa697add3e439b266cf5611a40e65e1a7d162de78ae2`
- summary、gate、fold metrics、fold priors、stability readoutなど小さい出力だけを
  `/tmp/exp343-output-v1`へ選択取得し、summaryとmetricsのbyte一致、decision、
  実行量0 guardを検証した。
- 大きいknown residual、ACF、tau schedule archiveは、Kaggle logsとcontent SHAで
  判定可能なため取得していない。
- local configは完了状態へ更新し、`kaggle_push_approved=false`、
  `run_stage_0=false`、`train_run_on_push=false`として偶発的な再実行を防止した。
- 完了記録更新後のlocal SHA:
  - config: `b6487539bee38ba4307c382dc875437792ddbb7518688a4b12e47c65b9e72386`
  - contract test: `fe550e8f8f4850e707079a5beb1dd35858d66994db73e969e2c358d1e75dc9db`
  - metrics: `a6f3d3f5c0f184fd6a4db3a8d44999ffd3b923782932f1d3eb9618ee6a5bd2c6`
  - result: `a87f44b77f5f55bff4a10ee31fc22bcf386409cfc26baecf97a285dc093a9c79`
