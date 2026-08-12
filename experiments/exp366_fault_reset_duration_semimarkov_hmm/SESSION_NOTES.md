# exp366_fault_reset_duration_semimarkov_hmm セッションノート

## 目的

target-free triggerとexplicit durationを持つfault/reset HMMの設計を確定する。

## 現在の状態

- Route: `pf_beam`
- 状態: `completed_stage0_gate_failed_closed`
- 優先度: 閉鎖
- CV / LB: なし
- compact self-contained Stage 0 train / fail-closed inference: 実装済み。
- 正規train Notebook: compact候補の採用を承認済み。
- Kaggle Stage 0: version 2でtechnical PASS / scientific FAIL。
- Stage 1 / inference / submission: 不適格・未実装・未実行。

## コマンドログ

### 2026-07-23 実行済み

```bash
make new-steering EXP=exp366_fault_reset_duration_semimarkov_hmm
make new-exp EXP=exp366_fault_reset_duration_semimarkov_hmm
```

### 2026-07-25 Stage 0実装

- ユーザーの`exp366を実装してください`を、設計済みStage 0だけの実装承認として記録した。
- Jupytext percent形式のcompact self-contained train候補とfail-closed inference候補を追加した。
- 既存の正規`*_train.ipynb` / `*_inference.ipynb` placeholderは明示的な採用承認がないため
  上書きしていない。
- 実行対象契約は1 diagnostic / 13 fixed branches / 5 reporting folds。
  semi-Markov HMM well-run 0、LightGBM config 0、trained fold 0、booster 0、
  parent control rerun 0。
- `execution.run_stage_0=false`、Kaggle package / push / run、Stage 1、inference、
  submissionはすべて無効のまま。
- 固定512行評価窓を採用し、durationが128/256のbranchはactive期間後にbaseへ戻す。
  これにより13候補の累積GR log emissionとtruth RMSEを同一窓で比較する。
- branch順は`base → |jump| → sign(-,+) → duration(128,256,512)`。
- raw GR changeはabsolute first differenceをprefix median / 1.4826 MADでz化し、
  GR change zとexp209 Gaussian emission surpriseのprefix q99.5 ANDをtriggerにする。
- trigger / branch path content SHA / score / evidence rank / foldをtruth前にfreezeし、
  gzip decompressed SHA再読込後だけtruthとhidden-like roleを読む。

### 2026-07-25 静的検証

```bash
.venv/bin/python -m py_compile \
  experiments/exp366_fault_reset_duration_semimarkov_hmm/exp366_fault_reset_duration_semimarkov_hmm_compact_selfcontained_train.py \
  experiments/exp366_fault_reset_duration_semimarkov_hmm/exp366_fault_reset_duration_semimarkov_hmm_compact_selfcontained_inference.py
.venv/bin/ruff check \
  experiments/exp366_fault_reset_duration_semimarkov_hmm/exp366_fault_reset_duration_semimarkov_hmm_compact_selfcontained_train.py \
  experiments/exp366_fault_reset_duration_semimarkov_hmm/exp366_fault_reset_duration_semimarkov_hmm_compact_selfcontained_inference.py \
  experiments/exp366_fault_reset_duration_semimarkov_hmm/tests/test_exp366_fault_reset_duration_semimarkov_hmm.py
.venv/bin/pytest -q experiments/exp366_fault_reset_duration_semimarkov_hmm/tests/test_exp366_fault_reset_duration_semimarkov_hmm.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb \
  experiments/exp366_fault_reset_duration_semimarkov_hmm/exp366_fault_reset_duration_semimarkov_hmm_compact_selfcontained_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb \
  experiments/exp366_fault_reset_duration_semimarkov_hmm/exp366_fault_reset_duration_semimarkov_hmm_compact_selfcontained_inference.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test \
  experiments/exp366_fault_reset_duration_semimarkov_hmm/exp366_fault_reset_duration_semimarkov_hmm_compact_selfcontained_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test \
  experiments/exp366_fault_reset_duration_semimarkov_hmm/exp366_fault_reset_duration_semimarkov_hmm_compact_selfcontained_inference.py
make validate-exp EXP=exp366_fault_reset_duration_semimarkov_hmm
```

- `py_compile`: PASS
- `ruff`: PASS
- 専用test: `11 passed`
- Jupytext conversion / `--test`: train / inferenceともPASS
- `validate-exp` strict: PASS
- `make validate-template`: PASS
- `make test`: exp366を含む`1052 passed / 7 skipped`、exp366専用11件は全PASS。
  リポジトリ全体では既存状態とtest期待値がずれているexp296の2件、exp393の1件だけFAIL
  (`3 failed`)。いずれもexp366外のexperiment status / execution flag期待値であり、
  exp366実装による失敗ではない。
- train sourceは1,699行 / 10章。親exp209にはcompact self-contained train sourceがないため、
  最も近いStage 0実装のexp364（1,697行 / 10章）と比較し、runtime/config、input、
  trigger/branch、freeze、late truth、metrics/gate、orchestrationの役割を欠かしていない。
- 同じexp内helper import、notebook上の`__file__`参照は0件。

### 2026-07-25 Kaggle Stage 0実行承認

- ユーザーの`実行してください`を、compact self-contained train候補の正規Notebook採用と、
  private Kaggle CPU Stage 0 package / push / run承認として
  `2026-07-25 13:17:43 JST`に記録した。
- 実行対象はdiagnostic variant 1、fixed branch 13、reporting fold 5。
- semi-Markov HMM well-run 0、LightGBM config 0、trained fold 0、booster 0、
  parent control rerun 0。GPU / internetは無効。
- Stage 1の773 semi-Markov HMM runs、inference、submissionは引き続き未実装・未承認。
- canonical kernelは
  `kentookumura/exp366-fault-reset-duration-semimarkov-hmm-train`
  (`exp366 fault reset duration semimarkov hmm train`)。slug / titleは各48文字で一致する。
- push前の`kaggle kernels pull`は403、続く自分のNotebook検索は`Not found`。
  canonical kernelは未作成と判断し、同slugの初回pushとして扱う。
- push対象packageは24 cell（bootstrap 1 + canonical 23）、埋め込みsupport fileは24件。
  zip内全fileがmanifestのbyte数 / SHA256と一致し、埋め込み`config.yaml`はloose fileと
  byte一致した。push時点のSHA256は以下。
  - executed config:
    `327375bc7a7945d41161d6c66fdec800abc58904475cce64287aa9d7ba3ae321`
  - canonical / compact train notebook:
    `ca2bfdf993d73cc6573c106ba197bfa0e142698815c3b0faede536751a80e64f`
  - packaged train notebook:
    `c63000f6f9fa9baf38ecdbe6165a38429bd499d090c1b90eb4efb6407b61c728`
  - kernel metadata:
    `2581971d35e832ecaa0e446142c4e643fdf1bab4b45e2d451699f7f366e61ced`

### 2026-07-25 Kaggle version 1 fail-closed

- kernel version 1 / `id_no=128543224`をprivate CPU、internet offでpushした。
- bootstrap 24 filesは正常に展開されたが、約61秒後、学習・trigger生成・truth join前の
  raw well identity guardで停止した。
- 原因は、親実験群がraw well identityに使用するcolumn-aware logical SHA契約
  （column名 + dtype/bytesまたは文字列行）に対し、exp366だけがCSV serialization SHAを
  適用していた実装誤り。期待値`bbb687a1...b32`やraw inputの変更ではない。
- raw identity専用に親契約と同じ`logical_dataframe_sha256`を追加し、artifactの既存CSV
  content SHA契約には影響させない。mismatch時はexpected / actualも表示する。
- 科学契約、実行対象数、入力、gateに変更はない。同じcanonical slugのversion 2として
  packageを再生成し、再実行する。
- version 2 push対象は再び24 cell / support 24件で全manifest SHAとconfig byte一致。
  修正後SHA256はsource
  `6b75a8cd4d910b1133ffa16ffe3e9c00d41e43afe9070eb268d149bf36c26ff0`、
  canonical notebook
  `df2c548cd21ba473b45ced22d14dc53602c371849f56a26b0d609013a9a00133`、
  package
  `57b1579db097342a42c57a93d17c645d04a68d1ebbaf4a31246cdc1c8278ecba`。
- 修正後の専用testは`12 passed`、ruff / py_compile / Jupytext `--test` /
  `validate-exp`はPASS。

### 2026-07-25 Kaggle version 2完了

- 同じcanonical kernel version 2 / `id_no=128543224`をprivate CPU、
  GPU/internet offで実行し、状態`COMPLETE`。Stage 0本体`666.798832 sec`、
  Kaggle jobは約`695.920789 sec`。
- `3,783,989 rows / 773 wells`、eligible trigger rows `3,389,090`を評価した。
  発火は`40 events / 30 wells`、branch ledgerは`40 × 13 = 520 rows`。
- technical gateはPASS。freeze前truth / hidden-like role readは`0 / 0`、
  freeze後は`5,090,197 / 773`。semi-Markov HMM well-run、LightGBM config、
  trained fold、booster、parent control rerun、GPUはすべて0。
- overall:
  - trigger row fraction `0.0000118026`（閾値`[0.001,0.10]`、FAIL）
  - trigger bad-event AUC `0.5000036`（`>=0.60`、FAIL）
  - circular AUC `0.5000002`、差`0.0000034`（`>=0.05`、FAIL）
  - alternative within-10 coverage `0.90`（`>=0.60`、PASS）
  - evidence MRR gain vs base-first `-0.1233564`（`>=0.01`、FAIL）
  - base / selected / oracle RMSE
    `17.892031 / 18.897338 / 12.500137 ft`
  - selected gain vs base `-1.005307 ft`
  - passing folds `0 / 5`（`>=4 / 5`、FAIL）
- hidden-like spatial / typewell-purged selected gainは
  `-0.344537 / -0.666357 ft`で両方FAIL。
- decision:
  `stage0_failed_close_without_semimarkov_hmm`。
  Stage 1は不適格、inference / submissionへ進まない。
- 小さいinput manifest、branch ledger、event readout、scope/fold metrics、
  gate report、summaryだけを選択取得した。取得した全fileはKaggle summary内の
  raw / decompressed SHAと一致。126.8 MB trigger readoutと58.7 MB trigger ledgerは
  archive全量を取得せずmanifest SHAだけを記録した。
- 完了記録後は`execution.run_stage_0=false`と`train_run_on_push=false`へ戻し、
  ローカルKaggle packageも`run_on_push=false`で再生成した。埋め込みconfigはloose fileと
  byte一致し、24 support filesのmanifestも全件一致。
  - closed config:
    `71d080d2a0db728385875133161827128e2dd8663d74f5ea10310c6194f1da63`
  - closed package:
    `a28a30756a8f2c9d79d393c1b62caf071cab936ac71e7ae5bd36762b0352c4d8`
  - closed metadata:
    `df1b9ff0a4aecf892a13066363ca79137d497b415e5e0f5439783adf280cb696`
- version 2のKaggle logとpull済みmetadataは`kaggle/output/train_v2/`へ保存した。
  log SHAは`10b0e6f0d761458dac10472d2f09635a47037c44375dfeeddc8d7fc6dee1986a`、
  metadata SHAは`1393ab846579f67fed72b50e500a863ec29c46da5f671e6dfc525d258d6cebb7`。

## 変更点

- trigger、4 jump、3 duration、commit margin、refractoryを固定した。
- Stage 0をtrigger evidenceとbranch coverageのAND gateにした。
- Stage 1は1 variant / 5 folds / 773 HMM runs / booster 0 / control rerun 0。

## 再現性メモ

- seed policy: RNGなし、well / row / branch順を固定。
- stochastic components: なし。
- CPU/GPU: Kaggle CPU single worker。Stage 0本体`666.798832 sec`、GPU 0。
- SHA: trigger ledger、branch path/score、selection、foldをtruth join前にfreeze。
- contract:
  `f59ad2d8ac0b084c23461805c4f393c6938cd17e953c3a922fd4fe531905c604`
- trigger ledger decompressed:
  `5f2bce9e53245b0dd2e19a30364ed0504cc52d1f9f759f7bd77788d9b9ff9f51`
- branch ledger decompressed:
  `e4f9fa04318fcf1856d370334e272b62f83e22d01da2c2b82b0be4aa3f913800`
- summary:
  `cc048adc957ab8e71148c3dd68c3767157b9fdaba754c40c874f71643d99e7e6`
- prediction / submissionはStage 1不適格のため未生成。

## 次のアクション

1. 固定Stage 0 scientific FAILとしてbranchを閉じる。
2. threshold、AND条件、refractory、jump、duration、margin、negative controlを救済しない。
3. 独立した新しい識別根拠なしにStage 1、inference、submissionを行わない。
