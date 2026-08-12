# exp297_prefix_calibrated_latent_registration_gr_evidence セッションノート

## 目的

exp293 deployable12を固定したまま、prefix-calibrated Type Well/horizontal GR evidenceがtruth-good
candidateへ確率質量を置けるかを、target-free freeze後のexpected candidate SSEで監査するStage 2を実装する。

## 現在の状態

- Route: `pf_beam`
- 状態: Kaggle private CPU version 2完了、固定判定`FAIL_STOP_NO_STAGE4`
- active audit variant / LightGBM config / evaluation fold / trained fold / booster: `1 / 0 / 5 / 0 / 0`
- PF/Beam well run / control再学習: `0 / 0`
- CV/LB/submission: fixed 5-fold readout完了・LB/submission対象外

## コマンドログ

2026-07-19に以下を実施した。

```bash
make new-steering EXP=exp297_prefix_calibrated_latent_registration_gr_evidence
make new-exp EXP=exp297_prefix_calibrated_latent_registration_gr_evidence
.venv/bin/ruff format <exp297 train/inference source and test>
.venv/bin/ruff check --select F821,E9 <exp297 train/inference source and test>
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb <compact train source>
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb <compact inference source>
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test <compact train/inference source>
.venv/bin/pytest -q experiments/exp297_prefix_calibrated_latent_registration_gr_evidence/tests/test_exp297_prefix_calibrated_latent_registration_gr_evidence.py
```

結果は`10 passed`。構文チェックと`ruff --select F821,E9`もPASSした。

exp293 full-bank loaderのtarget-free preflightも試したが、ローカル3.8 GiB環境では3,783,989行
assignmentの論理SHA計算中にプロセスが終了した。suffix truth、score、posterior、readoutは読み書きしておらず、
実データ検証結果としては扱わない。Kaggle開始時のstrict loader検証は省略しない。

## 変更点

- 別名compact self-contained train source/notebookを実装した。canonical notebookは変更していない。
- 別名compact inferenceをfail closedで実装した。
- exp293 candidate matrix/manifest/block assignmentをraw/decompressed/logical SHA固定で読む。
- calibration、21-state registration、3 component posterior、unreliable safe fallback、stable shuffleを実装した。
- target-free artifactをclose・SHA再検証した後だけtruth loaderを開く。
- expected SSE readoutと固定Stage-2 PASS/FAIL判定を実装した。
- synthetic contract testsを10件追加した。

## 再現性メモ

- seed policy: realはRNGなし。shuffleだけ`SHA256(experiment, seed, well)`由来のlocal RNG。
- stochastic components: well内finite GR circular rotation negative controlのみ。
- parallelism: single process、global RNGなし。
- CPU/GPU runtime: Kaggle private CPU予定、GPU/AMP/internet off。未実行。
- input SHA: exp293 raw/decompressed/logical SHAとhidden-like SHAをconfig固定。raw horizontalのtruth-bearing
  file SHAはfreeze前に読まず、target-safe selected-frame logical/schema SHAを先に固定する。
- posterior SHA: joint NPY、candidate/registration Parquet、block/calibration/input artifactをfreeze manifestへ記録予定。
- model/prediction/submission SHA: model、TVT prediction、submissionを生成しないため対象外。
- deterministic anchor: fixed-input diagnosticでありsubmission anchorではない。
- kernel id/version/rerun: 未実行。

## 実行承認前に未承認・未実施だった範囲

- canonical notebook採用
- Kaggle package/bootstrap生成と照合
- Kaggle train kernel push/run
- raw-test inference、submission

`execution.kaggle_push_approved: false`のため、compact trainをNotebook runtimeで直接実行しても開始前に停止する。

## 実行承認前に予定していた次のアクション

ユーザーの別承認後にcanonical train notebookへ採用し、1 variant × 0 model × 5 evaluation folds × 0 boosterの
Kaggle private CPU auditを1回だけ実行する。PASS時だけStage 3へ進み、FAIL時は停止する。

## 2026-07-19 Kaggle実行承認

- ユーザーの「実行してください」を、compact trainのcanonical採用と固定済みKaggle private CPU audit
  1回の承認として反映した。
- push対象はactive audit 1、LightGBM config 0、evaluation fold 5、trained fold 0、booster 0、
  HMM/PF/Beam well-run 0、control/parent再学習なし、CPU、GPU/TPU/internet off。
- inference、submission、candidate追加削除、registration/component/weight/prior/threshold変更、
  FAIL後のStage 4自動分岐は承認範囲外のまま維持する。
- credential preflightはAPI token未設定、OAuth credentialとlegacy credentialはOK。Kaggle CLI実行に使える。
- 親compactとexp297 compactを比較し、双方とも8章構成。行数は`1963 / 2044`で、exp297は入力・posterior・
  truth freeze・判定の上位ロジックをNotebookセルに展開している。同一実験helper importと`__file__`依存はない。
- compact train notebookをcanonical trainへbyte-identicalで採用した。17 cells、code output 0、
  canonical/compact SHAは`8aa2fbfe1b8e17e7328b6df2fabf1224d75fd6fc82e6d445d58f0c8e5cca342c`。
- canonical kernelは`kentookumura/exp297-latent-registration-gr-audit-train`、titleは
  `exp297 latent registration gr audit train`に固定した。事前の`kernels list --search exp297 --mine`はNot found。
- push前検証はproject/experiment strict validation、Jupytext train/inference、Ruff、exp292/293/297関連tests
  `32/32`がPASSした。
- 親kernelのmetadata取得に成功した。exp293はid_no `127891171`、exp115はid_no `124519917`で、
  いずれもprivate CPU、GPU/TPU/internet off。
- strict packageはbootstrap 1 + canonical 17 = 18 cells、code cells 9、cell output 0。private CPU、
  GPU/TPU/internet off、run-on-push true、competition source 1件、kernel source 2件を確認した。
- executed config SHA: `80169ce8fd80f8f8780ddbd62e5cb31c73d4beaf10afae2d47466c44ddc56ccd`。
- compact train source SHA: `3ed99e4a9c3826f00b3f3caab28d390f7da96da8748446934533e2a6dfd9bd46`。
- canonical notebook SHA: `8aa2fbfe1b8e17e7328b6df2fabf1224d75fd6fc82e6d445d58f0c8e5cca342c`。
- packaged notebook SHA: `1d861e659b07c5f91b293b2d23f77af988a0e32618586db9a93ad50efccfbddc`。
- kernel metadata SHA: `d057ab637a7355b920a943f7977749095267d4d31387a6c1dbfce2b06cfa9728`。
- bootstrap ZIP SHA: `2da42821800aa9cf05f53e1c21ea202cd2d8e57cc364873c1d0c445dfef9c7cb`。
  loose/package/bootstrap内configとcompact train sourceはbytes一致し、packaged body 17 cellsもcanonicalと一致した。

### Kaggle version 1実行

- canonical kernel version 1をpushし、URLは
  `https://www.kaggle.com/code/kentookumura/exp297-latent-registration-gr-audit-train`。
- Kaggle側id_noは`127897451`。pullしたmetadataでprivate CPU、GPU/TPU/internet off、competition source 1件、
  kernel source 2件を再確認した。
- 初回statusは`RUNNING`。通常logsは実行中のため空であり、同じkernel IDを維持して監視する。

#### version 1停止とversion 2修正

- version 1はbootstrap後、exp293 block assignmentのkey logical SHA guardで停止した。実測SHAは
  `bd0e4e47d548de1fe52e323a2a789a6677983ecbaed70b9d0753e933e5071562`、期待値は
  `42ede2f53e28dc2ccb28f847e0bc23680d1121bb29e980016dee989dbdddfef7`。
- candidate matrix検証、raw horizontal/typewell読込、posterior生成、truth読込には到達していない。
- 原因はexp293のkeyがexp263 parquet由来の`well_row_idx int32 / outer_fold int8 / md_since float32`であるのに、
  exp297 loaderがCSV再読込時に`int64 / int64 / float64`へ拡張していたこと。値が同一でもhash契約はdtype名を含む。
- scientific contract、候補値、fold、block、registration、score、prior、閾値は変更せず、親の物理dtypeを
  明示固定するversion 2修正とした。同一canonical kernel IDへ再pushする。
- 修正後に実block assignment 3,783,989行を100,000行ずつstreaming hashし、key SHA
  `42ede2f53e28dc2ccb28f847e0bc23680d1121bb29e980016dee989dbdddfef7`とfull logical SHA
  `63f9a26a243ce3b1dd0cbec85c9674fd69a0768246220728ee9d54defba046e5`が親contractと完全一致した。
- version 2専用tests `10/10`、strict validation、Jupytext、Ruff、構文checkがPASS。
- version 2 config SHA: `d3f533d46b0fba143cf89bf3f025b8e7bd2e996ec0f616ce0f1235c6d4800205`。
- version 2 compact train source SHA: `5e186c62c16fd8217e30647fb04d0b99f68175e9d7ecf96769661f9432d4929f`。
- version 2 canonical notebook SHA: `4bb3914c66853abd6c980947334ef79fda91aa77f990218afae656863a7de5b9`。
- version 2 packaged notebook SHA: `cf78998e40ab9c46336e79699a3c966131a09cb8eb57d4204c22fe54bdbe9004`。
- version 2 bootstrap ZIP SHA: `d5f55a42d43a29466e69798ae074e1e01870f8da2beb33bd0f04a561e763d8d4`。
  packaged body、loose/package/bootstrap内config/sourceは再度bytes一致した。
- 同一canonical kernelへversion 2をpushし、run-on-pushで再実行を開始した。
- version 2はcandidate-bank guardを通過してwell単位posterior生成へ進み、最後に確認したstatusは`RUNNING`。
  Type Well範囲外stateの`All-NaN slice` warningは出たがfatal tracebackはなかった。
- ユーザー指示によりCodex側のCLI logs追跡だけを停止した。Kaggle kernel自体は停止していない。
  完了連絡後に同じkernel IDのlogs/summaryを回収し、Stage-2 PASS/FAILと実験記録を確定する。

## 2026-07-20 version 2完了と固定Stage 2判定

- ユーザーの完了連絡後、同一canonical kernelの通常logsとstatusを回収した。statusは`COMPLETE`、
  scientific summaryは`status=completed`、decisionは`FAIL_STOP_NO_STAGE4`。
- 3,783,989 rows / 773 wells / 105,818 block-controlを1,070.799801秒で完走した。
- primary H256 pooledはanchor RMSE `8.23833174548`、oracle RMSE `3.55282885137`に対し、
  real posterior expected RMSE `8.62004120371`、headroom recovery `-0.116475829913`。
  固定PASS閾値`>=0.35`をFAILした。
- H256 real fold recoveryはfold 0..4で
  `-0.089589355255 / -0.069221869041 / -0.181847545810 / -0.158430920411 / -0.092005453074`。
  5/5 foldsで負だった。
- H256 shuffle pooledはexpected RMSE `8.57158319474`、recovery `-0.101396928037`。
  shuffle fold recoveryは
  `-0.073010519871 / -0.060738813169 / -0.164502990477 / -0.130027501301 / -0.085741214772`。
  realはpooledでも5/5 foldsでもshuffleを下回った。
- H512 real recoveryは`-0.119000215155`。H256からの低下は`0.002524385242 <= 0.05`で
  continuity checkだけPASSしたが、recovery自体は負である。
- H256 subgroupは1000+がanchor/expected `9.04232410527 / 9.45966258461`、hidden-like spatialが
  `8.74810807355 / 9.03482025057`、typewell-purgedが`8.69413182489 / 8.97578719618`で、
  3面ともanchor非劣化をFAILした。
- calibrationは704/773 wells valid。69 wellsは`prefix_typewell_gr_std_below_minimum`、34 wellsはslope clip。
  valid residual scale中央値は`10.0`、p90は`12.64185672321`、prefix RMSE中央値は`9.69129190204`。
- H256 15,174 blocksのeligible-stateあり割合は`0.295044154475`。real reliable probabilityは
  平均`0.118509462442`、中央値`0.0`で、多くのblockがunreliable-safeへ退避した。
- truth accessはfreeze前`0`、freeze後`773`。target-free 8件、post-freeze truth 773 raw files +
  1 memory vector、post-freeze readout 4件のphase順序をmanifestで確認した。
- `kaggle kernels output`で`kaggle/output/train_v2`へ実ファイルを取得し、Kaggle SHA manifestに載る
  `/kaggle/working`の12ファイルを再計算して全一致した。
  - candidate content SHA: `29477141685662bae7417e788ec5dbe914c2220b7f4cc45ab01befa4e5e3b474`
  - truth content SHA: `b0a1bebf24ec925728c40a690147d1820b88d0bf3f403333d9452a79ef179c8d`
  - joint posterior file SHA: `2f4f443a93491e3dd1d5b87ac239d1802dcff1ddef5ad9584aedb982e5376060`
  - candidate posterior file SHA: `93b4e9cfe0eae25742cb0fbf90a40ff5f76267372cd25ecbcdac3549a77cf54a`
  - registration posterior file SHA: `e48d6b40878edc50734a5a83e0ee405adda00a7f5da2fa3780868c9b4384d0ca`
  - readout metrics file SHA: `ff7a0ec39c1f17a1f8e28d404cd07a7d2df3384dd86e7216fde35130b7cdcf37`
  - summary / Kaggle metrics SHA: `1e599b6b179940f6557dc739b64ee67e907ad12b4efc2addc0f1b9361a631d03`
- joint posterior shapeは`[105818, 12, 21]`で全finite。candidate / registration marginalは
  `1,269,816 / 2,222,178` rows。contract上のselected/corrected TVT predictionとsubmissionはabsent。
- `All-NaN slice` / `Mean of empty slice` warningはType Well範囲外やeligible stateなしblockで出た。
  unreliable-safe fallbackを経てsummary保存まで完走しており、fatal tracebackではない。

## 最終解釈と次のアクション

- 技術guardとtruth freezeは成立したが、real GR posteriorが全foldでanchorを悪化させ、matched shuffleよりも悪い。
  exp293 fixed12のoracle headroomを識別する今回の観測モデルを棄却する。
- `execution.one_run_authorization_consumed=true`、`kaggle_push_approved=false`へ戻した。
- 固定契約どおりStage 3、Stage 4、inference、submissionを閉じる。同じtruth上でregistration grid、
  component weight、prior、thresholdを救済調整せず、exp297由来の新規救済backlogも追加しない。
- 物理routeの次候補は独立設計済みのexp298 local-shape source監査とexp295 candidate-free SSM。
  exp297の自動分岐ではなく、それぞれ既存の事前guardと別承認に従う。
