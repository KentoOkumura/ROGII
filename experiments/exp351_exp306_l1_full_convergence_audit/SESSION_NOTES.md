# exp351_exp306_l1_full_convergence_audit セッションノート

## 目的

exp306 Stage 0で唯一full-eligibleになったL1固定設定を、truthや科学scoreを使わず全773 wells / 1,546 seriesでtechnical auditする設計を確定する。

## 現在の状態

- Route: `pf_beam`
- 状態: Kaggle CPU full audit version 1完了、technical FAILでclosed
- 親: `exp306_robust_rts_l1_convergence_calibration_audit`
- active branch: L1 1 branch
- 予定実行量: 1,546 L1 solver series-runs
- model / LightGBM / fold / HMM / PF / Beam / booster / control再実行 / GPU: すべて0
- CV / LB / prediction / submission: なし / なし / なし / なし

## 2026-07-23 設計

ユーザーの依頼により、exp306内の後続stageではなく新規実験ディレクトリとしてexp351を作成した。評価条件が64-well Stage 0から773-well full auditへ変わる一方、solver仮説・設定は変更しない。

実行したコマンド:

```bash
make new-steering EXP=exp351_exp306_l1_full_convergence_audit
make new-exp EXP=exp351_exp306_l1_full_convergence_audit
```

- `task` executableがないため、`kaggle-review-exp`で許可されたMakefile同等手順を使った。
- steeringをexperiment scaffoldより先に作成した。
- 親コードはコピーせず、templateからdesign-only scaffoldを作成した。
- `docs/06_reproducibility.md`とexp306 requirements/design/metrics/source SHA utilityを確認した。

## 固定した設計

- L1 branchは`l1_iter2000_rho1_tol1e4`だけ。
- exp306のobjective、lambda式、rho 1、max iterations 2000、abs/rel tolerance 1e-4を固定。
- 全773 wells x horizontal/typewell = 1,546 seriesを1回だけ実行。
- Stage 0 control、full rerun、parity rerunは0。full run内の親sample subsetをSHA比較する。
- full gateは1,546/1,546 convergence/technical、finite/order、fallback 0、error 0、runtime 8.5時間以内、parent/cross-run SHA完全一致のAND。
- RTS、grid、truth/scientific score、HMM/PF/Beam、prediction、inference、submissionは除外。
- PASSしてもexp304 selected SWTを変更せず、別科学評価の資格だけを得る。

## 再現性メモ

- seed policy: RNGなし、stable well/series sort。
- stochastic components: なし。
- CPU/GPU runtime: Kaggle CPU、single worker、BLAS threads 1、GPU/TPU/internet offでversion 1実行済み。
- parent kernel: `kentookumura/exp306-rts-l1-convergence-calibration-audit-train` version 1、id_no `128231380`。
- raw identity SHA: `bbb687a1998092578583ce259309b49031d095bde57cbb26c0ab8808d2379b32`。
- parent 64-well input/L1 output/L1 status content SHA: `3eb28b189cd77b3d48f9745dcf49e2f8587551abfed8dcdc674101f5b1f406c8` / `186d9682147563fe4cf1609a067004460f2e3b250a8026b7611dd712db0cbf42` / `7b3a292ff99f8cb12abfd5917893f7d0f7b3a99109d312c0b66ae3b5966edd89`。
- parent 8-well output/status/iteration parity SHA: `d0b4a19788e7a13df9603b566eca35dba998b0063fb7f9e25e64b2b0b4fedec0` / `3535381302938a7a467f78d0b2fe45bb8ca587e1db9091a3a730e6262197724f` / `6488af59de4ad6ac28eb5d68b3407f5a45ea67bb42a23b051834ae3ca834b036`。
- full output SHA: input/output/statusのschema/content、gzip raw/decompressedをversion 1結果として記録済み。
- model/prediction/submission SHA: 非該当。
- deterministic anchor: submission anchorではなく、solver technical reproducibilityだけを対象とする。

## 2026-07-23 実装

ユーザーの`exp351を実装してください`をcompact self-contained候補とsynthetic contract testsの実装承認として扱った。正規Notebook採用、Kaggle package/push/run、科学評価は承認範囲に含めていない。

実装内容:

- exp306 compact self-contained trainからhorizontal/typewell allowlist、common GR preparation、L1 `max2000/rho1/tol1e-4`、dataframe/SHA utilityを抽出した。
- RTS、Stage 0 branch selection、truth/scientific score、HMM/PF/Beam、prediction、submission pathは持ち込んでいない。
- 親kernel version 1のcontract/gate/summary/sample/input/output/status/parityをsolver前にfile/content SHAで検証する。
- 全773 wellsを1回だけprepare/solveし、1,546 status coverage、all convergence/technical、finite/order、fallback/error 0、8.5時間をAND gateにした。
- full runから親64-well sampleと先頭8-well parityを抽出し、input/output/status/iteration SHAをexact比較する。再solverは行わない。
- input/output/status gzipは`mtime=0`で保存し、raw/decompressed/dataframe content SHAを記録する。
- 別名compact inferenceは常にfail-closeする。

親compact比較:

- exp306 compact train: 11章、1,400行。
- exp351 compact train: 10章、1,461行。
- exp351はparent anchor guards、target-free preparation、L1 kernel、full execution、cross-run parity、full gate、生成物保存、setup/run guardをNotebook上で追える。
- `__file__`と同一exp helper importはない。

実行した検証:

```bash
.venv/bin/pytest -q tests/test_exp351_exp306_l1_full_convergence_audit.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb <compact train/inference.py>
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test <compact train/inference.py>
.venv/bin/python -m py_compile <compact train/inference.py>
.venv/bin/ruff check <compact train/inference.py> --select F821
make validate-exp EXP=exp351_exp306_l1_full_convergence_audit
make validate-template
.venv/bin/pytest -q tests/test_exp351_exp306_l1_full_convergence_audit.py tests/test_kaggle_notebooks.py tests/test_scaffold.py
```

検証結果:

- 専用contract tests: `11 passed`。
- exp306とのsynthetic preparation/L1 output bitwise parity: PASS。
- 関連tests: `22 passed`。
- Jupytext round-trip、py_compile、Ruff F821、strict experiment validation、template validation: PASS。
- 全suite: `668 passed, 5 skipped, 2 failed`。2件はいずれも既存`exp296_exp223_self_gr_known_tvt_support_gate`の完了済みconfigに対して古いrun前状態を期待する既知の不一致で、exp351専用/関連testsには失敗なし。
- 正規Notebookは上書きせず、別名compact `.py` / `.ipynb`だけを生成した。
- 実データ/full audit/Kaggle実行は行っていない。

## 未実施

- scientific score、HMM/PF/Beam、inference、submission

## 次のアクション

1. exp351をtechnical negativeとして閉じる。
2. iteration/tolerance/lambda/rho/grid救済、scientific score、inference、submissionへ進まない。

## 2026-07-23 Kaggle CPU full audit実行承認

ユーザーの`実行してください`により、正規Notebook採用とKaggle CPU full auditのpackage/push/run 1回を承認済みとした。

push前実行量:

- active branch: L1 `l1_iter2000_rho1_tol1e4` 1。
- wells / series: 773 / 1,546。
- L1 solver series-runs: 1,546。
- Stage 0 control / full rerun / parity rerun: 0 / 0 / 0。
- model / LightGBM config / trained fold / HMM / PF / Beam / booster / parent control再実行 / GPU: すべて0。
- runtime: Kaggle CPU、internet off、single worker、BLAS threads 1、hard gate 30,600 sec。
- parent kernel: `kentookumura/exp306-rts-l1-convergence-calibration-audit-train` version 1 / id_no `128231380`をkernel sourceにし、固定artifact SHA不一致ならsolver前に停止する。
- canonical kernel: `kentookumura/exp351-exp306-l1-full-convergence-audit-train`。
- canonical title: `exp351 exp306 l1 full convergence audit train`。
- pre-push pull: `403 Forbidden`。既存versionは確認できなかったため、別slugを作らずcanonical IDへ初回pushする。

正規Notebook採用:

- compact self-contained train/inferenceから正規`*_train.ipynb` / `*_inference.ipynb`をJupytext生成した。
- train 21 cells、inference 9 cellsについて、compact候補と正規Notebookのcell type/source完全一致を確認した。
- 採用後の専用/Notebook/scaffold testsは`22 passed`、strict experiment validationとtemplate validationもPASSした。

Kaggle package確認:

- `make prepare-kaggle-notebooks EXP=exp351_exp306_l1_full_convergence_audit EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp351-exp306-l1-full-convergence-audit-train --title 'exp351 exp306 l1 full convergence audit train' --run-on-push --strict"`を実行した。
- metadataはcanonical id/title一致、private、CPU、GPU/TPU/internet off、run_on_push true。
- competition sourceは`rogii-wellbore-geology-prediction`、kernel sourceは親`kentookumura/exp306-rts-l1-convergence-calibration-audit-train`のみ。
- bootstrap内configは正のconfigとbyte-identicalで、canonical adopted / push approved / run full L1がすべてtrue。
- bootstrap内L1は1 branch、max2000、rho1、abs/rel tol `1e-4`。実行量は1,546 series-runs、全model/control/GPU count 0。
- bootstrap内parent version / id_no / Stage 0 L1 output SHAは`1` / `128231380` / `186d9682...0cbf42`。
- bootstrap ZIP SHA256: `9fa2a70df33f219a0f240f8512ea5e4b5a2b9fc4765aadc47892d82b02b94975`。
- package Notebook / metadata / config SHA256: `8ea85a4c...6a1160` / `b9406463...c95d` / `a51c7764...b68f`。

Kaggle push:

- `make push-kaggle-train EXP=exp351_exp306_l1_full_convergence_audit`でcanonical kernel version 1をpushした。
- kernel: `kentookumura/exp351-exp306-l1-full-convergence-audit-train`
- version / id_no: `1` / `128354027`
- URL: `https://www.kaggle.com/code/kentookumura/exp351-exp306-l1-full-convergence-audit-train`
- pulled metadataはprivate、CPU、GPU/TPU/internet off、親kernel source一致。
- docker image: `gcr.io/kaggle-images/python@sha256:dafd4ce5668bbf1ad422e4c109e0f18c9623c3a7c7f48b0235f13142755c40b9`
- 状態: 実行中。別slugやversionを追加せず、同じversion 1を監視する。

## 2026-07-23 Kaggle CPU full audit version 1結果

Kaggle logsで約355秒時点のNotebook完了を確認した。summaryが`full_technical_fail_closed`だったため、full output archive全体は取得せず、原因判定に必要なgate、cross-run parity、parent anchor、summary、solver statusだけを`/tmp/exp351-v1-diagnostic`へ選択取得した。

結果:

- kernel version / id_no: `1` / `128354027`
- wells / series: `773 / 1,546`
- convergence / technical PASS: `1,537 / 1,546`（`0.9941785252`）
- typewell: `773/773` PASS、iterations mean/max `518.485123 / 1124`
- horizontal: `764/773` PASS、iterations mean/max `790.159120 / 2000`
- 未収束: horizontal 9 series。全件max iteration `2000`到達、exception/error/fallbackなし。
- failed wells / rows: `5138a660/4829`、`53f23031/10200`、`591cc951/6669`、`5def1ce5/7997`、`81bf5923/6386`、`ae069086/6741`、`b37fd114/9786`、`c59b6c4a/7460`、`d924e971/5308`。
- preparation / solver / audit runtime: `12.829504 / 222.476799 / 329.250058 sec`
- runtime gate: `329.250058 <= 30,600 sec` PASS
- parent artifact anchors: PASS
- raw identity 773 wells: PASS
- coverage/duplicate、finite input/output、length/order、fallback 0、error 0: PASS
- 64-well input/output/status exact SHA: PASS
- 8-well output/status/iteration exact SHA: PASS
- truth/scientific score loaded、prediction、submission: `false / null / null`
- FAIL gate: `all_converged`と従属する`all_technical_pass`のみ。

生成物SHA:

- raw identity content/raw: `bbb687a1...b32` / `f0dec823...e422`
- input content/raw/decompressed: `96ed2ebc...c6b0` / `1852b9ab...2624` / `fa71faa3...92c3`
- output content/raw/decompressed: `45d51d60...3a61` / `64b422a3...0eef` / `e005aec9...12d2`
- status content/raw/decompressed: `d1968a08...3e66` / `d85af923...c155` / `b83702d3...281d`
- gate/parity/parent-anchor/summary file SHA: `6f10c757...fade` / `b745c15a...34eb` / `75918c84...beee` / `7151d6ac...40c5`
- downloaded Kaggle log SHA: `a6541d12341f3ef646895acb2694716dd1a1a1ccb2f73b73d9dc2f0f238d019a`
- 選択取得したstatusのraw/decompressed SHAはsummary記録と一致した。

判断:

- 事前固定した`1,546/1,546` AND gateを満たさないためtechnical negative。
- 64-well Stage 0のfull feasibility外挿は9本の長いhorizontal seriesを捕捉できなかった。
- failure policyどおりiteration、tolerance、lambda、rho、adaptive rho、gridで救済しない。
- `execution.run_full_l1=false`へ戻し、追加version、科学評価、exp304 selected SWT変更、HMM/PF/Beam、inference、submissionは行わない。
- 本結果だけを根拠とする新規L1 solver救済backlogは追加しない。
