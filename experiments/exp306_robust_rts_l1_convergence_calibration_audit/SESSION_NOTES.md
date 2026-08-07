# exp306_robust_rts_l1_convergence_calibration_audit セッションノート

## 目的

exp304で未収束だったrobust RTS / L1 trendを、truthや科学scoreを使わず、事前固定した最小solver変更だけで全series technical PASSへできるか監査する。

## 現在の状態

- Route: `pf_beam`
- 状態: Kaggle CPU Stage 0 version 1完了、L1のみfull-eligible、full audit未実装・未承認
- Stage 0 core: RTS A/B/L1各128、実績384 series-runs
- Stage 0 parity rerun: L1 8 wells x 2 series、実績16 series-runs
- Full: eligible branchごと1,546 series-runs、最大2 branches / 3,092
- model / LightGBM config / trained fold / HMM / PF / Beam / booster: `0 / 0 / 0 / 0 / 0 / 0 / 0`
- CV / LB / submission: なし / なし / なし

## コマンドログ

### 2026-07-21 設計

```bash
make new-steering EXP=exp306_robust_rts_l1_convergence_calibration_audit
make new-exp EXP=exp306_robust_rts_l1_convergence_calibration_audit
```

- `kaggle-review-exp`に従いsteeringを先に作成した。
- exp304 solver設定/actual failure、backlog契約、`docs/06_reproducibility.md`を確認した。
- requirements/design/tasklist、config、README、result、metricsをdesign-onlyとして確定した。
- scaffold Notebookは誤実行でmetrics/submissionを作らないdesign-only guardとする。
- solverロジック、Jupytext source、Kaggle package、push、runは作成・実行していない。

## 固定した設計

- Stage 0 sampleは`SHA256("exp306-stage0-v1|" + well_id)`順の先頭64 wells。
- RTS A=`32, 1e-6`。Aが128/128を満たさない場合だけB=`32, 1e-4`。
- L1はlambda/rho/tolerance固定、max ADMM 2000のみ。
- Stage 0先頭8 wellsのexact parityとbranch別8.5時間外挿を必須とする。
- fullはeligible branchを別runで監査し、各1,546/1,546収束を必須とする。
- truth/scientific scoreは読まず、exp304 selected SWTを変更しない。

## 再現性メモ

- seed policy: RNGなし、固定salt SHA256 sample。
- stochastic components: なし。
- CPU/GPU runtime: Kaggle CPU予定、GPU/TPU/internet off、固定single worker/BLAS thread、branch別8.5時間上限。未実行。
- Kaggle kernel id / version: 未作成。
- input SHA: raw well identity `bbb687a1...b32`、exp304 scientific contract `8822df96...064`を固定。
- feature/output/status SHA: 未生成。実行時はsample/prepared input/output/statusのcontent SHAを保存する。
- model/prediction/submission SHA: 非該当。

## 次のアクション

1. Stage 0のKaggle package/push/runを行う場合は、別途ユーザー承認を得る。
2. Stage 0でeligible branchとSHA/runtime evidenceが確定しても、full auditは別承認まで実装・実行しない。
3. exp306内では科学score、prediction、inference、submissionを生成しない。

## 2026-07-22 Kaggle CPU Stage 0実行承認

- ユーザーの`実行してください`を、Kaggle CPU Stage 0のpackage/push/run承認として記録した。
- 実行対象は固定SHA256順64 wellsのhorizontal/typewell各128 series。
- coreはRTS A 128 + L1 128 = 256 series-runs。RTS Aにtechnical FAILが1件でもある場合だけRTS B 128を追加し、最大384。
- provisional eligible branchだけ先頭8 wells x 2 seriesを再実行し、parityは最大32 series-runs。
- model / LightGBM config / trained fold / HMM / PF / Beam / booster / 親control再実行 / GPU: `0 / 0 / 0 / 0 / 0 / 0 / 0 / 0 / 0`。
- CPU、internet off、worker 1、BLAS thread 1。full audit、科学評価、inference、submissionは未承認のまま。
- canonical kernel: `kentookumura/exp306-rts-l1-convergence-calibration-audit-train`。56文字の実験名全体slugはKaggleの既知のtitle長制約を超えるため、`robust`だけを省き、calibration/auditを保持した49文字slugを採用する。
- `task` executableは環境に存在しなかったため、同等の`make prepare-kaggle-notebooks`へ切り替えた。
- package metadataはprivate、CPU、GPU/TPU/internet off、run-on-push、competition source 1件、親kernel source 1件を確認した。
- local/package/bootstrap ZIP内config SHAは`d55d723c8ee08fe91a329f1aee607cb56a792461a132d7a4a4f350c144d44c78`、train source SHAは`62b555ad6b42f72c53d01608a8ba2389523258296c89226a1af200260c337036`で一致した。
- package Notebook SHAは`36103a1e5b0e183549cb417858c7190adff6fe2b1324374ec39c2ec7194f4556`。
- bootstrap内でRTS A/B 32 iterations、L1 2000 iterations、Stage 0 true、full/scientific/inference/submission false、model/LightGBM/fold/booster 0を再確認した。
- Kaggle kernel version `1`、id_no `128231380`としてpush成功。pull metadataで同じcanonical id/title、private、CPU、internet off、competition/kernel sourceを再確認した。
- push直後は`KernelWorkerStatus.RUNNING`。CLI logsは空だが、実行中は空が既知挙動なので同じversion 1を監視し、再pushしない。

## 2026-07-22 Stage 0実装

ユーザーの`exp306を実装してください`を、steeringで事前登録したStage 0実装の明示承認として扱った。Kaggle package/push/run、full audit、科学評価への承認には拡張していない。

追加・更新:

- compact self-contained train source / Notebookとfail-closed inference source / Notebookを追加した。
- compact Notebookを正規train/inference Notebookへ採用し、compact/canonical SHA一致を確認した。
- raw horizontalは`MD/GR/TVT_input`だけを`usecols`で読み、frame guardで`TVT/truth/error/formation/MRR/top3/RMSE/score/prediction`を拒否する。
- exp304と同じGR missing policy、coordinate normalization、Student-t RTS kernel、L1 objective/lambda/rho/toleranceを持ち込み、変更点をRTS最大32 IRLSとL1最大2000 ADMMに限定した。
- RTS Aが128/128 technical PASSしない場合だけRTS Bを1回実行する。runtime FAILだけではBを実行しない。
- fixed-salt SHA256順64 wells、各branch 128 series、sample順先頭8 wellsのoutput/status/iteration exact SHA parity、branch別773-well runtime外挿を実装した。
- contract/sample/input/output/status/parity/gate/summaryを保存し、gzipはraw SHAとdecompressed SHAを分ける。truth/scientific scoreは生成物に含めない。
- full run flags、科学score、inference、submissionはcontractでfail-closeし、Stage 0のrun flagもfalseのままにした。

実装時点の実行量:

- Stage 0 core: RTS A 128 + L1 128 = 256 series-runs。
- RTS Aに1件でもtechnical FAILがある場合だけRTS B 128を追加し、最大384 series-runs。
- provisional eligible branchだけ8 wells x 2 seriesを再実行し、最大32 series-runs。
- model / LightGBM config / trained fold / HMM / PF / Beam / booster: `0 / 0 / 0 / 0 / 0 / 0 / 0`。
- 親control再学習: 0。GPU: 0。

静的・contract検証:

```bash
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb <train.py> <inference.py>
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test <train.py>
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test <inference.py>
.venv/bin/python -m py_compile <train.py> <inference.py>
.venv/bin/ruff check <train.py> <inference.py> tests/test_exp306_*.py
.venv/bin/pytest -q tests/test_exp306_robust_rts_l1_convergence_calibration_audit.py
make validate-exp EXP=exp306_robust_rts_l1_convergence_calibration_audit
make validate-template
make test
```

- Jupytext round-trip、py_compile、Ruff、strict experiment validation、template validationをPASSした。
- 専用contract testは`10 passed`。schema allowlist、固定sample、RTS A→B条件分岐、RTS/L1決定性、runtime gate、exact parity mutation検知、inference fail-closeを確認した。
- 親exp304 compact trainは2,258行・10役割章、exp306 compact trainは1,400行・11役割章。exp304のtruth attachment、shift score、科学gateを除外し、Stage 0 sample/branch/parity/runtime/gateをNotebook上へ展開したため、薄いhelper呼び出し構成ではない。
- train Notebook SHA: `11c97175fdc1d2f0591f9647ad11852644b8b7be3e7f578547acee059eb41dff`（compact/canonical一致）。
- inference Notebook SHA: `909b5479dbca610e77762fa18e96aa612d883f82540317eae2972ac38d15a05b`（compact/canonical一致）。
- Notebookは変換・静的検証だけで、ローカルsolver Stage 0、Kaggle package/push/run、科学評価は実行していない。
- 全体testは621 collected、`615 passed / 3 skipped / 3 failed`。failureは既存exp296の完了後configに対して旧testがKaggle実行前status/run flagを期待する2件と、既存exp345の現行Kaggle承認状態に対するtest期待不一致1件で、exp306の変更ファイル・contractとは無関係。exp306専用10件は全件PASSした。

## 2026-07-22 Kaggle CPU Stage 0 version 1完了

ユーザーの完了連絡後、同一canonical kernelのstatus、logs、必要生成物を回収した。別versionのpush、slug変更、full audit、科学評価、inference、submissionは行っていない。

確認コマンド:

```bash
kaggle kernels status kentookumura/exp306-rts-l1-convergence-calibration-audit-train
kaggle kernels logs kentookumura/exp306-rts-l1-convergence-calibration-audit-train
kaggle kernels output kentookumura/exp306-rts-l1-convergence-calibration-audit-train -p /tmp/exp306-kaggle-output-4WgIrQ
```

- statusは`KernelWorkerStatus.COMPLETE`。kernel version `1`、id_no `128231380`、private CPU、GPU/TPU/internet off。
- Notebook summary到達は`1774.422 sec`、nbconvert完了は約`1784.452 sec`。frozen-module/mistune/nbconvertのwarningはあったが実行結果に影響する例外はない。
- raw well identity content SHAはexpected `bbb687a1998092578583ce259309b49031d095bde57cbb26c0ab8808d2379b32`と一致。truth/scientific score loadedはfalse、CV/prediction/submissionはnull。
- 実行量はRTS A 128 + 条件付きRTS B 128 + L1 128 = 384 core series-runs。full-eligibleがL1だけだったためparityは8 wells x 2 series = 16 series-runs。model/LightGBM/fold/HMM/PF/Beam/booster/control再実行/GPUはすべて0。

Branch gate:

- L1 `max_admm=2000,rho=1,tol=1e-4`: convergence/technical `128/128`、iterations min/mean/max `264/656.758/1993`、実測`25.160889 sec`、full外挿`303.896362 sec`。finite/order/status/fallback/runtimeを全PASSし、8-well parityのoutput/status/iteration SHAも完全一致。唯一のfull-eligible branch。
- RTS A `max_irls=32,tol=1e-6`: convergence/technical `7/128`、FAIL `121`（horizontal 59 / typewell 62）、iterations mean/max `31.844/32`、実測`999.043880 sec`、full外挿`12066.576863 sec`。finite/order/status/fallback/runtimeはPASSしたがall-convergenceでFAIL。
- RTS B `max_irls=32,tol=1e-4`: convergence/technical `108/128`、FAIL `20`（horizontal 7 / typewell 13）、iterations mean/max `23.219/32`、実測`695.614844 sec`、full外挿`8401.723041 sec`。Aより改善したがall-convergenceでFAIL。事前固定分岐を使い切ったため追加救済は行わない。

取得生成物のSHA再計算:

- Stage 0 input raw/decompressed: `308d09d90dc13ba29db6b0a5e7c5930833fa1d3833ef63fe7376b4ef074126ec` / `ee4b26f34177d3367e4c3e84900727bc115e497bd076268c1e62a1f5276ce50b`。
- Stage 0 output raw/decompressed: `649a98cdd5591bdac35582e69ca5c347c7b66809376e0c2a261f441fa1d0284b` / `8a6f7e38bcea659f5ab7d0fd0cf37475c6d5d84bfed1826b5febfcb7ecf67df7`。
- Solver status raw/decompressed: `fcb2fe6a658cec66be314353475c475b7e19c7335145cda89c49df2850147592` / `bef261e1b905dd59e05f91c9966d481ede3fb063566db0f5fca0fe829fb665e9`。
- Sample manifest raw: `67508cba8dab2de14e13d77edec6b8faadab8fdacd44334ca2ce029b6ddcf691`。
- Scientific contract / Stage 0 gate / parity manifest / summary file SHA: `a13bd5a7ff2119e002bfe6f8bae08207e4b2c45c9e8be0de581c27045921ee54` / `cdfd1397425d98076d0c4da029b5bac6640f50bff1f935ae46019024077e3887` / `1a4952023fe006f082af5be6cdcdc95097e996bebb440a37ea6a2060bab05089` / `1217039de7d5db45c6e5d2ab9a207c555c673e662bf7bf74e82825321449e6ea`。
- gzip raw/decompressed SHAはgate JSON記録値と全件一致。大容量生成物は`/tmp/exp306-kaggle-output-4WgIrQ`で検証し、repositoryへ保存していない。
- `kaggle/train/config.yaml`はversion 1で実行したpackage snapshot（SHA `d55d723c8ee08fe91a329f1aee607cb56a792461a132d7a4a4f350c144d44c78`）として変更しない。完了記録とrun flag falseを反映したcanonical `config.yaml` SHAは`a168e3d1096b45979140fd267bd37992aea276a3fa176150497342bba5255e5f`、`metrics.json` SHAは`21e65afd4a96d3c37d0788784e84fe802bff858b190f86d237ec90cf08f3c883`。

判断:

- L1だけがfull auditへ進む技術資格を得た。full auditは未実装・未承認なので、run flagをfalseへ戻して停止する。
- RTS A/Bは不適格としてexp306内で閉じる。再訪するなら残る20 FAILのtarget-free failure profileと単一変更を別steeringで事前固定する。
- technical-only結果であり、exp304 selected SWT、exp305 closed判断、科学score、inference、submissionを変更しない。

## 2026-07-23 L1 full audit後続設計

- ユーザーの明示依頼により、L1全773 wells / 1,546 series technical auditを新規`exp351_exp306_l1_full_convergence_audit`へdesign-onlyで切り出した。
- exp351は親kernel version 1、raw identity、scientific contract、Stage 0 gate/sample/input/L1 output/L1 status/parity SHAを固定anchorとして参照する。
- exp306のfull implementation/run flagはfalseのまま維持し、code/package/Stage 0 artifactを変更しない。
- exp351の実装、canonical Notebook採用、Kaggle package/push/runは未承認。scientific score、RTS救済、inference、submissionも対象外。
