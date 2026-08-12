# exp383_all_tvt_stratigraphic_vector_drift_field セッションノート

## 目的

全outer-train正解TVTと6地層面から、prefix校正付きabsolute/vector drift fieldを作り、
exp226を1 ft以上改善できる物理signalかを調べる。

## 現在の状態

- Route: `pf_beam`
- 状態: `stage0_resource_fail_closed`
- CV / LB: まだなし
- 実装: compact self-contained trainとfail-closed inferenceを実装済み
- Notebook: compact候補を正規train/inferenceへ採用済み
- Kaggle package: version 1履歴を保持し、ローカル生成物は修正版 /
  `run_on_push=false` / execution無効へ更新済み
- Kaggle push/run: canonical version 1は`ERROR`。Stage 0 resource FAILでfull run停止
- inference/submission: 無効

## コマンドログ

2026-07-24:

```bash
make new-steering EXP=exp383_all_tvt_stratigraphic_vector_drift_field
make new-exp EXP=exp383_all_tvt_stratigraphic_vector_drift_field
.venv/bin/python -m py_compile experiments/exp383_all_tvt_stratigraphic_vector_drift_field/exp383_all_tvt_stratigraphic_vector_drift_field_compact_selfcontained_train.py
.venv/bin/ruff check experiments/exp383_all_tvt_stratigraphic_vector_drift_field/exp383_all_tvt_stratigraphic_vector_drift_field_compact_selfcontained_train.py experiments/exp383_all_tvt_stratigraphic_vector_drift_field/tests/test_exp383_all_tvt_stratigraphic_vector_drift_field.py
.venv/bin/pytest -q experiments/exp383_all_tvt_stratigraphic_vector_drift_field/tests/test_exp383_all_tvt_stratigraphic_vector_drift_field.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp383_all_tvt_stratigraphic_vector_drift_field/exp383_all_tvt_stratigraphic_vector_drift_field_compact_selfcontained_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp383_all_tvt_stratigraphic_vector_drift_field/exp383_all_tvt_stratigraphic_vector_drift_field_compact_selfcontained_inference.py
EXP383_IMPORT_ONLY=1 .venv/bin/python -c "<read-only raw/exp226 contract audit>"
make prepare-kaggle-notebooks EXP=exp383_all_tvt_stratigraphic_vector_drift_field EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp383-all-tvt-stratigraphic-vector-drift-field-train --title 'exp383 all tvt stratigraphic vector drift field train' --run-on-push --strict"
make prepare-kaggle-notebooks EXP=exp383_all_tvt_stratigraphic_vector_drift_field EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp383-tvt-vector-drift-field-train --title 'exp383 tvt vector drift field train' --run-on-push --strict"
```

`task prepare-kaggle-notebooks ...`も先に試したが、この環境に`task` binaryがないため
`make`の同等入口へ切り替えた。
初回の53文字title / 51文字slug本体はKaggle `SaveKernel` 400となりkernelを作成しなかった。
既知の50文字前後の制約に合わせ、科学contractを変えず意味を残したcanonical
`kentookumura/exp383-tvt-vector-drift-field-train` /
`exp383 tvt vector drift field train`へid/titleを同時に短縮した。
短縮後のcanonical packageはversion 1としてpush成功。id_noは`128459031`。
pull-backでprivate / CPU / internet off / competition source /
exp115・exp226 kernel sourceを確認した。Kaggle statusは`RUNNING`を確認し、
約32分までerror logなし。ユーザーの「監視は止めていい。完了したら連絡する」
という指示により、その後の自動監視を停止した。

2026-07-25:

```bash
kaggle kernels status kentookumura/exp383-tvt-vector-drift-field-train
kaggle kernels logs kentookumura/exp383-tvt-vector-drift-field-train
kaggle kernels pull kentookumura/exp383-tvt-vector-drift-field-train -p /tmp/kaggle-pull/exp383-v1-failed -m
.venv/bin/pytest -q experiments/exp383_all_tvt_stratigraphic_vector_drift_field/tests/test_exp383_all_tvt_stratigraphic_vector_drift_field.py
.venv/bin/ruff check experiments/exp383_all_tvt_stratigraphic_vector_drift_field/exp383_all_tvt_stratigraphic_vector_drift_field_compact_selfcontained_train.py experiments/exp383_all_tvt_stratigraphic_vector_drift_field/tests/test_exp383_all_tvt_stratigraphic_vector_drift_field.py
.venv/bin/python -m py_compile experiments/exp383_all_tvt_stratigraphic_vector_drift_field/exp383_all_tvt_stratigraphic_vector_drift_field_compact_selfcontained_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp383_all_tvt_stratigraphic_vector_drift_field/exp383_all_tvt_stratigraphic_vector_drift_field_compact_selfcontained_train.py
```

- terminal status: `KernelWorkerStatus.ERROR`
- 最初の意味のある例外: `pandas.errors.MergeError`。
  `build_multiscale_donor_catalog()`がscaleの異なるdonor nodeを
  `(fold, well_id, row_idx, MD)`だけで`one_to_one`結合していた。
- 64/256/1024 ftのwindowは同じcenter MDを意図的に共有するため、この重複は
  データ異常ではなくmultiscale契約そのもの。scale込みの一意`query_id`を
  surface query前に付与し、そのIDをjoin keyに追加した。
- 失敗はnotebook開始後`22,069.784 sec`、experiment出力後
  `22,055.465 sec`（約6.13時間）で、まだfold 0のdonor surface付与直後、
  donor vector field・target 16 wells・truth join前だった。
- donor windowsはfold別に
  `209,467 / 207,822 / 209,218 / 209,423 / 207,506`、合計`1,043,436`。
  fold 0実測をnode数比例で投影したsurface stageだけで
  `109,866.787 sec`（30.52時間）、固定gate`30,600 sec`の`3.5904倍`。
- join bugはローカル修正済みだが、同じ科学contractの再実行はresource gateを
  明確にFAILするため、version 2はpushしない。
- ローカルKaggle packageを修正版・`run_on_push=false`へ再生成し、埋め込みconfigの
  status=`stage0_resource_fail_closed`、execution authorization=falseを確認した。
  ZIP SHA256は`f82bdac98a8b94df18b85ae83e707c1eb8ec8e20fc02524b506f89460539cb3e`、
  config SHA256は`05edae086f9d1ef1940eff4e59699b9a295dc362b5d184283157ed2eddcd9c92`。

## 変更点

- exp226のK16 scalar donor reweightではなく、全TVT multiscale donor catalogへ変更する設計を固定。
- 6地層surface、absolute/vector field、prefix vertical bias、uncertainty shrink、
  banded physical path solveを固定。
- Stage 0 target-free gateとStage 1の1 ft改善gateを固定。
- exp226と同じSHA256 well fold割当を再現し、保存OOFのfold identityと
  decompressed content SHAをfail-closed検証する。
- outer-trainだけで32 ft surface point、self-excluded 6-surface local plane、
  64/256/1024 ft Huber window、29次元signatureを生成する。
- 最大32 well / 1 well 4 node / 128 nodeの6-surface relative
  absolute/vector field、uncertainty、全prefix Huber bias、exp226 rate shrink、
  hard-prefix banded WLS pathを実装した。
- exp384が読む256 ft donor nodes、query fields、truth-free OOF keys、
  Stage 1後のOOF-with-truthとmanifest/SHA契約を実装した。
- targetの29次元signatureに必要な構造Sは、truth-freeな保存exp226 OOF path
  (`tvt_pred + Z`)を固定参照にした。fallback rateも同pathのraw MD勾配から導出する。
- ユーザーの`実行してください`指示を正規Notebook採用、Kaggle package/push、
  16-well preflight、PASS後full runの承認として反映した。
- compact train/inferenceを正規Notebookへ採用した。推論Notebookはfail-closedを維持する。
- 親kernelを`kentookumura/exp226-k16-kappa-repro-train`と
  `kentookumura/exp115-hidden-like-spatial-holdout-from-ppt-train`へ固定し、
  必須OOF/fold assignmentファイルの存在をKaggle上で確認した。
- multiscale donor surface joinをscale込み一意`query_id`へ修正し、
  row count不変とscale別payload保持を回帰testに追加した。

## 予定実行量

- 1 physical candidate
- 5 reporting folds
- fitted ML model / HMM / PF / Beam / booster: `0 / 0 / 0 / 0 / 0`
- parent exp226 control再実行: 0。保存済みOOFを使う。
- GPU: 0。Kaggle CPUを予定。
- Kaggle version 1 preflight: 実行済み・code/resource FAIL
- 再push / full run: Stage 0 gateにより停止
- 現在の実行mode: `stage0_resource_preflight`
- canonical kernel: `kentookumura/exp383-tvt-vector-drift-field-train`
  version 1 / id_no `128459031`
- inference / submission: 未承認

## 再現性メモ

- seed policy: RNGなし、stable fold/well/MD/scale順
- stochastic components: なし
- CPU/GPU runtime: Kaggle CPU予定、GPUなし
- input / feature schema SHA: 実行時に記録
- surface / donor / field / calibration / path content SHA: 実行時に記録
- model manifest: fitted modelなし。solver contract manifestを記録予定
- prediction / submission SHA: OOFは実行時、submissionは未承認
- deterministic anchor: 初回runでは主張しない

## 検証

- 専用contract test: 修正後`15 passed`
- Ruff: PASS
- py_compile: train / inference PASS
- Jupytext compact train / inference生成: PASS
- Jupytext train / inference round-trip: PASS
- `make validate-exp EXP=exp383_all_tvt_stratigraphic_vector_drift_field`: strict PASS
- OAuthとlegacy API credentialはKaggle CLI認証に使用可能。独立API Tokenは未設定だが
  今回のCLI実行を妨げない。
- canonical train kernel IDはpush前pullで未作成/アクセス不可を確認し、重複versionはない。
- version 1 package metadata: private / CPU / internet off / run-on-push、
  competition source `rogii-wellbore-geology-prediction`、親kernel 2件を確認。
- package埋め込みZIP監査: PASS。
  ZIP SHA256 `dc8281678dbfadc1dbe84962fc8a7c15c1d4cb67b307603a982e33642424cf5b`、
  埋め込み`config.yaml` SHA256
  `f1f1ec5a745482082ce0139b9200dafbf526f8d6a04c49466a0dc9c67d5da933`。
  mode=`stage0_resource_preflight`、16 wells、CPU、1 candidate / 5 folds /
  model・booster・HMM・PF・Beam各0を確認。
- read-only入力contract監査: raw train `773 wells`、保存exp226 OOF
  `3,783,989 rows / 773 wells / folds [0,1,2,3,4]`、fold identityと
  decompressed SHAが一致。
- 全repository test: `866 passed / 6 skipped / 4 failed`。
  exp383専用14件とscaffold/notebook testは全PASS。FAILは未変更の既存状態である
  exp296のstatus/run flag期待2件、exp377のpush flag期待1件、
  exp384の実行承認flag期待1件のみ。
- 2026-07-25修正後の全repository test: `930 passed / 6 skipped / 2 failed`。
  exp383/384を含む今回の変更範囲は全PASS。残る2件は未変更の既存exp296で、
  完了statusと過去の実行承認を期待するtestが現在のfail-closed configと不一致。
- trainは10章構成で、同一exp helper importや`__file__`に依存しない。
- 親exp226にcompact self-contained版はなく、正規Jupytext sourceは約120行の
  helper orchestrationである。exp383 compact trainはsurface、catalog、field、
  path、Stage 0/1をNotebookセル内で追える構成にした。
- 2026-07-24の最終実装監査で、専用contract test `14 passed`、Ruff、
  train/inference/settingsの`py_compile`、train/inferenceのJupytext
  round-trip、strict experiment validationを再実行して全PASSした。
- 正規train/inference Notebookとcompact生成Notebookはそれぞれbyte一致を確認した。
- join修正後のcompact/正規train Notebook SHA256はともに
  `a7a5e9f1897184b0324d812f9961b645808f8e5d21cd5b63898acf1485599bbc`。
- 16-well preflight対象は5 foldすべてを含む
  (`fold 0..4 = 3 / 2 / 2 / 5 / 4 wells`)。
- `KAGGLE_DIRECTION.md`のexp383を「未着手バックログ」から
  「実装済み・Kaggle train待ち」へ移し、未実行のpreflightを
  「実行準備完了」として記録した。

## 次のアクション

1. exp383はStage 0 resource FAILとして閉じ、version 2/full run/Stage 1へ進めない。
2. 低計算量のsurface basis/cacheへ仮説を変える場合は、別実験として設計し、
   parameter rescueではないことと早期resource gateを事前固定する。
3. exp384/385はexp383 PASS artifactがないため引き続きblockedとする。

## 2026-07-25 ユーザー閉鎖判断

- ユーザーがexp383とその後続実験の閉鎖を明示確認した。
- 対象はexp383、exp384、exp385。
- version 2、full run、Stage 1、inference、submission、同一実験内の救済を再開しない。
