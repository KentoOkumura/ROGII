# exp513_hjyact_v2_final_standalone_public_lb_audit セッションノート

## 目的

exp512の50% public成分である完全な`hjyact_v2_final`を単独でhidden-safe再生成し、source parityと
Public LBを確認する。今回はexp512のKaggle失敗を反映して実装と静的検証まで行う。

## 現在の状態

- Route: `ensemble`
- 状態: Kaggle同一package 2回完走、visible final source parity PASS、hidden RNG決定性は未確定
- 親: `exp512_hjyact_v2_final_10pct_hedge_on_exp413`
- CV / LB / submission: なし / なし / なし
- 正規train / inference Notebook: template placeholder
- 正規Notebook採用、提出: 未承認
- Kaggle package / run / 完了監視 / output技術検証: 2026-08-05承認済み

## 2026-08-05 設計記録

- `kaggle-review-exp`に従い、steeringを実験ディレクトリより先に作成した。
- `kaggle-strategy`に従い、exp512、`backlog/KAGGLE_DIRECTION.md`、`experiment_summary.md`、
  `SUBMISSIONS.md`、再現性ガードを確認した。
- 最終境界はexp512の
  `after_complete_hjyact_v2_final_stack_and_pf_seed_branch_hedge`に固定した。
- exp413、等率blend、cross-consumer candidate reuseを除外し、公開成分内の
  SP45 / learned、guarded overlap、visible-prefix、model-package、seed-branch hedgeは保持する。
- 公開sourceのPF/Beamと保存済みMLがともに最終予測へ本質的に寄与するためrouteを`ensemble`とした。
- source score `6.568`は参照値であり、exp513のLB実績としては記録していない。

## exp512失敗診断

`kaggle-review`の失敗実行手順に従い、最初の意味のあるtracebackと設定/path前提を確認した。

### version 1

- runtime: 129.020秒で`ERROR`。
- traceback境界: `FormationPlaneKNN`へ空well listが入り`KeyError: wid`。
- 原因分類: data path / Kaggle runtime mount。
- 根本原因: source version 2の旧competition root
  `/kaggle/input/competitions/rogii-wellbore-geology-prediction`固定。現在のmountは
  `/kaggle/input/rogii-wellbore-geology-prediction`だった。
- scientific prediction: Ridge fit、保存model predictionとも0回。

### version 2

- runtime: 135.781秒で`ERROR`。
- competition rootと全input SHA監査はPASS。
- 原因分類: data path / config propagation。
- 根本原因: Ridge特徴量tableだけが旧dataset root
  `/kaggle/input/datasets/ravaghi/wellbore-geology-prediction-artifacts`を直接参照し、監査済みrootが
  `CFG.artifacts_path`へ伝播していなかった。
- scientific prediction: Ridge fit、保存model predictionとも0回。

### package preflight

- exp512初回SaveKernelは`The kernel source must be less than 1 megabytes in size.`で400。
- exp512は全`src/` bundleを除外して909--932 KiBへ縮小した。
- exp513はexp413 runtime / 75 modelsのbootstrap依存 / shared-DAGを除外し、candidate sourceを
  236,961 bytesへ縮小した。その後の実行preflightでpackage 437,126 bytesを確認した。

## 2026-08-05 実装記録

- source Notebook SHA `4b4879a6...39a23933c`とcode-cell SHA
  `ee93ce4c...4c126088`を再確認した。
- exp512 v3 generator SHA `de18a4ec...10a57e`を固定参照する専用generatorを作成した。
- active source 37 code cellsを抽出し、診断/CV-only cellを除外した。
- exp512 v3と同じcompetition root resolverを採用し、`train/test/sample_submission.csv`を持つrootを
  旧/current候補から一意解決する。
- Ridge rootは`data/train.csv`と5 trainer wrapperを含むrequired-file setで一意解決し、全SHA確認後の
  `HJYACT_INPUT_AUDIT["roots"]["ridge"]`を`RIDGE_ARTIFACT_ROOT`と`CFG.artifacts_path`へ渡す。
- learned 3 modelsとmodel-package inputsもrequired-file SHAで監査する。
- precomputed visible learned submission探索とinference-time training fallbackを除外した。
- exp413 runtime、50/50 blend、candidate reuse trackerを生成対象から除外した。
- source final後は値を変更せず、dynamic sampleのschema / ID / row / duplicate / finiteを監査する。
- visible ID-order一致時だけsource final SHAをpost-hoc assertionする。
- input/model/reproducibility manifestとmetricsをKaggle workingへ保存する。
- learned feature frameはprediction後に解放し、exp512のshared-DAG用保持によるメモリを避ける。

## 実行量preflight

- active scientific variant: 1
- final output: 1（standalone、weight fit / gridなし）
- LightGBM train config: 0
- new booster: 0
- parent/control retraining: 0
- source Ridge: 1 config × 5 folds = 5 runtime fits
- saved model files: 13
- contained estimators: 33
- excluded exp413 saved model files: 75
- accelerator contract: GPU / internet off
- original source runtime reference: 787.7964199秒

## 2026-08-05 Kaggle実行承認

- ユーザーの「実行してください」を、候補のKaggle GPU / internet-off実行、完了監視、
  `submission.csv`とmanifestのoutput取得・技術検証までの承認として記録した。
- competition submit、正規Notebook採用、profile / threshold / weight変更には承認を拡張しない。
- push前実行量を再確認した: scientific variant 1、final output 1、LightGBM train config 0、
  new booster 0、parent/control retraining 0、runtime Ridge 1 config × 5 folds、saved model 13。
- 正規`*_inference.ipynb`はplaceholderのまま維持し、compact候補から実行専用
  `*_current_test_inference.ipynb`を生成してcanonical kernel
  `kentookumura/exp513-hjyact-v2-standalone-lb-audit-inference`へpushする。
- exp413側の75 saved modelと11 kernel inputはstandalone境界外なのでattachしない。公開成分に必要な
  7 datasetとcompetition inputだけをattachし、`--no-src`でpackageする。
- 実行専用Notebook SHAは`d6e247a8...4bab000`。bootstrap済みpackageは437,126 bytes、
  SHA`9d7efcca...8e6d4b`で1 MiB制限をPASSした。
- package metadataはprivate、GPU `Gpu`、internet off、run-on-push、7 datasets、0 kernel sources、
  competition inputを確認した。bootstrap内のconfig / standalone contract / model manifest / source SHAもPASS。
- push前の同一canonical kernel pullは403だったため、既存private kernelは存在しないものとして初回pushへ進む。
- canonical kernelへversion 1のpushが成功し、push直後statusは`KernelWorkerStatus.RUNNING`。
- 初回readbackはKaggle APIの一時的DNS failureで失敗したため、実行監視と併せてretryする。
- version 1は`KernelWorkerStatus.COMPLETE`。科学runtimeは819.939秒で、14,151行 / 3 wellsを生成した。
- competition rootとRidge rootはいずれもrequired-content resolverで解決され、全required input SHAを確認した。
  Ridge 5 fold fitと全保存model推論まで完走し、exp512 v1/v2の早期path failureは再発しなかった。
- `submission.csv`と`hjyact_v2_final_component.csv`は完全一致し、SHAはsource visible finalと同じ
  `b192d3f3...9ded4a`。sample ID-order SHAも`e6a2a380...7e269`で一致した。
- Kaggle readbackは49/49セルのsource完全一致。kernel id_noは`129735820`、private、GPU、internet off、
  7 datasets、0 kernel sources、competition inputを確認した。
- 必要な6 outputとlogだけを`kaggle/output/inference_v1/`へ取得し、row / columns / unique ID / finite / SHAをPASS。
- external competition submissionは未実行。再現性ガードの2回目は同一packageを変更せずversion 2として行う。
- version 1実行中に親exp512 generatorがv4対応でSHA`e5a7d59c...a5915`へ更新された。
  exp513の公開component抽出境界への影響を診断し、新しい親で一時再生成した候補が凍結済み候補と
  byte-for-byte一致（SHA`542f0947...9e9baf43`）することを確認した。履歴契約は実装時の親SHAを保持し、
  専用testはmutableな親の現在値ではなく固定dependency SHAと候補SHAを検証する。
- 同一package SHA`9d7efcca...8e6d4b`をcanonical kernelのversion 2へpushし、直後statusは
  `KernelWorkerStatus.RUNNING`。科学コード、input、profile、seed、metadataはversion 1から変更していない。
- version 2も`KernelWorkerStatus.COMPLETE`。科学runtimeは816.881秒、final `submission.csv` SHAは
  version 1とsource visible finalの両方に一致する`b192d3f3...9ded4a`だった。sample ID-order SHAも一致。
- visible final 2-run reproducibilityはPASS。ただしpre-overrideの
  `submission_sp45_learned_w0.60.csv`統計はv1とv2で異なった。
  - v1 mean/std/RMSE-vs-SP45/p95: `11905.160092 / 277.945461 / 2.214184 / 4.379985`
  - v2 mean/std/RMSE-vs-SP45/p95: `11904.617062 / 277.397902 / 2.302374 / 4.613829`
- source由来の`_pf_ancc` / `_pf_z`はNumba内乱数を明示seedせず、well-levelをthread並列生成する。
  visible 3 wellsは後段overrideで全行置換されるためfinalは一致するが、未知wellの決定性は証明できない。
  `parallel_rng_policy`に従いdeterministic anchorはfalseのまま、hidden code submission readyをfalseとした。
- Kaggle CLIは`owner/slug/1`をparseしても`ListKernelSessionOutput.version_label`へ渡さずlatestを返す。
  version 1中間CSVとして誤取得したlatest-v2複製は`kaggle/output/cli_latest_alias_diagnostic/`へ隔離した。
  version 1の最終output/metricsはversion 2 push前に取得済みなので、final 2-run SHA判定には影響しない。
- competition submit、正規Notebook採用、profile / threshold / weight変更は行っていない。

## 生成物とSHA

- generator: `experiments/exp513_hjyact_v2_final_standalone_public_lb_audit/prepare_exp513_hjyact_v2_standalone_candidate.py`
  - SHA: `d90b8b18...ef9625f`
- candidate source:
  `exp513_hjyact_v2_final_standalone_public_lb_audit_compact_selfcontained_inference.py`
  - 5,097行 / 236,961 bytes / SHA `542f0947...9e9baf43`
- candidate Notebook:
  `exp513_hjyact_v2_final_standalone_public_lb_audit_compact_selfcontained_inference.ipynb`
  - 48 cells / SHA `1e919d36...35826c5e`
- standalone contract SHA: `eabd8451...18df569`
- model manifest SHA: `afa1bfd5...9a3247d`
- 正規Notebook: 変更なし。

## 静的検証

- `py_compile`: PASS
- Ruff F821: PASS
- Jupytext round-trip: PASS
- 専用pytest: `7 passed`
- exp512依存test込み: `13 passed`
- strict `validate-exp`: PASS
- template validation: PASS
- legacy competition/Ridge root直接assignment: 0
- exp413 runtime / candidate reuse tracker / fixed blend: 0
- `Path(__file__)`: 0。source内のmodule shim用`module.__file__`はpath解決に使わない安全な参照として残る。
- parent compact比較: exp512 6,879行 / 8章、exp513 5,097行 / 7章。
  公開finalの6章を保持し、exp413 runtimeとshared-DAG/fixed-blend outputだけを削除した。

## 実行コマンド

```bash
uv run python experiments/exp513_hjyact_v2_final_standalone_public_lb_audit/prepare_exp513_hjyact_v2_standalone_candidate.py \
  /tmp/exp512-hjyact-v2-source/ultimate-pf-config-strategy-a-reproducible-score.ipynb
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb \
  experiments/exp513_hjyact_v2_final_standalone_public_lb_audit/exp513_hjyact_v2_final_standalone_public_lb_audit_compact_selfcontained_inference.py
.venv/bin/python -m py_compile \
  experiments/exp513_hjyact_v2_final_standalone_public_lb_audit/exp513_hjyact_v2_final_standalone_public_lb_audit_compact_selfcontained_inference.py
.venv/bin/ruff check \
  experiments/exp513_hjyact_v2_final_standalone_public_lb_audit/exp513_hjyact_v2_final_standalone_public_lb_audit_compact_selfcontained_inference.py \
  experiments/exp513_hjyact_v2_final_standalone_public_lb_audit/prepare_exp513_hjyact_v2_standalone_candidate.py \
  experiments/exp513_hjyact_v2_final_standalone_public_lb_audit/tests/test_exp513_contract.py \
  --select F821
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test \
  experiments/exp513_hjyact_v2_final_standalone_public_lb_audit/exp513_hjyact_v2_final_standalone_public_lb_audit_compact_selfcontained_inference.py
.venv/bin/pytest -q \
  experiments/exp512_hjyact_v2_final_10pct_hedge_on_exp413/tests/test_exp512_contract.py \
  experiments/exp513_hjyact_v2_final_standalone_public_lb_audit/tests/test_exp513_contract.py
make validate-exp EXP=exp513_hjyact_v2_final_standalone_public_lb_audit
make validate-template
```

ローカルNotebook実行、submit-check、competition submitは行っていない。Kaggle package、同一package 2 run、
必要output取得と技術検証は実行済み。

## 再現性メモ

- seed policy: source version 2の固定profile/seed semanticsを保持する。
- stochastic components: SP45/learned likelihood-PF、visible-prefix seed bank、PF seed-branch hedge。
- visible finalの2-run SHAは一致したが、pre-override PF/Ridge blend統計差によりhidden deterministic anchorは不可。
- gzip生成物はdecompressed content SHAを主証拠とする。
- bootstrapはprepare後にmetadataと展開config/source/model manifestのSHAをreadbackする。

## 次のアクション

1. source RNG semanticsをそのまま維持するか、well別明示seedを加えた別candidateを作るかユーザー判断を得る。
2. seed固定を選ぶ場合は、source visible final parityを崩さないことと未知well相当の中間予測2-run一致を確認する。
3. 正規Notebook採用、submit-check、competition submissionはそれぞれ別承認を得る。
