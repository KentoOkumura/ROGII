# exp390_parallel_strip_surface_registration_readout セッションノート

## 目的

近接horizontal wellの強い平行配置をquery-centricな`(s,n)` strip座標へ変換し、
同じalong-track位置のouter-train `S=TVT+Z`をcross-track方向へtwo-sided補間する
1本の物理candidateを、exp226保存OOFに対してfold-safeに評価する。

## 現在の状態

- Route: `pf_beam`
- 状態: Kaggle CPU Stage 0 preflight完了・two-sided support gate FAILでclosed
- CV / LB: なし
- 設計承認: あり
- 実装: 2026-07-24のユーザー指示で承認・完了
- 正規train Notebook採用 / package / push / 16-well Stage 0 preflight: 完了
- 773-well full run / Stage 1 / 2 / inference / submission: Stage 0 FAILでblocked
- 実行優先順位: exp383は未確定だが、2026-07-24のユーザー指示で固定値を変えず
  exp390 Stage 0 preflightを先行

## コマンドログ

### 2026-07-24 設計セッション

実行済み:

```bash
make new-steering EXP=exp390_parallel_strip_surface_registration_readout
make new-exp EXP=exp390_parallel_strip_surface_registration_readout
```

確認・設計根拠:

- `kaggle-review-exp`の実験作成・設計ルールを確認した。
- `AGENTS.md`と`docs/06_reproducibility.md`を確認した。
- exp114 geometry summaryをread-only再集計した。
  - nearest axial-angle median `0.411°`
  - nearest angle `<5°` `95.34%`、`<10°` `97.80%`
  - nearest centroid distance median `482.57 ft`
  - cross-track distance median `383.90 ft`
  - projected overlap median `0.99936`
  - tortuosity `<=1.01` `98.58%`、`<=1.05` `100%`
- exp114 / exp119 / exp201 / exp226 / exp273 / exp383の結果・設計を比較した。
- exp390をexp226 parent、`pf_beam` route、exp383結果確認後の独立P1として固定した。

今回はドキュメント・config・backlogのみを変更した。実装、Notebook生成/採用、
Kaggle package/push/run、ローカルNotebook実行、inference、submissionは行っていない。

### 2026-07-24 実装セッション

ユーザーの`exp390を実装してください`をimplementation-only承認として反映した。
exp383はStage 0 preflight実行中で性能結果未確定だが、exp390の固定threshold、
fit、gateは変更せず、Kaggle実行順だけを維持した。

実装:

- `exp390_parallel_strip_surface_registration_readout_compact_selfcontained_train.py`
  - 10章、2,270行のJupytext percent形式。
  - exp226 fold/SHA contract、role-read ledger、query PCA axis、modulo-π pair角度、
    overlap/cross-track/monotone gate、max16 donor固定を実装。
  - 64 ft node、same-s donor補間、two-sided weighted Huber local-linear、
    5-node median、prefix Huber intercept、exp226 exact fallbackを実装。
  - Stage 0 target-free、Stage 1 prefix末尾512 rolling-origin＋circular control、
    Stage 2 truth-late direct/scope/by-well/oracle readout、logical SHAを実装。
- `exp390_parallel_strip_surface_registration_readout_compact_selfcontained_inference.py`
  - Stage 2両gateと別承認がない限り停止するfail-closed候補。
- `experiments/exp390_parallel_strip_surface_registration_readout/tests/test_exp390_parallel_strip_surface_registration_readout.py`
  - 専用contract test 11件。
- compact train/inferenceの`.ipynb`候補をJupytext変換で生成。

検証:

```bash
.venv/bin/pytest -q experiments/exp390_parallel_strip_surface_registration_readout/tests/test_exp390_parallel_strip_surface_registration_readout.py
.venv/bin/ruff check experiments/exp390_parallel_strip_surface_registration_readout/exp390_parallel_strip_surface_registration_readout_compact_selfcontained_train.py experiments/exp390_parallel_strip_surface_registration_readout/exp390_parallel_strip_surface_registration_readout_compact_selfcontained_inference.py experiments/exp390_parallel_strip_surface_registration_readout/tests/test_exp390_parallel_strip_surface_registration_readout.py
.venv/bin/python -m py_compile experiments/exp390_parallel_strip_surface_registration_readout/settings.py experiments/exp390_parallel_strip_surface_registration_readout/exp390_parallel_strip_surface_registration_readout_compact_selfcontained_train.py experiments/exp390_parallel_strip_surface_registration_readout/exp390_parallel_strip_surface_registration_readout_compact_selfcontained_inference.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp390_parallel_strip_surface_registration_readout/exp390_parallel_strip_surface_registration_readout_compact_selfcontained_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp390_parallel_strip_surface_registration_readout/exp390_parallel_strip_surface_registration_readout_compact_selfcontained_inference.py
make validate-exp EXP=exp390_parallel_strip_surface_registration_readout
```

- 専用pytest: `11 passed`
- Ruff: PASS
- `py_compile`: PASS
- Jupytext train/inference round-trip: PASS
- strict experiment validation: PASS
- `__file__`依存: 0
- read-only入力contract監査: raw train `773 wells`、保存exp226 OOF
  `3,783,989 rows / 773 wells / folds [0,1,2,3,4]`、fold identityと
  decompressed SHAが一致。
- 全repository test: `917 passed / 6 skipped / 3 failed`。この時点のexp390専用10件は全PASS。
  FAILは未変更の既存状態であるexp296のstatus/run flag期待2件と、
  exp384のKaggle実行承認flag期待1件のみ。
- 親exp226にcompact self-contained版はなく、正規Jupytext trainは111行・6章。
  exp390候補は重いgeometry/fit/Stage 0/1/2をNotebookセル内で追える10章・2,270行とした。
- 正規train/inference Notebookは既存template scaffoldのままで上書きしていない。
- Kaggle package、push、run、ローカルNotebook実行、inference、submissionは行っていない。

### 2026-07-24 Kaggle CPU Stage 0 preflight実行セッション

ユーザーの`実行してください`を、正規train Notebook採用、Kaggle package/push、
private CPU / internet offの16-well Stage 0 preflight承認として反映した。
設計で別承認とした773-well full run、inference、submissionは未承認のまま維持する。

push前の実行量:

- scientific variant / candidate: `1 / 1`
- reporting folds / preflight query wells: `5 / 16`
- full-run query strip solves: `773`（今回は未実行）
- fitted model / model config / trained fold / booster: `0 / 0 / 0 / 0`
- HMM / PF / Beam: `0 / 0 / 0`
- exp226 parent control再生成: `0`
- accelerator: Kaggle CPU、internet off

確認:

- OAuth credential、KAGGLE_USERNAME、legacy keyは利用可能。独立API Tokenは未設定だが
  Kaggle CLI 2.2.3の実行を妨げない。
- compact train候補を正規train Notebookへ採用し、23 cellsのtype/source内容一致を確認した。
- canonical kernel候補は
  `kentookumura/exp390-parallel-strip-registration-train` /
  `exp390 parallel strip registration train`。
- push前の`kaggle kernels pull`は403で、同canonical kernelの既存versionは確認されなかった。

Kaggle version 1:

- kernel: `kentookumura/exp390-parallel-strip-registration-train`
- numeric id: `128480051`
- metadata: private / CPU / internet off
- attached sources: exp226 train output、exp115 train output
- status: `ERROR`
- 実験本体のpair/support/fit計算に入る前、入力件数検証で停止した。
- 原因: `/kaggle/input`の最初の`*__horizontal_well.csv`の親を採用したため、
  773件の`train/`ではなく3件の`test/`を選択した。
- 対応: 親ディレクトリごとにファイル数を数え、
  `validation.expected_wells=773`と一意に一致するディレクトリだけを採用する
  fail-closed resolverへ修正した。3件test / 7件trainの回帰testを追加し、
  専用pytest `11 passed`、Jupytext source/canonical 23 cells一致を再確認した。
- version 1はscientific metrics、Stage 0 coverage、runtime/RSS gateを生成していないため、
  仮説に対するPASS/FAILには数えない。

Kaggle version 2:

- URL: `https://www.kaggle.com/code/kentookumura/exp390-parallel-strip-registration-train`
- numeric id: `128480051`
- status: `COMPLETE`
- metadata: private / CPU / internet off
- kernel sources: exp226 train output、exp115 train output
- package: canonical body 23 cells + bootstrap 1 cell、support files 23
- runtime package SHA:
  - bootstrap ZIP:
    `c56282687de83267fc39ac6d05e167e169268e1260b51e24d552e3b221e52a01`
  - embedded config:
    `2b460160833acd77ba6c6e5321a545a518475b93ba1b6740d20a969c6112110a`
  - embedded compact train source:
    `3f9dd8abe2ab77b191447db69d8149a2d51e08cbb8f14e83510615e1dc50b1c6`
- exp226 OOF decompressed SHA:
  `709eb726cc30da523f017ed0dbd0371967b88a91ddcf25578eb9356f28e4c609`
- 実行量: 1 candidate、5 reporting folds、16 query wells、73,586 processed rows、
  fitted model / booster / HMM / PF / Beam / parent replayはすべて0。
- runtime: `60.401419 sec`、773-well projected runtime `2,918.143546 sec`
  （約48.64分）、projected peak RSS `0.657509 GB`。

Stage 0結果:

- input `3,783,989 rows / 773 wells / folds [0,1,2,3,4]`: PASS
- exp226 fallback finite coverage `1.0`: PASS
- pair angle p95 `1.769352° <= 5°`: PASS
- pair overlap p05 `0.897013 >= 0.80`: PASS
- target suffix truth / raw Formation / GR reads `0 / 0 / 0`: PASS
- source-valid overlap `0`: PASS
- runtime / RSS: PASS
- two-sided strip row coverage `0.0 < 0.50`: FAIL
- two-sided strip well coverage `0.0 < 0.75`: FAIL
- eligible-node unique-donor p05 `0.0 < 4`: FAIL
- 16 wells中eligible pairを1本以上持つqueryは8本、全eligible pairは10本、
  queryあたり最大2本。1,598 nodes中、最大unique donorは2、
  正負両側support nodeは0、valid fitは0。
- Stage 0 overall: `FAIL`、最終status:
  `stage0_fail_closed`。

成果物:

- Kaggle outputからguard、summary、fold/geometry/pair/node/fit/calibration/role-read、
  SHA manifestだけを
  `/tmp/kaggle-output/exp390_parallel_strip_surface_registration_readout/train_v2`
  へ取得した。output archive全体は取得していない。
- `metrics.json` SHA:
  `7d5eb16909f9835939660fc69369e3f8945b65396f64e2c6aea7e9fb56d04344`
- Stage 0 guard SHA:
  `3d4859e8c4ac6a0128fbfc38ae1f71a0d3dbe261c8c2bba623129041284d8054`
- summary SHA:
  `4e5d68664a14e77f4ed399df02851a73a8e550c1385af02ecef3424f65906dfd`
- SHA manifest SHA:
  `1f8bc775423dca2a75690a97899f5f5f2aec35602948bebff31746ddae5fc825`
- logical SHA:
  - fold manifest:
    `da811cb73a797a3feadbb7635c3add5218daf022dbc2f09fec16efebe8979bfa`
  - geometry:
    `bb7c2d48fe2294b3eab6c452847eace20f1fc7838a46ac7075fd5a19fe7ba296`
  - eligible pairs:
    `c75733c1281bcbf6feb9e9654e63259554f6bebc4347843600dd05c802afdd43`
  - query-node donors:
    `25d2b03114a7bbb053d73b789dc8442b530faf41218b2498489f87451348daaa`
  - fit diagnostics:
    `7ab50d59ccfd48779dfca475b40368a9d617f2a00989e23aed1d48b07f64fb5e`

判断:

- geometry angle / overlapと実行安全性は成立したが、事前固定したminimum 4 donorかつ
  two-sided supportが完全に退化した。
- Stage 0 FAIL後のthreshold、distance、overlap、min donor、one-sided、bandwidth、
  smoothing、Huber、soft blend、selector救済は禁止事項に該当するため行わない。
- 773-well full run、Stage 1 / 2、inference、submissionは実行せず、branchを閉じる。

## 変更点

- well centroid KNNではなく、query PCA軸のalong-track `s`とcross-track `n`を使う。
- same-s donor `S=TVT+Z`をtwo-sided Huber local-linear fitする。
- pair eligibility、max16/min4 donors、64 ft node、1000 ft bandwidth、5-node median、
  prefix Huber intercept、exp226 fallbackを実行前に固定した。
- candidateは`parallel_strip_two_sided_fallback_exp226`の1本だけとした。
- Stage 0 target-free、Stage 1 prefix rolling-origin、Stage 2 truth-lateを分離した。
- scientific-supportとpromotion-safetyを分け、後者FAILではinferenceへ進めない。

## 予定計算量

- scientific variant / candidate: `1 / 1`
- reporting folds / query well strip solves: `5 / 773`
- fitted model / LightGBM config / trained fold / booster: `0 / 0 / 0 / 0`
- HMM / PF / Beam runs: `0 / 0 / 0`
- exp226 parent control再生成: `0`
- accelerator: CPU、internet off
- 16-well Stage 0 resource preflightでFAILしたため、full runはblocked

## 現在の実行境界

- 正規train Notebookはcompact候補を採用済み。正規inference Notebookは
  template scaffoldのまま保持する。
- package、push、runは16-well Stage 0 preflightだけ承認済み。
- 773-well full run、current-test生成、inference、submissionは禁止。
- exp383結果を見てexp390の固定thresholdやfit parameterを変更しない。
- Stage FAIL後のangle/distance/overlap/donor/bandwidth/smoothing/Huber/calibration/
  one-sided/soft-blend/selector救済を行わない。

## 再現性メモ

- seed policy: RNGなし。fold/query/donor/pair/node/fit/rowのimmutable keyでstable sort。
- stochastic components: なし。
- parallel RNG: RNGなし。parallel時もwell/pair/node keyで再sort。
- CPU/GPU runtime: 将来のKaggle private CPUのみ。GPUなし、internet off。
- input / content SHA: exp226 control、fold、geometry、pair、node donor、fit、calibrationの
  schema/logical SHAをversion 2で記録済み。valid fit / strip predictionはsupport FAILで未生成。
- model manifest / model SHA: fitted modelなし。
- prediction SHA: Stage 0 support FAILでstrip prediction artifactは未生成。
- submission SHA: inference/submission未承認のため対象外。
- deterministic anchor: Stage 0 FAILのため未確立。成功rerunは行わない。
- Kaggle bootstrap: version 2 push前にmetadata、config/source/bootstrap SHAを照合済み。

## 次のアクション

1. exp390は`stage0_failed_closed_sparse_two_sided_support`として終了する。
2. 同じthresholdを緩和せず、parallel-strip系を再検討する前に、全773 wellsの
   target-free pair-densityだけを測る0-fit support censusを別アイデアとして検討する。
3. exp390のfull run、Stage 1 / 2、inference、submissionへは進まない。
