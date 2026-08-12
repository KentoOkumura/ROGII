# exp415_fold_safe_rmse_prior_bounded_nudge_on_exp264 セッションノート

## 目的

exp407の悪化原因を定量的に確定し、候補別RMSEをtask weightにせず、
fold-safe additive priorとrisk-bounded TVT nudgeとして使う方法を
再現可能なzero-booster readoutにする。

## 現在の状態

- Route: `ensemble`
- 状態: Kaggle private CPU version 1完了、保存OOF診断上の方法確立
- Kaggle確認: 親`8.587004`、bounded nudge`8.563474`
- LB / inference / submission: 対象外
- 正規train Notebook: compact self-contained候補の採用を承認済み
- compact self-contained候補: 18セル、作成済み

## exp407の根本原因

3,783,989 base rows / 45,407,868 candidate-long rowsを同一keyで比較した。

- 親 hard RMSE: `8.587004386703422`
- exp407 hard RMSE: `8.66814102464331`
- candidate×fold平均score shiftだけを親へ適用:
  `8.580476914703985`、親以下4/5 folds
- exp407から平均shiftを除いたrow-local変化だけ:
  `8.673599263270791`、親以下1/5 folds
- final weightとrow-local score差stdのSpearman: `-0.593387`
- final weightとscore MAE悪化のSpearman: `-0.411670`
- final weightとbinary logloss悪化のSpearman: `-0.603779`
- final weightとcandidate定数shiftのSpearman: `-0.073243`
- 親margin 0.5--2.0のswitchがnet delta SSEの約74%を占めた。

したがって、候補別RMSE値や候補一律biasが原因ではない。inverse-RMSE weightingが
共有木の局所gradient / splitを変え、特に低重み候補のrow-local rankingを
壊したことが主因である。同じRMSE weightをbinary objectiveにも使った点も目的不整合。

## 固定policy

```text
parent_pos = argmin(parent_pred_abs_error)
prior_pos  = argmin(parent_pred_abs_error + fit_candidate_rmse)
raw_nudge  = 0.5 * (prior_tvt - parent_tvt)
correction = clip(raw_nudge, -0.25, +0.25)
prediction = parent_tvt + correction
```

候補RMSEはweightにもfeatureにもせず、prior候補の方向決定だけに使う。
各行の補正幅を0.25 ftに抑えるため、任意scopeで
`RMSE_new - RMSE_parent <= RMS(correction) <= 0.25 ft`が成り立つ。

## 保存OOF探索値

- bounded nudge RMSE: `8.563473931791524`
- 改善: `0.023530454911897536 ft`
- fold: 5/5改善
- 距離bucket: 4/4改善
- hidden-like: 2/2改善
- worst-well delta: `+0.17137908743683994 ft`
- nonworse wells: 544/773
- changed rows: 2,685,663
- clipped rows: 2,012,310
- max abs correction: `0.25 ft`

policy発見に使ったreadoutとKaggle確認値は一致した。これは再現性確認であり、
独立したprospective evidenceとは表記しない。

## コマンドログ

実行したコマンドを時系列で記録します。未実行のコマンドは予定として明記します。

### 実行済み

```bash
.venv/bin/pytest -q experiments/exp415_fold_safe_rmse_prior_bounded_nudge_on_exp264/tests/test_exp415_candidate_rmse_bounded_nudge.py
.venv/bin/python -m py_compile \
  src/candidate_rmse_bounded_nudge.py \
  experiments/exp415_fold_safe_rmse_prior_bounded_nudge_on_exp264/\
exp415_fold_safe_rmse_prior_bounded_nudge_on_exp264_compact_selfcontained_train.py
.venv/bin/ruff check <exp415 implementation and tests>
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb \
  <exp415 compact self-contained train.py>
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test \
  <exp415 compact self-contained train.py>
```

- dedicated synthetic / contract tests: 8件PASS
- 関連tests: exp264 / 414 / 415をrepo rootから31件PASS、
  exp407を自身の実験cwdから9件PASS、合計40件PASS
- py_compile / Ruff: PASS
- Jupytext変換 / round-trip: PASS
- real parent OOF先頭row group 20,000 base rowsの統合probe:
  candidate layout、model_fold、truth reconstruction residual 0、
  declared / applied correction差0、max correction 0.25を確認
- candidate Kaggle package audit:
  31 support files、18 source / 19 packaged cells、埋め込みconfig SHA一致、
  RMSE / hidden入力あり、CPU・internet offを確認してPASS
- canonical予定kernel id / title:
  `kentookumura/exp415-fold-safe-rmse-prior-bounded-nudge-train` /
  `exp415 fold safe rmse prior bounded nudge train`
- 最終実行package SHA:
  `2ef5f85dea2c8b538fa981fc72641adbce69806d8b41bb2246f2c25667832539`
- Jupytext source SHA:
  `8494936082f72c1e7e959b87b61890961acb7cbcfa780d95bf10e8b103c13326`
- 実OOF先頭20,000 rowsのzstdサイズから推定したfull freeze + prediction:
  約`0.243 GiB`
- Kaggle private Dataset
  `kentookumura/exp409-exp264-stage-b-v5-oof-input`をread-only確認:
  `candidate_score_oof.parquet` 398,132,069 bytes、Dataset descriptionのSHAと
  local file SHA
  `9a91b62599278d4e56d57074df4725d4a09391460458b8eccc02dd50af34d48a`
  が一致
- Notebookローカル実行: 未実施

`make test`は1,225件のcollection中、今回触れていない
`test_exp408_hmm_message_rate_basin_audit.py`が作る`numba.__spec__`なしstubと
`test_exp411_predictive_filtered_rate_innovation_destick.py`の
`importlib.util.find_spec("numba")`が衝突してcollection errorになった。
exp415を含む関連40件は独立実行で全PASSしており、exp415由来failureは0。
4実験をrepo rootから一括実行した最終確認では、今回未変更のexp407がrootにある
別実験用`config.yaml`を先に拾い1件FAILした。exp407自身の実験cwdから再実行すると
9/9 PASSし、exp264 / 414 / 415の31件もrepo rootからPASSした。

### 2026-07-27 実行承認

ユーザーから「実行してください」と明示承認を受けた。承認範囲は正規train
Notebookへのcompact候補採用と、Kaggle private CPU zero-booster readout。
GPU、親/control再学習、model fit、PF/HMM/Beam再生成、inference、
submissionは含まない。

実行内訳:

| variant | model config | fold fit | booster | control再学習 | GPU | inference | submission |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

### Kaggle実行結果

```bash
task prepare-kaggle-notebooks \
  EXP=exp415_fold_safe_rmse_prior_bounded_nudge_on_exp264 \
  EXTRA_ARGS="--notebook train \
  --kernel-id kentookumura/exp415-fold-safe-rmse-prior-bounded-nudge-train \
  --title 'exp415 fold safe rmse prior bounded nudge train' \
  --run-on-push --strict"
task push-kaggle-train EXP=exp415_fold_safe_rmse_prior_bounded_nudge_on_exp264
kaggle kernels pull \
  kentookumura/exp415-fold-safe-rmse-prior-bounded-nudge-train \
  -p /tmp/kaggle-pull/exp415-fold-safe-rmse-prior-bounded-nudge-train -m
kaggle kernels logs \
  kentookumura/exp415-fold-safe-rmse-prior-bounded-nudge-train
```

初回は実験名全体を使った57文字slug
`exp415-fold-safe-rmse-prior-bounded-nudge-on-exp264-train`を指定したが、
Kaggle `SaveKernel 400`で実行前に拒否された。直後のmetadata pullも403で、
kernel/version未作成を確認した。Kaggleの50文字title上限に合わせ、科学条件を
変えず、意味を保った47文字の上記canonical id/titleへ短縮して再packageした。

短縮後のcanonical kernelをpushし、version 1（id_no `128717911`）が
`KernelWorkerStatus.COMPLETE`になった。Notebook計測時間は`126.338 sec`。

- technical gate: 15 / 15 PASS
- scientific gate: 6 / 6 PASS
- decision:
  `rmse_prior_bounded_nudge_method_confirmed_on_saved_oof`
- parent / bounded RMSE:
  `8.587004386703422 / 8.563473931791524`
- improvement: `0.023530454911897536 ft`
- fold: 5 / 5改善
- distance bucket: 4 / 4改善
- hidden-like: 2 / 2改善
- nonworse wells: 544 / 773
- worst well `cc08aa63`: `+0.17137908743683994 ft`
- changed / clipped rows: `2,685,663 / 2,012,310`
- correction RMS / max: `0.18978710718863193 / 0.25 ft`

小容量artifactとlogだけを
`kaggle/output/train_v1_small/`へ取得した。gate manifestに記載された
全ダウンロードartifact SHAとの一致、technical/scientific check、truth-read
ledger、risk certificateをread-only監査し、すべてPASSした。98 MBのfreezeと
159 MBのprediction Parquetは丸ごと取得せず、Kaggle出力内で計算・保存された
SHAをreproducibility manifestとgateから記録した。

## 変更点

- `src/candidate_rmse_bounded_nudge.py`に再利用可能なpolicy、truth reconstruction、
  Minkowski risk certificateを追加した。
- compact self-contained Jupytext候補にtruth-free freezeとevaluationを二相実装した。
- 入力、candidate/fold/row/well、truth read、correction、全scope gateをfail closedにした。
- package preflightでrepo rootの別実験用`config.yaml`を先に読むpath優先順位バグを
  検出した。exp415実験ディレクトリを先に探索するよう修正し、専用tests、
  Jupytext、strict validationを再実行してから最終packageを作成した。

## 再現性メモ

- seed policy: RNGなし、保存row orderと宣言candidate orderで決定論的
- stochastic components: なし
- CPU/GPU runtime: Kaggle private CPU `126.338 sec`、GPU false、internet off
- model / booster / candidate生成: 0 / 0 / 0
- parent OOF SHA:
  `9a91b62599278d4e56d57074df4725d4a09391460458b8eccc02dd50af34d48a`
- candidate RMSE table SHA:
  `ecf3e93b161e2a173ed3cadbf69cc369d367f38d939d8463be1624e4c851922b`
- hidden assignment SHA:
  `5f9ac9fac6bb3725a7c613f09856a85bdf73b8206fd2edf1b79e8eaa9bca6597`
- feature schema logical SHA:
  `aaef4ffdd90667893b099b76a52f1957b22197aea9cee5e5b57bc81048ddd3a4`
- Kaggle kernel id / version / id_no:
  `kentookumura/exp415-fold-safe-rmse-prior-bounded-nudge-train` /
  `1` / `128717911`
- 実行config SHA:
  `21450bcff8569dd4a2ed84c872af13a01aed488634f61466a130d8c5d22bca90`
- canonical Notebook SHA:
  `00a95cb16856c47fc2929c2f37411169f26edb938a6fe5135c835988815daa3f`
- freeze SHA:
  `0dd3f55991969da433d65391d5f94efaeff61a615608635767962866a0971aec`
- prediction SHA:
  `cb820ae7c499db8cc6aad37d5665b08e517c88d503a5176b27b03c1b45035f61`
- all-scope metrics SHA:
  `469e3985a4217e0621eb3d5386395aad0b262f7386909fbad442362f723177a8`
- gate / risk certificate SHA:
  `30bc689e5e4fe178735c150adac1879226541ee68560aeb5bfb5bff51d8054a0` /
  `25ed51b33bfb2123316075b655e43bc832682448abb152de771f8570fbc148ad`
- model SHA / submission SHA: 対象外

## 次のアクション / 完了判断

全technical / scientific gateを満たしたため、候補別RMSEをfold-safe additive
priorとrisk-bounded TVT nudgeへ利用する方法を保存OOF診断上で確立した。
exp415からcurrent-test inference、route anchor更新、submissionへは進まない。
