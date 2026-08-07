# exp442_symmetric_broad_jump_rate_transition_hmm セッションノート

## 目的

exp209の局所rate遷移を保ちながら、低確率の対称broad branchで急変時の到達経路を作る。

## 現在の状態

- Route: `pf_beam`
- 状態: Kaggle private CPU Stage 0 version 1完了、`stage0_fail_closed`
- 親: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- 優先度: P2
- 実装承認: 2026-07-29のユーザー依頼「exp442を実装してください」
- 正規train Notebook採用 / package / Stage 0: 完了
- Stage 1 / inference / submission: gate FAILにより不可
- CV / LB: なし

## 2026-07-30 exp441先行条件判定

- exp441 technical: 16 / 17 PASS。runtime projection
  `38,217.120 > 30,600 sec`でFAIL。
- exp441 mechanism: 2 / 7 PASS。under-response share削減
  `0.022974 < 0.05`、forward/persistent SSEは悪化、改善
  `8 / 16 wells`・`1 / 5 folds`。
- control safety 2件はPASSしたが、technical成立かつ方向正・量不足という
  exp442のAND先行条件は不成立。
- この時点では現行exp442の正規Notebook採用、package、Stage 0、Stage 1、
  inference、submissionを行わない判断だった。後続の独立仮説再設計で
  Stage 0までの扱いを更新した。

## 2026-07-30 独立仮説への再設計とStage 0承認

- ユーザー依頼:
  「steeringと検証契約を更新してから実行してください」。
- exp442はexp209 local kernelを99%維持し、固定1% broad branchだけを加える
  独立defensive mixture仮説として評価する。
- exp441はkernel全体をfull-support OUへ置換した別仮説であり、terminal FAILを
  negative contextとして保持する。positive evidence、実行前提、gate救済には使わない。
- `jump_weight=0.01`、`broad_sigma_rate=0.02`、fixed32、technical/mechanism
  AND gateはexp441結果を見る前の値から変更しない。
- 承認範囲: 正規train Notebook採用、Kaggle private CPU package、Stage 0。
- 実行量:
  - scientific variant: 1
  - candidate exact-HMM well-runs: 32
  - reporting folds: 5
  - saved exp209 control rerun: 0
  - LightGBM config / trained fold / booster / fitted model / PF / Beam / GPU:
    `0 / 0 / 0 / 0 / 0 / 0 / 0`
- Stage 1、inference、submissionは対象外。

### 正規train Notebook採用と静的検証

```bash
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb \
  --output experiments/exp442_symmetric_broad_jump_rate_transition_hmm/\
exp442_symmetric_broad_jump_rate_transition_hmm_train.ipynb \
  experiments/exp442_symmetric_broad_jump_rate_transition_hmm/\
exp442_symmetric_broad_jump_rate_transition_hmm_compact_selfcontained_train.py
```

- compact / canonical train: 24 cells、3,407 source lines。
- cell source SHA:
  `b87fa7a988f9d5eb78e7613550435f0cebee251d71ecdc626082825d6532d494`。
- canonical train Notebook file SHA:
  `1f8b124d8d185b9c03d234a9282d552f8426b1d0967ccb1c594994a0433e8610`。
- canonical inference Notebook: 未変更。
- Stage 0承認に合わせ、inference guardはtrain採用/package/Stage 0を許容する一方、
  Stage 1 / inference / submissionをfail-closedに維持するよう更新した。
- 専用pytest: `12 passed`。
- Jupytext train / inference `--test`: PASS。
- py_compile / Ruff / strict experiment validation / template validation: PASS。

### Kaggle package前検証

canonical kernel IDの事前pull:

```bash
kaggle kernels pull \
  kentookumura/exp442-symmetric-broad-jump-rate-transition-hmm-train \
  -p /tmp/exp442-kernel-preflight-v1 -m
```

- `GetKernel` 403で既存kernelは確認できなかった。
- 別slugは作らず、計画済みcanonical IDへ初回pushする。

package:

```bash
make prepare-kaggle-notebooks \
  EXP=exp442_symmetric_broad_jump_rate_transition_hmm \
  EXTRA_ARGS="--notebook train \
  --kernel-id kentookumura/exp442-symmetric-broad-jump-rate-transition-hmm-train \
  --title 'exp442 symmetric broad jump rate transition hmm train' \
  --run-on-push --strict --no-src"
```

- metadata: private `true`、CPU、TPU/GPU/internet `false`。
- competition source: `rogii-wellbore-geology-prediction`。
- kernel sources:
  `kentookumura/exp209-joint-exact-parity-train`、
  `kentookumura/exp408-hmm-message-rate-basin-audit-train`。
- bootstrap: 8 files。config、2 compact source、settings、project、
  fixed32 / persistent episode / exp408 causeの3 assetを含み、展開後SHAは全PASS。
- package configは正のconfigとbyte一致。
- package Notebook SHA:
  `87d977d05484f03140b2fa9299706cce5c8e2a501a0b02e58a04ef32a2eb14aa`。
- kernel metadata SHA:
  `0891869b622415e492d40265660908931cc496b622e4dc06f46e802b42fe1815`。

### 初回push 400とcanonical slug短縮

- 初回push:

```bash
make push-kaggle-train EXP=exp442_symmetric_broad_jump_rate_transition_hmm
```

- 結果: `SaveKernel 400 Bad Request`。Kaggle実行は開始していない。
- 初回slug
  `exp442-symmetric-broad-jump-rate-transition-hmm-train`は53文字で、
  IDとtitle由来slug自体は一致していた。
- Kaggle kernel title/slugの長さ制約に合わせ、科学内容・実験番号を変えず
  `transition`だけを`trans`へ短縮する。
- 採用canonical ID / title:
  `kentookumura/exp442-symmetric-broad-jump-rate-trans-hmm-train` /
  `exp442 symmetric broad jump rate trans hmm train`（48文字）。
- 別の実験や科学variantは作らず、同じexp442 packageを再生成してpushする。

### Kaggle Stage 0 version 1

- kernel:
  `kentookumura/exp442-symmetric-broad-jump-rate-trans-hmm-train`
- URL:
  `https://www.kaggle.com/code/kentookumura/exp442-symmetric-broad-jump-rate-trans-hmm-train`
- push: `Kernel version 1 successfully pushed`
- target: private CPU、internet disabled。
- scientific candidate 1、candidate HMM 32 wells、saved control rerun 0。
- model / booster / PF / Beam / GPU: 全て0。
- Stage 1 / inference / submission: 未実行・未承認。

## コマンドログ

### 2026-07-29 design-only作成

```bash
make new-steering EXP=exp442_symmetric_broad_jump_rate_transition_hmm
make new-exp EXP=exp442_symmetric_broad_jump_rate_transition_hmm
```

### 2026-07-29 compact self-contained実装

作成:

- `exp442_symmetric_broad_jump_rate_transition_hmm_compact_selfcontained_train.py`
- `exp442_symmetric_broad_jump_rate_transition_hmm_compact_selfcontained_train.ipynb`
- `exp442_symmetric_broad_jump_rate_transition_hmm_compact_selfcontained_inference.py`
- `exp442_symmetric_broad_jump_rate_transition_hmm_compact_selfcontained_inference.ipynb`
- `tests/test_exp442_symmetric_broad_jump_rate_transition_hmm.py`

検証:

```bash
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb \
  experiments/exp442_symmetric_broad_jump_rate_transition_hmm/\
exp442_symmetric_broad_jump_rate_transition_hmm_compact_selfcontained_train.py \
  experiments/exp442_symmetric_broad_jump_rate_transition_hmm/\
exp442_symmetric_broad_jump_rate_transition_hmm_compact_selfcontained_inference.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test \
  experiments/exp442_symmetric_broad_jump_rate_transition_hmm/\
exp442_symmetric_broad_jump_rate_transition_hmm_compact_selfcontained_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test \
  experiments/exp442_symmetric_broad_jump_rate_transition_hmm/\
exp442_symmetric_broad_jump_rate_transition_hmm_compact_selfcontained_inference.py
.venv/bin/python -m py_compile \
  experiments/exp442_symmetric_broad_jump_rate_transition_hmm/\
exp442_symmetric_broad_jump_rate_transition_hmm_compact_selfcontained_train.py \
  experiments/exp442_symmetric_broad_jump_rate_transition_hmm/\
exp442_symmetric_broad_jump_rate_transition_hmm_compact_selfcontained_inference.py
.venv/bin/ruff check \
  experiments/exp442_symmetric_broad_jump_rate_transition_hmm/\
exp442_symmetric_broad_jump_rate_transition_hmm_compact_selfcontained_train.py \
  experiments/exp442_symmetric_broad_jump_rate_transition_hmm/\
exp442_symmetric_broad_jump_rate_transition_hmm_compact_selfcontained_inference.py \
  tests/test_exp442_symmetric_broad_jump_rate_transition_hmm.py
.venv/bin/pytest -q \
  tests/test_exp442_symmetric_broad_jump_rate_transition_hmm.py
.venv/bin/pytest -q \
  tests/test_exp408_hmm_message_rate_basin_audit.py \
  tests/test_exp411_predictive_filtered_rate_innovation_destick.py \
  tests/test_exp440_ambiguity_gated_predictive_prior_hmm.py \
  tests/test_exp442_symmetric_broad_jump_rate_transition_hmm.py
make validate-exp EXP=exp442_symmetric_broad_jump_rate_transition_hmm
make validate-template
```

結果:

- 専用pytest: `12 passed`
- Jupytext `--test`: train / inferenceともPASS
- `py_compile`: PASS
- Ruff全選択ルール: PASS
- strict experiment validation: PASS
- template validation: PASS
- exp408/411/440/442関連pytest: `50 passed, 1 failed`
  - failureは既存exp440 config statusが
    `kaggle_v1_running_stage0`へ進んでいる一方、testが
    `stage0_authorized_pending_run`を期待している状態差。
  - exp442のcollection / assertion failureではなく、対象外のexp440は変更しない。
- 親/参照compact比較:
  - exp411: 9章、2,255行
  - exp440: 10章、2,576行
  - exp441: 10章、3,040行
  - exp442: 10章、3,468行
  - exp442はinput、local/broad/mixture kernel、exact HMM、branch
    responsibility、truth-late、gate、metrics/orchestrationをNotebook上で追える。
- `__file__`: compact train/inferenceとも0件。
- 正規`*_train.ipynb` / `*_inference.ipynb`: 未変更。

## 設計契約

- `0.99 * parent + 0.01 * broad Gaussian(sigma=0.02)`の1候補だけ。
- broad branchはtarget-free、両方向対称、全bin CDF積分。
- position/emission/prior/state/readoutは親固定。
- Stage 0は32 candidate HMM well-runs、parent rerun 0。
- Stage 1は全gate PASS・別承認時だけ773 candidate HMM well-runs。
- LightGBM config / fold / booster / model / PF / Beam / GPUは全て0。
- weight、sigma、方向trigger、duration、reset、gateのsame-OOF救済は禁止。

## 実装内容

- exp209と同じTVT/rate state、per-well rate grid、destination-rate position
  transition、Gaussian GR emission、prior、prefix sigma、posterior mean/stdを
  self-containedで実装した。
- local branchはexp209の3状態Euler kernelを再現し、境界外向きmassをedgeへ
  再配分しない。
- broad branchはparent Euler conditional meanを中心に`sigma=0.02`のGaussianを
  finite 41 rate-bin Voronoi cellへCDF積分し、support外massを捨てる。
- `0.99 * local + 0.01 * broad`を確率空間で厳密混合し、jump samplingはしない。
- smoothed transition edge measureから、broad branch responsibility、
  non-adjacent broad edge mass、responsibility加重signed rate deltaを計算する。
- future rate方向はexp411と同じpast32 / next32 physical interval rate中央値差。
  agreementとfold判定はnon-adjacent posterior edge massで加重する。
- local parity、mixture分解、broad in-support mass、centered symmetry、
  branch responsibility brute-forceをsynthetic contractとして実装した。
- jump=0の独立exp209正規smoother照合はprediction最大差約`0.0011 ft`。
  local kernel自体は`1e-12`以内で一致するため演算表現差として保持し、
  Stage 0では保存exp209 controlとの差へ含めて安全性を判定する。
- fixed32 manifestはfreeze前に`well`列だけを読み、role/foldは全32 wellの
  kernel/responsibility/prediction/diagnostic SHA freeze後に再読する。
- truth、persistent episode、exp408 causeも全freeze後にだけ読む。
- Stage 0のtechnical/mechanism gateはconfig全キーのANDで判定し、
  1件でもFAILならno-rescueで閉じる。

## 再現性メモ

- jumpはsamplingせず確率的に厳密周辺化するためRNGなし。
- local/broad/mixture kernelとbranch responsibilityのSHAを保存する。
- truth/role/fold/episodeはprediction/diagnostic freeze後だけjoinする。
- 初回runはdeterministic anchorとしない。

## 実行量契約

- Stage 0:
  - scientific candidate: 1
  - candidate HMM well-runs: 32
  - parent control rerun: 0
  - reporting folds: 5
- Stage 1（Stage 0全PASS・別承認時のみ）:
  - scientific candidate: 1
  - candidate HMM well-runs: 773
  - parent control rerun: 0
- LightGBM config / trained fold / booster / fitted model / PF / Beam / GPU:
  `0 / 0 / 0 / 0 / 0 / 0 / 0`

## 再現性メモ（実装後）

- seed policy: RNGなし、well / row / position / source rate / destination rate /
  branch / reduction順固定。
- stochastic components: なし。
- CPU/GPU runtime: CPU-only予定、GPU 0、internet disabled。
- Kaggle kernel id / version: version 1完了。slugは
  `kentookumura/exp442-symmetric-broad-jump-rate-trans-hmm-train`。
- input SHA: fixed32、exp209 saved control、exp408 episode/causeをconfig固定。
- 実rate log-kernel content SHA / responsibility / prediction / diagnostic SHA:
  実行時に保存。
- model manifest / model SHA / submission SHA: 非該当。
- 初回runをdeterministic anchorとせず、独立rerun SHA一致後だけ再判定する。

## 2026-07-30 Stage 0完了と判定

- Kaggle worker status: `COMPLETE`
- kernel version / id_no: `1` / `129101211`
- scientific contract SHA:
  `cd97572dc08d68e4a2018b27e0309cde3e129c3e35814953ec0f237048c60752`
- 実行: 1 scientific candidate、32 HMM wells、156,088 rows、
  156,056 transition rows、5 reporting folds。
- elapsed: `9,190.989644 sec`、peak RSS: `1.191029 GiB`。
- 773 wells runtime投影: `222,019.843582 sec`。
- parent control rerun / LightGBM config / trained fold / booster /
  fitted model / PF / Beam / GPU: 全て0。
- fixed32はmechanism-onlyであり、CV / Public LB / Private LBはいずれもない。

### Technical gate

- 14 / 15 PASS。
- local parity、broad mass、mixture、brute-force responsibility、
  prediction/diagnostic/transition SHA readback、rows/wells、finite、
  role/fold、truth-late、normalization、RSSはPASS。
- full runtime投影だけが
  `222,019.843582 > 30,600 sec`でFAIL。

### Mechanism gate

- 4 / 9 PASS。
- PASS:
  - non-adjacent posterior edge mass:
    `0.006845573 >= 0.001`
  - direction positive folds: `5 / 5 >= 4 / 5`
  - matched-control pooled RMSE delta:
    `-0.155413848 <= +0.02 ft`
  - matched-control by-well delta p95:
    `+0.069364249 <= +0.25 ft`
- FAIL:
  - future rate direction agreement:
    `0.529731633 < 0.60`
  - forward-cause episode SSE reduction:
    `0.002430606 < 0.10`
  - persistent episode SSE reduction:
    `-0.044384932 < 0.05`
  - persistent improved wells: `9 / 16 < 10 / 16`
  - persistent improving folds: `2 / 5 < 4 / 5`
- persistent fold別削減率:
  `[-0.147628, +0.003785, -0.012156, -1.184506, +0.131886]`。

### 解釈

- pooled broad branch responsibilityは`0.009766954`、non-adjacent massは
  `0.006845573`であり、候補branchは実際にposteriorで使われた。
- 集計control safetyは保ったが、future directionは全foldでほぼcoin-flipに近く、
  persistent episodeは全体で4.44%悪化した。特にfold 3の悪化が大きい。
- 対称escape supportを足すだけでは、正しい方向と持続区間を選ぶ情報を補えない。
- runtimeと主要mechanismの双方がFAILしたため、固定fail actionどおり
  `stage0_fail_closed`とする。
- 初回runをdeterministic anchorとは扱わないが、FAIL候補の独立rerunは行わない。

### Artifact SHA

- Kaggle `metrics.json`:
  `92d09d57c9afc0ff7397a54d231424ddfbe172478c1e91381ddab946e3ed53c7`
- prediction logical:
  `01fb64c820f4f68d8a0c8d7a8891f5ceedad7c8a12d647b79206fb0f2acfe59e`
- target-free diagnostic logical:
  `cd6809a1bbaa07fa30f2989f5bf317a2d8be1fad8ec6f29b07a43b65b72330fd`
- transition diagnostic logical:
  `2439708574644cc73f7f7647d04686a61566e67f3cf2726d144350c12c19bf5f`
- well metrics:
  `608819968361d380d11091eb7f90e39dfc94aa2b6c9324d17e346de8ab34bae0`
- episode truth-late:
  `abdf815cf787f95a7be95b185369c8ffd8ec3bb3e5983d55b3aad7846fa3de6e`

## 次のアクション

1. exp442を完了済みとしてbacklogから削除する。
2. rerun、Stage 1、inference、submissionは行わない。
3. weight / sigma / trigger / emission / grid / gateのsame-OOF救済は行わない。
4. exp444/exp446などtrend/persistenceを明示する既存仮説は独立候補のまま扱い、
   exp442 FAILをpositive evidenceや実行承認には使わない。
