# exp259_coordinate_equivariance_path_warp_augmentation セッションノート

## 目的

`coordinate_equivariance_path_warp_augmentation` backlogを実験化する。厳密な座標対称性と
近似的な坑井・地質path変換を分け、モデル学習前にinverse consistency、official-start連続性、
typewell GR再sample、trajectory/prefix/FFT/DWT再生成、real-train分布guardを固定する。

## 現在の状態

- Route: `ensemble`
- 状態: Kaggle transform audit / exact datum training version 1完了、train-side guard不通過
- stage: `train_exact_datum_after_transform_audit`
- parent: `exp251_raw_test_safe_dual_objective_candidate_ranker`
- CV / LB: `8.427125551` / なし（Kaggle ID `127436131`、status `COMPLETE`）
- inference / submission: disabled

## 実行前コストガード

- active variant: 1（`exact_tvt_datum_shift`）
- model / LightGBM config: 2 objectives
- fold: 5
- booster: 10 CPU
- parent/control再学習: なし
- PF/Beam/HMM/dense/geometry/exp218再生成・再学習: なし
- runtime: Kaggle CPU、GPU false、internet false
- 全773 wellsをclean OOF評価し、stable SHAで固定した25% wellsのouter-train rowsだけに
  datum-shift viewを追加する。
- exp251の295列clean controlは別kernelの結果を固定参照し、同runでは学習しない。

## 実装した変換

### 厳密変換

- `heel_center_translation`: heel XYを原点へ平行移動。
- `lateral_reflection`: heel→known-prefix末尾のheading軸に対してcross-trackを反転。
- `yaw_rotation`: heel中心に固定angle gridからyaw回転。
- `tvt_datum_shift`: horizontal true/input/candidate TVTとtypewell TVTを同じ量だけshift。

各変換はinverse後の全数値列最大絶対誤差と、local geometry・GR spectral summaryの相対差を
guardする。

### 近似変換

- `md_stretch`: official-start以降のMD距離だけを固定factorで伸縮。
- `tvt_shear`: official-startからtail末端まで線形TVT deltaを付与。
- `xy_plane_tilt`: `slope_x*deltaX+slope_y*deltaY`をTVTへ付与。
- `low_frequency_spline_warp`: anchor 0のsmoothstep control curveをTVTへ付与。
- `smooth_xyz_control_perturbation`: anchor 0のsmooth XYZ control-point変位を付与。

known prefixは変更せず、近似変換後のevaluation tail GRは同じtypewellのtransformed TVTから
線形補間する。MD/XYZ微分、slope、curvature、along/cross-track、prefix統計、FFT band energy、
3-level Haar-DWT energyを再生成する。

## リーク・分布ガード

- transform parameterは`sha256(seed, well, transform_kind, view_slot, parameter_name)`で決める。
- Python `hash()`、global RNG、target error、oracle candidate、hidden-like role、Public LBを使わない。
- real train summaryのq0.005〜q0.995をrelative margin 0.25で広げたenvelopeを固定する。
- approximate viewは非単調MD、anchor不連続、typewell coverage 0.95未満、slope/curvature分布外をrejectする。
- TVT slopeはsynthetic分布診断だけに使い、model featureには使わない。
- 後続outer-fold学習ではouter-train wellsだけでenvelopeをfitし、outer-validはclean viewだけにする。
- absolute spatial/KNN branchをlocal geometryのexact equivarianceとして扱わない。

## 再現性メモ

- `docs/06_reproducibility.md`を2026-07-15に確認した。
- wellとtransformはsortし、parameterはwell-local stable keyから決定する。
- global RNG、Python `hash()`、thread順依存はない。
- raw input file SHA、config SHA、envelope SHA、manifest raw/decompressed SHA、summary SHAを保存する。
- gzip manifest/previewは`mtime=0`で保存し、decompressed content SHAを主証拠にする。
- model/prediction/submissionは生成しないためSHA対象外。deterministic submission anchorとは扱わない。
- Kaggle prepare後にembedded configと`src` helper、CPU/internet metadataを確認する。

## 2026-07-15 steering / scaffold

    make new-steering EXP=exp258_coordinate_equivariance_path_warp_augmentation
    make new-exp EXP=exp258_coordinate_equivariance_path_warp_augmentation SOURCE=experiments/exp251_raw_test_safe_dual_objective_candidate_ranker

- 作業開始時はexp257まで使用済みだったためexp258をscaffoldした。
- 実装中に別の`exp258_gr_residual_noise_transplant_augmentation`が同じworkspaceへ追加されたことを
  reviewで検出した。既存側には触れず、本実験のdirectory、steering、config、notebook、package
  slugを次の空き番号exp259へ移した。
- exp251のraw-test-safe candidate bank/feature contractを将来の学習参照親とした。
- 親のcopied notebook/helperは使用せず、reusable engineを`src/`、上位auditをcompact notebookへ実装した。

## 2026-07-15 実装

- strict/approximate transform、stable parameter selection、exact inverseを実装した。
- typewell GR interpolation、candidate TVT equivariant delta hook、local geometry、FFT、Haar-DWT再生成を実装した。
- real-well distribution envelopeとaccept/reject reasonを実装した。
- compact Jupytext train notebookへconfig/input/transform/guard/metrics/artifact/SHA orchestrationを展開した。
- inference notebookは明示停止し、submissionを生成しない。

## 2026-07-15 静的・合成契約検証

    .venv/bin/python -m py_compile src/coordinate_path_augmentation.py experiments/exp259_coordinate_equivariance_path_warp_augmentation/*.py tests/test_coordinate_path_augmentation.py
    .venv/bin/ruff check src/coordinate_path_augmentation.py experiments/exp259_coordinate_equivariance_path_warp_augmentation/*.py tests/test_coordinate_path_augmentation.py
    .venv/bin/pytest -q tests/test_coordinate_path_augmentation.py
    JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp259_coordinate_equivariance_path_warp_augmentation/exp259_coordinate_equivariance_path_warp_augmentation_compact_selfcontained_train.py
    JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp259_coordinate_equivariance_path_warp_augmentation/exp259_coordinate_equivariance_path_warp_augmentation_compact_selfcontained_inference.py
    make validate-exp EXP=exp259_coordinate_equivariance_path_warp_augmentation
    make validate-template

- py_compile: PASS。
- Ruff: PASS。
- exp259 unit test: 13件PASS。
- Jupytext convert/test: train / inference PASS。
- strict validate-exp / validate-template: PASS。
- notebook sourceの`__file__`参照なし。
- full pytest: 45件PASS / 1件FAIL。FAILは既存exp251 configの現在stage
  `train_after_feature_audit`に対し既存testが`feature_audit_only`を期待する不一致で、exp258変更前からの
  repository stateに属する。exp259 testは全PASSのため、ユーザーのexp251状態を変更していない。

## Notebook構成比較

- 親exp251 train source: 7章 / 約249行、同一exp helper importを使う構成。
- exp259 compact train source: 7章。reusable数値engineだけ`src/`へ置き、実験契約、入力確認、
  clean envelope、transform loop、guard、保存、SHAはnotebook cell上に展開した。
- notebook sourceで`__file__`を使用しない。

## 2026-07-15 Kaggle package検証

    make prepare-kaggle-notebooks EXP=exp259_coordinate_equivariance_path_warp_augmentation EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp259-coordinate-equivariance-warp-audit-train --title 'exp259 coordinate equivariance warp audit train' --run-on-push --strict"

- canonical kernel id: `kentookumura/exp259-coordinate-equivariance-warp-audit-train`。
- metadata: notebook / private / CPU、`enable_gpu=false`、`enable_internet=false`、
  `run_on_push=true`、competition source `rogii-wellbore-geology-prediction`。
- root configとpackage configはbyte一致。
- bootstrap manifestに`src/coordinate_path_augmentation.py`を含み、embedded configは
  `transform_audit_only`、`planned_boosters: 0`。
- `scripts/prepare_kaggle_notebooks.py`のsupport bundle収集が`src/__pycache__`と`.pyc`まで
  含めていたためfilterを追加し、再生成packageにbytecodeが含まれないことを確認した。
- 初回packageのid/titleはともに`exp259-coordinate-equivariance-path-warp-augmentation-train`
  由来でslug一致していたが59文字だった。2026-07-15の初回pushはKaggle
  `SaveKernel` HTTP 400で拒否され、kernel実行は開始しなかった。
- 実験番号は変えず、意味を保つ47文字のcanonical id/title
  `exp259-coordinate-equivariance-warp-audit-train`へ同時に短縮し、packageを再生成した。
  root configとpackage configのbyte一致、CPU/internet off、`0/0/0/0`契約を再確認した。

## 2026-07-15〜16 Kaggle transform audit version 1

    kaggle kernels push -p experiments/exp259_coordinate_equivariance_path_warp_augmentation/kaggle/train
    kaggle kernels pull kentookumura/exp259-coordinate-equivariance-warp-audit-train -p /tmp/kaggle-pull/exp259-coordinate-equivariance-warp-audit-train-v1 -m
    kaggle kernels logs kentookumura/exp259-coordinate-equivariance-warp-audit-train
    kaggle kernels status kentookumura/exp259-coordinate-equivariance-warp-audit-train
    kaggle kernels output kentookumura/exp259-coordinate-equivariance-warp-audit-train -p experiments/exp259_coordinate_equivariance_path_warp_augmentation/kaggle/output/train_v1

- canonical kernel: `kentookumura/exp259-coordinate-equivariance-warp-audit-train`。
- Kaggle kernel id / version / status: `127328846` / `1` / `COMPLETE`。
- CPU、GPU false、internet false、parent/control再学習なし。
- 実行契約はactive variant / model config / fold / booster `0 / 0 / 0 / 0`。
- 773 wells × 9 transforms × 1 slot = 6,957 views。採択6,129、拒否828、
  manifestのwell×transform×slot重複0。
- audit summary出力は約208.95秒、最終log時刻は約218.80秒。

### 厳密変換

- heel-centered translation、lateral reflection、yaw rotation、TVT datum shiftは各773 / 773採択。
- inverse最大絶対誤差: `9.313225746154785e-10`。
- local metric相対差最大: `0.0`。
- `inverse_tolerance=1e-7`を通過した。ただしabsolute spatial / KNN branchは契約対象外。

### 近似変換

| transform | accepted / views | accept rate |
|---|---:|---:|
| TVT shear | 770 / 773 | 99.61% |
| XY plane tilt | 768 / 773 | 99.35% |
| low-frequency spline warp | 766 / 773 | 99.09% |
| smooth XYZ control perturbation | 733 / 773 | 94.83% |
| MD stretch | 0 / 773 | 0.00% |

- `md_stretch`はfactor 0.97の395 views、1.03の378 viewsが全件reject。
- 全773 viewsで`xy_slope_q95`、`xy_curvature_q95`、`z_curvature_abs_q95`が
  real-train envelope外。0.97側は`md_step_q01`、1.03側は`md_step_q99`も外れた。
- MDだけを伸縮してXYZを据え置く現在の定義はtrajectoryのMD微分量と不整合なので、
  現仕様のままtraining viewに採用しない。

### 生成物とSHA確認

- output: `kaggle/output/train_v1/artifacts/`。
- executed config SHA256:
  `4460f35ee0ea018790f84e7ddc982e995f1b2ce872a808f35837bd6616dfe2d2`。
- real-well summary SHA256:
  `f7d8c1c586941a8d9895adb84b1c1033738852b01fe3c9ecd2812d637207852f`。
- envelope raw/content SHA256:
  `478068423fbb7678a5a67aa44836e5740c7bf7938cdcb5e2deaa0f9002b8bcd1` /
  `583a3d896a1a485135295458f2375c2740da3167d9391760a2c8b1d6d9574c1e`。
- manifest raw/decompressed content SHA256:
  `f3969121e959d2bec96b528be66f3d799b5daf4d182407fc4eb53e1e67757af0` /
  `0afcc19742ca20b2dbd0ef6b48b78ca7657fc1df5eb89eb0476ba5efba6276bd`。
- transform summary SHA256:
  `353ec69c03ff005437438689e7576c291361f3dde1323466313767a950779c89`。
- preview raw/decompressed content SHA256:
  `b4a97256441fcdc434086b411413d66add4883e008095aac6b8fac8a5b37b5a7` /
  `ac469d091e9a639ea5a0b85950679acb7a4d0cacfa0c079dd1acfe4458de0956`。
- 上記をdownload後に再計算し、Kaggle summary記載値とすべて一致した。
- model / prediction / submissionは生成していないため各SHAは対象外。

## 監査完了時点での次アクション（履歴）

1. `md_stretch`を後続学習から除外するか、trajectoryとMDの整合性を保つ変換へ再設計するか決める。
2. その判断後、clean controlと採択済みtransformを比較するranker学習variant、config、fold、
   booster数を設計する。
3. GPU/control再学習コストを提示し、ユーザーの明示承認後にだけ同じexp259へ学習stageを追加する。

## 2026-07-16 full-well exact datum学習契約

- ユーザー指示により`md_stretch`を除外し、full-well学習へ進む。
- 全773 wellsをclean GroupKFold 5-fold OOFに含める。outer-validはclean viewだけをscoreする。
- 学習variantは`exact_tvt_datum_shift` 1件。2 objectives × 5 folds = 10 CPU boosters。
- clean training rowsは保持し、SHA256で固定した全wellの25%に`-40/-20/+20/+40 ft`の
  exact datum viewをouter-trainだけ追加する。
- exp251 version 3 feature auditのselected 295列schemaをSHA
  `7a9217d6ed96f5f1e569dbefff2a1fb17751405d6ddccae5e5d9dbf12da787ae`で固定する。
- exp251 version 4 clean modelは別kernelで並列学習し、exp259ではcontrol/parentを再学習しない。
  control metricsがexp259開始時に未完了なら比較をpendingとしてOOFを保存する。
- `md_stretch`を含む近似5変換は無効。監査acceptだけではPF/HMM/geometry candidateと
  absolute spatial priorの再生成契約にならないため、近似raw pathだけを変えたrowは作らない。
- heel translation/reflection/yawはselected 295列にraw XY/local geometryがなくfeature-identicalな
  duplicateになるため学習無効。strict inverse監査だけを履歴として保持する。
- 既存audit notebookを上書きせず、Jupytext source/notebook
  `exp259_coordinate_equivariance_path_warp_augmentation_train_variant0.py/.ipynb`を追加する。
- Kaggle package/kernelはaudit canonical slugと分離し、
  `kentookumura/exp259-exact-datum-fullwell-train`を使う。

### 静的検証とKaggle version 1投入

- `py_compile`: exp259 training source、parent integration module、exact datum helper PASS。
- Ruff `F821`: PASS。
- unit tests: coordinate transform、exact datum long-view、exp259 compute contractの17 tests PASS。
- Jupytext convert/test: `*_train_variant0.py/.ipynb` PASS。
- strict `make validate-exp`: PASS。
- package: `kaggle/train_variant0`。root/package config byte一致、CPU、GPU false、internet false、
  competition source 1、kernel source 10、bootstrapにparent exp251 sourceと`src` helperを含む。
- push: `kentookumura/exp259-exact-datum-fullwell-train` version 1、Kaggle id `127436131`。
- push直後status: `RUNNING`。初回logsは空で、即時errorは未確認。継続監視は行わない。
- exp251 clean controlも同時点で`RUNNING`。exp259はcompleted v3 schema SHAを使うため学習は独立し、
  control結果はsaved metricsの後比較に限定する。

### 2026-07-16 最終整合確認

- 実行中packageのconfig SHA256とroot config SHA256はともに
  `260ec5dc70f8865d95804b98143f1072c7d6c37b2781fed07b497b64365d4d17`でbyte一致。
- pinned 295列schema SHA256は
  `7a9217d6ed96f5f1e569dbefff2a1fb17751405d6ddccae5e5d9dbf12da787ae`でconfigと一致。
- `py_compile`、Ruff、focused unit tests 17件、Jupytext audit/train/inference round-trip、
  strict `validate-exp`、`validate-template`はすべてPASS。
- 学習中exp251の現行configに合わせて旧feature-audit-only testを更新し、repository全体pytestは
  50件すべてPASS。exp259 focused tests 17件も全件PASS。
- 再確認時点でexp259 ID `127436131`とexp251 clean controlはともに`RUNNING`。

## 2026-07-16 full-well exact datum学習version 1完了

- `kentookumura/exp259-exact-datum-fullwell-train` version 1 / ID `127436131`は
  `COMPLETE`。runtimeは`4776.872522 sec`（約79.6分）。
- 実行契約どおり1 variant × 2 objectives × 5 folds = 10 CPU boosters。
  exp251 clean controlとparentは再学習していない。
- clean OOFは3,783,989 rows / 773 wells、11 candidates、selected 295 features。
- SHA256固定の193 wells（24.97%）へexact TVT datum viewをouter-trainだけ追加。
  `md_stretch`を含む近似5変換とfeature-identicalなrigid XY 3変換は学習していない。
- 5/5 foldsでcandidate error、within10 label、相対特徴288列のequivariance guardがPASS。
  absolute TVT特徴7列の最大shift誤差は`0.0`、toleranceは`1e-5`。
- fold別augmented wellsは`158 / 147 / 156 / 152 / 159`、augmented long rowsは
  `166,628 / 155,441 / 160,820 / 158,202 / 164,846`。

### exp259単体metrics

| mode | RMSE | MAE | within10 | oracle accuracy | switches |
|---|---:|---:|---:|---:|---:|
| probability rowwise | 8.556052667 | 5.164365596 | 0.852288947 | 0.168975121 | 1,003,205 |
| expected-error rowwise | 8.532211478 | 5.041134407 | 0.847643849 | 0.189872381 | 842,877 |
| expected-error fixed Viterbi | 8.427125551 | 4.982795759 | 0.849401254 | 0.199173941 | 7,053 |

candidate AUC / logloss / Brier / expected-error MAEは
`0.924427649 / 0.327708967 / 0.103715130 / 4.586597201`。

## 2026-07-16 exp251 clean controlとの事後比較

exp259完了時点ではexp251が未完了だったためKaggle summaryのcontrol comparisonはpendingだった。
その後に完了したexp251 version 4の保存済みmetrics/by-wellを取得し、同一295列schema、
`expected_error_fixed_viterbi`同士をローカルで比較した。controlの再学習は行っていない。

| guard | exp251 clean | exp259 | delta | 判定 |
|---|---:|---:|---:|---|
| overall RMSE | 8.502212005 | 8.427125551 | -0.075086454 | PASS |
| candidate logloss | 0.327735530 | 0.327708967 | -0.000026563 | PASS |
| distance 1000+ RMSE | 9.326545505 | 9.244368080 | -0.082177424 | PASS |
| exp115 spatial RMSE | 8.788133228 | 8.855452660 | +0.067319432 | FAIL |
| exp115 typewell-purged RMSE | 8.746112958 | 8.791490214 | +0.045377256 | FAIL |
| 最大well回帰 | - | `aed44918` +6.370552990 | +6.370552990 | FAIL（上限+0.25） |

- fixed-Viterbi MAEは`-0.027399882`、within10は`-0.000284092`、path switchは54減少。
- candidate AUCは`-0.000031088`、candidate expected-error MAEは`+0.032752211`。
  candidate logloss/Brierの微改善に対しerror絶対値校正は悪化しており、overall gainは
  candidate-levelの一様な改善ではなくfixed Viterbiのpath選択変化が主と解釈する。
- well別は386改善 / 384悪化 / 3同値。最大改善は`389ae58f` `-6.229024405`、
  最大回帰は`aed44918` `+6.370552990`。
- 絶対RMSE最大は`fb03ae90`で、exp259 `58.021328890`、control `58.004236030`。
  最大絶対RMSEと最大差分回帰は別のguardである。

### 最終判定

- 6 guard中3 PASS / 3 FAIL。exact datum augmentationはtrain-side rejected。
- hidden-like 2面と最大well回帰を犠牲にoverall改善だけを採用しない。
- inference、prediction、submissionは生成しない。shift幅や25%比率の事後gridも行わない。
- 再訪する場合は、改善した1000+に限定しsynthetic比率を下げた独立variantとして扱い、
  同じhidden-like／最大well回帰guardを維持する。

### 取得artifactとSHA

- repo policyに従いKaggle output全体は取得せず、metrics、candidate/bucket/subgroup/by-well、
  augmentation inventory、equivariance guards、feature importance/schema、model manifest、
  control reference、summaryだけを
  `kaggle/output/train_variant0_v1/artifacts/`へ取得した。
- 取得したexp259 small artifactsはsummary記載SHAと一致。
- training summary SHA256:
  `74909725f644daea693320f61972d9c5b4ac85b6440c44bec4ab39194a2995e7`。
- feature schema SHA256:
  `7a9217d6ed96f5f1e569dbefff2a1fb17751405d6ddccae5e5d9dbf12da787ae`。
- model manifest SHA256:
  `7f6cd5c84dd7693c271359537ace1e359af2ecffd39cf6e7fa5e446c7d326d4b`。
- OOF decompressed content SHA256はKaggle log/summaryで
  `96d34a2f7eb68f576f6a7e51bec8c11b6ef7294ec8989442d9fda8d644290913`と一致。
  追加監査でOOF本体を`/tmp`へ選択取得し、decompressed content SHAを独立再計算して一致した。
- model本体とimputerも`/tmp`へ選択取得し、manifest記録に対してmodel 10/10、imputer 5/5のSHA一致を確認した。診断生成物は15/15一致。
- 大きなOOF/model/imputerは不採用runのため実験配下へ保存せず、小さい診断生成物だけを保持した。
- exp251/exp259 OOFを同一GroupKFoldで比較し、fold deltaは`+0.066196 / -0.087980 / -0.127236 / -0.202058 / -0.004557`、4/5 folds改善と確認した。
- 1000+はexp251比-0.082177で相対PASSだが、元の絶対上限9.234366に対してはexp259 9.244368で+0.010002未達。
- inference prediction / submissionは生成していないため各SHAは対象外。
