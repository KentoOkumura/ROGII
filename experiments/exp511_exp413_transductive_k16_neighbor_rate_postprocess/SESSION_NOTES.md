# exp511_exp413_transductive_k16_neighbor_rate_postprocess セッションノート

## 目的

exp226 の周辺井戸 local-linear K16 rate 推定を、保存済み exp413 Stage D OOF の
予測後処理として使えるか評価する。各 outer-valid fold を擬似 test batchとし、
同じfold内の他井戸の予測TVTとraw X/Y/Zだけから低周波rate consensusを作る。

## 現在の状態

- Route: `ensemble`
- 状態: Kaggle Stage A version 4完了・性能gate FAIL・終端閉鎖
- 親実験: `exp413_scale5_likpf_full_replacement_on_exp335`
- 手法参照: `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction`
- 親CV / Public LB参照: `7.884802794404715 / 7.201`
- exp511 CV / LB: `7.883964795205812` / なし
- implementation / canonical Notebook / Kaggle package / run approval: `1 / 1 / 1 / 1`
- inference / submission: `0 / 0`
- 正規train Notebook: compact self-contained候補を採用
- 正規inference Notebook: template placeholder、未変更
- blocker: なし。inference、submissionはfail-close判断により対象外

## 2026-08-04 設計確定

- ユーザー依頼によりbacklog、steering、実験scaffoldを作成し、実装前の設計を固定した。
- exp413 OOFの各outer-valid foldを同時に処理するpredicted-only transductive後処理とした。
- donorは同foldの他outer-valid wellsの予測TVTだけに限定し、自井戸を除外した。
- outer-train / same-foldの真のsuffix TVT、ANCC、formation面、typewell GRをprediction生成から除外した。
- K16は16区間、`rho=10`、`theta0=118.4`、projection guard `0.3`、
  local-linear `k=50 / bandwidth=500 / ridge=1`、unique donor wells `>=8`へ固定した。
- primaryは`alpha=0.05`、final correction cap `±0.25 ft`の1本だけとした。
- first score row補正を厳密に0とし、明示fade、reanchor、clip base、U projectionを入れない。
- raw exp413以外のcandidate、parameter grid、report-only rescue、routerを作らない。
- pooled gain `>=0.01 ft`、4/5 folds、固定MD/hidden-like scope、by-well p95/worst、
  continuityの全AND gateを固定した。
- FAIL時はparameter / support / scope / gateを同じOOFで調整せず終端する。
- OOF評価をtrain-side実験の完了とし、別のOOF段階は置かない。PASS時だけ同じexp511の
  inference portを別承認候補にする。

## 根拠

- exp226はstandaloneでCV `9.427109597` / Public LB `9.837`。根本原因監査では、
  donorとの小さなsigned rate mismatchが長距離累積して低周波offsetになると確定した。
- exp114 spatial priorは弱いbaseを`0.443079 ft`改善した一方、worst-wellを`+6.508121 ft`悪化させた。
- exp118の強いexp092 baseへの小さなgate補正は`0.000854 ft`改善に留まったが、
  worst regressionを`+0.208085 ft`まで抑えた。
- exp201ではXY 8-nearest bias sign一致率`0.494502`で、絶対bias転写は支持されなかった。
- exp263ではexp226を含む固定combinationがOOF `8.238331` / Public LB `7.800`となり、
  exp226の補完性はあるが現行exp413より弱い。
- exp418のsigned K16 cumulative-rate oracleは表現上のheadroomを示したが、
  deployableなrate予測可能性は未証明である。

## 実行予定inventory

| scientific variant | report-only | model config | trained fold | booster | PF/HMM/Beam | GPU | parent retraining |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

実装・push前にも同じinventoryを再確認する。親exp413/controlを再学習しない。

## 2026-08-04 Kaggle実行承認

- ユーザーの「実行してください」を、正規train Notebook採用、Kaggle package作成、
  private CPU Stage A 1回の実行承認として記録した。
- 実行inventoryを再確認した。scientific primary 1、report-only 0、model/config 0、
  trained fold 0、booster 0、PF/HMM/Beam 0、GPU 0、親/control再実行0。
- runtimeはCPU、internet off、1 process。保存exp413 Stage D OOFとfold manifestだけを使う。
- 初回packageのkernel id/titleは
  `kentookumura/exp511-exp413-transductive-k16-neighbor-rate-postprocess-train` /
  `exp511 exp413 transductive k16 neighbor rate postprocess train`としてslug一致を確認したが、
  62文字slugの`SaveKernel 400`で停止した。同kernelの`pull -m`は403で、Kaggle側に
  resourceは作成されていない。
- 既存成功metadataが50文字以下へ揃っていることからslug長制約を高確度原因と判断し、
  同じexp511内で49文字の
  `kentookumura/exp511-exp413-k16-neighbor-rate-postprocess-train` /
  `exp511 exp413 k16 neighbor rate postprocess train`へ固定し直した。
- Kaggle CLI 2.2.3、OAuth credentialとlegacy credentialを確認した。GPU残枠は8.15hだが、
  本実験はGPUを使用しない。
- 短縮slugでkernel version 1（id_no `129683070`）をpushした。private CPU、internet off、
  source 3件と正規train Notebookをmetadataで再確認したが、33.94秒でscientific処理前に
  `FileNotFoundError: No raw geometry root matched all OOF wells: []`となった。
- competition sourceの実mountが
  `/kaggle/input/competitions/rogii-wellbore-geology-prediction/train`であるのに対し、
  exp511の候補が短縮mountだけだったことが原因。version 2では正式mountを最優先候補へ追加し、
  設定候補が外れた場合もfilename由来well集合が凍結OOF inventoryと完全一致する`train`
  ディレクトリだけを採用するfail-closed fallbackを追加する。科学parameter、入力SHA、gate、
  inventoryは変更しない。
- kernel version 2はraw geometry解決を通過したが、82.40秒で
  `local_linear_consensus`のeffective support計算が`ZeroDivisionError`になった。500 ft
  bandwidthに対して全donorが遠いqueryで、共通の極小指数を直接expした後のweight二乗和が
  0へunderflowしたことが原因。version 3ではlog-weightから最大値を引いてからexpする。
  これは全weightへ同じ正定数を掛ける数学的に等価な安定化で、bandwidth、K、ridge、donor
  順序、科学parameter、gateは変えない。極端距離のregression testも追加する。
- kernel version 3は上記数値安定化を通過し、truth-free prediction生成・artifact freezeまで
  完了したが、107.00秒でtruth-late評価用のexp115 hidden-like assignmentを解決できず停止した。
  configには期待SHAとpathが固定済みだった一方、package metadataのkernel sourceへexp115を
  添付していなかったことが原因。version 4では
  `kentookumura/exp115-hidden-like-spatial-holdout-from-ppt-train`をsourceへ追加する。
  hidden-like roleはprediction freeze後にだけ読み、科学parameter、予測生成、gateは変えない。

## 2026-08-04 実装

- ユーザーの別承認により、固定済み1 primaryを
  `exp511_exp413_transductive_k16_neighbor_rate_postprocess_compact_selfcontained_train.py`
  へ実装した。正規train / inference Notebookは上書きしていない。
- exp413 Stage D OOFのtruth-free readerは
  `id,well,outer_fold,scale5_x1p0_full_replacement__lgb_mean__pred_tvt`だけを読み、
  raw horizontal wellは`X,Y,Z`だけを読むstrict allowlistとした。
- OOF / fold manifest / 親metricsを期待SHAで解決し、raw geometryは773-well inventory、
  file SHA、logical content SHAを記録する。
- score row 0をanchorにし、残りのtransitionをexp226互換K16 basisへ射影する。
  `rho=10`平滑係数、`theta0=118.4`、donor projection `>=0.3`、
  local-linear `k=50 / bandwidth=500 / ridge=1`を固定した。
- fold-local fieldだけを使い、自井戸を除外した。距離、donor well、segment、source rowで
  stable sortし、選択donorのunique well数が8以上のsegmentだけ補正する。
- correctionはfirst score rowを厳密に0、`alpha=0.05`、`±0.25 ft` capへ固定した。
- coefficient field、support ledger、frozen predictionをparquet write/readbackし、
  logical/content SHAを保存してからtruthとhidden-like roleを接続する。
- pooled/fold/MD/hidden-like/by-well/continuity/mechanism readoutと固定all-AND gateを実装した。
- report-only candidate、parameter grid、router、same-OOF rescue、model、booster、
  PF/HMM/Beam、inference、submission pathは追加していない。

### Notebook構成比較

- 親exp413 compact self-contained trainは766行、`Contents`を含む9章。
- exp511候補は1,818行、`Contents`を含む10章・22 cellsで、path/SHA、strict input、
  K16、fold-local field、freeze、truth-late metrics、orchestrationをセル上で追える。
- 親より章立て・記載量が欠けず、同一exp helper importや薄い`main()`呼び出しではない。
- candidate notebookはoutput 0。正規train / inference Notebookは各6-cell templateのまま保持した。

### 実装検証

- 専用contract test `11 passed`:
  - fixed config / authorization boundary
  - K16 basisとzero-intercept係数回収
  - stable local-linearとself exclusion
  - 3-well support不足時のidentity
  - support成立時のfirst-row 0とcap
  - exp413 truth-free allowlist / truth-late freeze
  - raw `X/Y/Z` allowlist
  - strict all-AND promotion gate
- `py_compile`: PASS。
- `ruff --select F821`: PASS。全rule `ruff check`もPASS。
- Jupytext `--to ipynb` / `--test`: PASS。
- `task validate-exp`は環境に`task`がなく実行不能だったため、同等の
  `make validate-exp EXP=exp511_exp413_transductive_k16_neighbor_rate_postprocess`を実行し、
  strict validation PASS。
- ローカルNotebook実行、Kaggle API、package、push、run、output取得は行っていない。

## コマンドログ

- `make new-steering EXP=exp511_exp413_transductive_k16_neighbor_rate_postprocess`
  で `.steering/20260804-exp511-exp413-transductive-k16-neighbor-rate-postprocess/` を作成した。
- steeringのrequirements / design / tasklistへ数式、固定parameter、validation、gate、
  再現性、リスク、未承認範囲を記録した。
- `make new-exp EXP=exp511_exp413_transductive_k16_neighbor_rate_postprocess`
  でtemplate scaffoldを作成した。
- `config.yaml`、`README.md`、`SESSION_NOTES.md`、`result.md`、`metrics.json`へ
  design-only状態を記録した。
- 実装コード、Jupytext source、Notebook変更、ローカルNotebook実行、Kaggle API、
  package、push、run、inference、submissionは行っていない。
- その後のユーザー実装承認によりcompact self-contained `.py` / candidate `.ipynb`と
  専用testを作成し、上記静的検証を完了した。

## 再現性メモ

- seed policy: `no_rng_stable_fold_well_row_segment_distance_source_order`
- stochastic components: 本実験内なし。保存exp413 OOFだけが上流入力。
- CPU/GPU runtime: 将来のKaggle private CPU、1 process、GPU/internet off。
- input SHA: exp413 OOF / fold / scope / hidden-like / by-wellの期待SHAをconfigへ固定した。
- field/support/prediction SHA: 実装でschema/content SHAとreadbackを必須化した。
  実値はKaggle run後に記録する。
- model manifest / model SHA: 新規modelなし。
- submission SHA: train-side auditでは対象外。
- deterministic anchor: false。独立rerun一致前は昇格しない。

## 実装完了時点の次アクション（すべて完了またはgate FAILで対象外）

1. ユーザーの別承認があれば、実装済み候補を正規train Notebookへ採用する。
2. package前に1 variant / model・booster・PF/HMM/Beam・GPU・親再学習各0を再確認する。
3. Kaggle private CPU run後、input/field/support/prediction SHA、kernel version、metrics、
   decisionを記録する。
4. OOF gateをPASSした場合だけ、同じexp511内のinference portを別途判断する。

## 2026-08-04 Kaggle Stage A完了

- kernel version 4がprivate CPU / internet offでCOMPLETEした。scientific summaryは111.71秒。
- inventory実績はscientific primary 1、report-only 0、model/config/fold/booster 0、
  PF/HMM/Beam 0、GPU 0、親/control再学習0。
- exp413 CV `7.884802794404715`に対しexp511 CV `7.883964795205812`、pooled gainは
  `0.0008379991989029278 ft`。最低`0.01 ft`を未達。
- fold 2/4だけ改善し、nonworseは`2/5`で必要`4/5`を未達。fixed scope最大悪化
  `+0.000416092824926384 ft`、by-well p95 / worst `+0.0032299246942677855 /
  +0.20419145756869028 ft`、continuityは固定閾値内だった。
- support成立は`350 / 12,368 segments = 0.028298835705045277`。unique donor wells中央値5、
  median selected distance中央値`4,291.4847 ft`で、固定support 8を満たす範囲が狭かった。
- technical checksは全PASS。prediction freeze前のtruth/hidden role access 0、self donor 0、
  first-row correction 0、最大補正0.25 ft、row-order変更/nonfinite各0を確認した。
- input manifest / geometry / field / support / prediction logical / prediction content / freeze SHA:
  `15e223ec...4298` / `6c199052...7cd` / `7776e696...e1a` /
  `de9c5a95...15d` / `5415c24c...264` / `8320ba95...fbf` / `986e0654...e6e`。
  完全値は`metrics.json`と`kaggle/output/train_v4/artifacts/exp511_sha_manifest.json`へ記録した。
- 最終gateは
  `FAIL_CLOSE_WITHOUT_ALPHA_CLIP_K_BANDWIDTH_RHO_THETA_SUPPORT_FADE_SCOPE_OR_GATE_RESCUE`。
  same-OOF rescue、独立rerun、inference、submissionは行わず、exp511を終端閉鎖する。
