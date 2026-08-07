# 設計

## アプローチ

exp226 は各 train well の正解 `TVT` から、unknown suffixをK=16区間に分けた
raw/smoothed slope係数を作り、target区間のXY midpointごとに近傍50 donor segmentを
local-linear kernelで補間する。本実験では、この正解TVT由来slopeと後段処理を固定し、
同じ近傍50に対する重みだけをfold-safeな地層相対座標でsoft reweightする。

### Fold-safeな地層面推定

- 地層列は `ANCC`、`ASTNU`、`ASTNL`、`EGFDU`、`EGFDL`、`BUDA` の6列に固定する。
- outer foldはexp226と同じstable SHA256 5-fold identityを使う。
- 各outer foldでouter-train wellsだけから、exp287 `FormationPlaneKNN` と同じ
  well-median reference / `k=10` / self-exclusion契約で6地層面を推定する。
- dense ANCC imputerは使わない。
- outer-valid queryはouter-train referenceだけを使う。
- outer-train donor segmentのsignatureも、自wellを除いたouter-train referenceから推定する。
- full-train inferenceではtrain donorをleave-one-well-outで推定し、
  test queryは全train referenceから推定する。

### 地層signature

segment midpoint `q` で推定した地層面を `F_m(q)`、trajectoryのZを`Z(q)`とする。
11次元signatureを次で固定する。

1. 6個の面相対距離: `r_m(q) = Z(q) - F_m(q)`。
2. 5個の隣接面厚: `h_m(q) = F_{m+1}(q) - F_m(q)`。

outer-train donor segmentだけから次元ごとのmedianとMADを計算し、
`scale=max(1.4826*MAD, 1.0 ft)`で標準化する。queryとdonorの標準化差を
各次元で`[-3,+3]`へclipし、その二乗平均を`d_form^2`とする。
median/MAD、clip、scale floorはfold内で固定し、truthやOOF errorを見て選ばない。

### donor soft weight

exp226がXYだけで選んだ同じ50 donor segmentと、親のXY Gaussian weight
`w_xy`を固定する。地層係数と最終weightは次の1式だけを使う。

```text
g_form = 0.5 + 0.5 * exp(-0.5 * d_form^2)
w_new  = w_xy * g_form
```

`g_form`は理論上`[0.5, 1.0]`に収まり、地層情報だけでdonor supportを消さない。
6地層面またはsignatureがnonfiniteなら、そのquery segmentだけ
`g_form=1.0`として親XY weightへ戻す。weight、k、bandwidth、clip、
surface k、signature構成のgridは行わない。

## 実験範囲

- 対象実験: `exp376_exp226_formation_conditioned_k16_donor_kernel`
- Route: `pf_beam`
- 親実験: `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction`
- 参照実験:
  - `exp287_fold_safe_formation_74_addonly_on_exp264`: fold-safe formation signalのpositive evidence。
  - `exp009_formation_surface_guide` / `exp150_formation_physical_imputer_revisit`:
    単純なformation direct guide/candidateのnegative evidence。
  - `exp293_physics_only_candidate_bank_headroom_contract`:
    fixed12 candidate bankとH512 novelty契約。
  - `exp329_donor_support_risk_bounded_weight_shrink` /
    `exp362_segment_local_donor_slope_exact_hmm`: donor supportとtailのnegative evidence。
- 変更する変数: exp226 local-linear donor weightへの上記1種類のformation soft factor。
- 固定する変数:
  - exp226 K=16 segment、raw/smoothed係数、XY近傍50、bandwidth 500 ft、ridge 1、
    smooth rho 10、adaptive kappa 12項、near-strike/ANCC local theta、
    GR correction、U-projection。
  - exp226 outer-fold identity、score rows、parent OOF prediction。
  - exp287由来FormationPlaneKNNのwell-median / k=10 / self-exclusion semantics。
  - exp293 fixed12 bank、H128/H256/H512/whole-well block identity。
- 実装時の予定量: 1 scientific variant / 5 reporting folds /
  model config 0 / trained fold 0 / booster 0 / parent control再生成0。
- Runtime: Kaggle CPU、internet off。GPU、PF particle sampling、HMM decodeなし。

## 検証段階

### Stage 0: target-free technical/support guard

truth/errorを開く前に、次を保存してSHA freezeする。

- fold/well/row/K16 segment identity。
- outer-train referenceとouter-valid queryのwell overlap 0。
- donor self-exclusionとtest target formation read 0。
- 6面、11 signature、median/MAD、`d_form^2`、`g_form`のfinite/範囲。
- parent/new effective sample size、nearest donor distance、fallback理由。
- parent exp226 OOF file/decompressed SHAとprediction parity。

hard guard:

- 3,783,989 score rows / 773 wells / 12,368 K16 segments。
- `g_form` finite coverage 1.0、範囲`[0.5,1.0]`。
- nonfinite fallback fraction `<=1%`。
- `n_eff_new / max(n_eff_parent, eps)` のsegment p05 `>=0.75`。
- validation fold由来のTVT/formation reference count 0。

Stage 0 FAILならprediction scoring、grid、救済を行わずbranchを閉じる。

### Stage 1: direct exp226 candidate

Stage 0 PASS時だけtrue TVTをjoinし、保存済みexp226 controlと比較する。

- pooled RMSE gain `>=0.05 ft`。
- nonworse folds `>=4/5`。
- near 0--250、mid 250--1000、1000+、hidden-like spatial、
  hidden-like typewell-purgedの各delta `<=+0.02 ft`。
- by-well p95 delta `<=0 ft`、worst-well delta `<=+0.25 ft`。

このAND gateをPASSした場合だけdirect candidateとして支持する。

### Stage 2: fixed12へのadd-one novelty

Stage 0 PASS後はdirect gateの成否と分けて、exp293 fixed12へ13番目として加えた
oracle headroomを診断する。selector/modelは学習しない。

- H512 add-one oracle RMSE gain `>=0.05 ft`。
- whole-well add-one oracle RMSE gain `>=0.05 ft`。
- H512 strict unique-best block率 `>=5%`。
- H512 oracle gainが正のfold `5/5`。

全条件PASSなら、direct replacementでなくcandidate-bank用のnovel pathとして支持する。
Stage 1またはStage 2のどちらかがPASSしても、current-test生成、selector組み込み、
inference、submissionは別承認まで行わない。

## 再現性設計

- seed policy: exp226と同じstable SHA256 well-id 5-fold assignment。RNGは使わない。
- stochastic 処理の有無: なし。近傍、median/MAD、weight、solver入力順をstable sortする。
- PF/Beam / likelihood-PF / seed bagging の有無: なし。routeは`pf_beam`だが、
  実体はdeterministic K16 geometry candidate generation。
- 並列処理と乱数の関係: RNGなし。実装時にwell並列化する場合もimmutable
  `(outer_fold, role, well_id)`順へ再整列し、cKDTree query workerは1に固定する。
- CPU/GPU runtime と deterministic flags: Kaggle CPU、BLAS/thread数を記録、
  GPU無効、internet無効。
- train cache / test feature regeneration の SHA 記録方針:
  raw input manifest、fold assignment、formation reference、signature、support ledger、
  OOF prediction、block predictionはschema SHAとlogical content SHAを記録する。
  gzipはdecompressed content SHAを主証拠にする。
- model manifest / prediction / submission SHA 記録方針:
  modelはないためmodel manifestは非該当。OOF/candidate prediction SHAを記録する。
  inference/submissionは未承認であり、実行する場合だけtest prediction/submission SHAを追加する。
- Kaggle package bootstrap 確認方針:
  実装承認後に正規notebookへ採用し、prepare後のmetadataとbootstrap内
  `config.yaml`、source SHA、CPU/internet設定、stage/run flagsを照合する。
- deterministic anchor: 初回runだけでは呼ばない。rerunのlogical formation/signature/
  prediction SHA一致を確認した場合に限り再評価する。

## リスク

- リークリスク:
  validation wellの6地層列をsurface fitやsignatureへ混ぜるとfeature availability leakageになる。
  outer-valid全well除外とouter-train donor self-exclusionをmanifestでhard checkする。
  正解TVTはouter-train donor slopeとlate scoringに限定する。
- CV/LB 不一致リスク:
  exp226はCV `9.427110`に対しPublic LB `9.837`で悪化した。
  exp287のformation signalもglobal gainとPublic LB改善があった一方、worst-well
  `+8.228410 ft`だった。pooledだけで昇格させず、hidden-likeとwell-tailをhard gateにする。
- ランタイム/メモリリスク:
  outer-fold formation imputation、train donor self-exclusion、K16 donor ledger再構築が必要。
  row-level 11次元を常駐させずsegment-level 12,368行を基本単位にする。
- donor supportリスク:
  exp362では別のlocal-gradient契約が全segmentでfallbackした。
  本実験はexp226と同じXY近傍を保ち`g_form>=0.5`とするが、ESS比をStage 0で必ず確認する。
- 再現性リスク:
  nearest-neighbor tie、LAPACK solve、並列返却順による微小差があり得る。
  stable sort、worker 1、logical content SHA、rerun差分で管理する。
- 事後選択リスク:
  surface k、signature、scale floor、clip、weight式、gateを本設計で固定し、
  同一OOFでのthreshold/weight/grid救済を禁止する。
