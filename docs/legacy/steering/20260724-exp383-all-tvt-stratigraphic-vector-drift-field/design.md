# 設計

## アプローチ

構造座標を`S = TVT + Z`と置く。exp226がK16 slopeとして補間している量は
`dS/dMD`に対応する。本実験ではouter-trainの正解TVT全体を使い、
`S`の絶対場と水平ベクトル勾配`g=(dS/dX,dS/dY)`を同時に推定する。

対象坑井の局所水平接線を`t_xy=(dX/dMD,dY/dMD)`とすると、
物理drift rateは`r_vec=t_xy^T g`である。対象pathは、地層条件付き絶対場、
vector rate、exp226 fallback rate、既知prefixを同時に満たす制約付き最小二乗で復元し、
最後に`TVT=S-Z`へ戻す。

### 1. Foldとrole

- exp226と同じouter 5-fold well splitを使う。
- outer-train raw rowsだけからsurface teacherとdrift donor catalogを作る。
- outer-validの`TVT`と6地層列はprediction/support freeze後のlate scoringだけで読む。
- outer-train donorの地層covariateも、自wellを除いたsurface referenceから生成する。
- full-train inferenceでは773 train wellsの全TVT/6地層列をteacherにし、
  target wellに同名train wellがある場合もreferenceからself-excludeする。

### 2. 全TVT multiscale donor catalog

- 各outer-train wellの全軌跡をMD基準の`64/256/1024 ft` windowで覆う。
- window center strideはそれぞれ`32/64/256 ft`に固定する。
- window内の`S~MD`、`X~MD`、`Y~MD`はfixed Huber IRLS
  (`delta=1.345`, `5 iterations`)でfitし、`dS/dMD`と`t_xy`を得る。
- 各windowのcenter`MD/X/Y/Z`、scale、source well、fit residual、
  6地層相対距離、隣接面厚、surface gradient/uncertaintyを保存する。
- 同じsource wellの密なwindowが支配しないよう、queryごとに1 source well当たり
  最大4 nodes、最大32 unique wells、最大128 nodesへ固定する。
- 64 ftを局所signal、256 ftをprimary、1024 ftをlong-trendとし、
  3 scaleをCV後にbest-of選択しない。3 scaleを1つの固定weighted systemへ同時投入し、
  各scaleの重みはwindow fit residualの逆分散だけから決める。

### 3. 6地層面とstratigraphic signature

- 6面をouter-trainのrow-level surface pointsから推定する。
- surface pointは各wellで32 ft MDごとに決定論的に間引く。
- queryごとに24 unique wells、1 well最大8 points、最大192 pointsの
  距離重み付きlocal planeをfitする。
- bandwidthは選択した24番目unique wellまでの距離を使うadaptive値とし、
  `[500, 4000] ft`へclipする。
- ridgeはweighted normal matrixのtraceに対する`1e-6`比へ固定する。
- 各queryで6面の値、`dF/dX,dF/dY`、weighted residual variance、
  nearest distance、unique-well数、condition numberを保存する。
- signatureは面相対距離6、隣接面厚5、surface gradient 12、
  surface variance 6の29次元に固定する。
- donor covariateもleave-one-well-out surfaceから作り、target availabilityに合わせる。

### 4. 地層条件付き絶対場とvector field

query nodeごとに、上記最大128 donor nodesへ次の固定weightを掛ける。

```text
w_xy    = exp(-d_xy^2 / (2 * h_adaptive^2))
w_form  = exp(-0.5 * mean(clip(z_donor - z_query, -3, 3)^2))
w_scale = 1 / max(window_residual_variance, 1e-4)
w       = w_xy * w_form * w_scale
```

`z_*`はouter-train donorだけでmedian/MAD標準化した29次元signatureである。
6面それぞれについて`R_f=S-F_f`のlocal affine planeをfitし、
target surfaceを足し戻した6個の`S_f`と`g_f`を得る。
primaryはCV後に面を選ばず、6系列のcomponent-wise robust medianに固定する。

vector solveは水平方位のweighted design conditionを記録し、
ridgeをweighted normal matrix traceの`1e-4`比へ固定する。
donor residual covariance、ESS、unique-well数、surface variance、condition numberから
absolute uncertaintyとrate uncertaintyを計算する。

### 5. prefix校正とexp226縮約

- 対象坑井の全finite`TVT_input`について`S_input=TVT_input+Z`を作る。
- raw absolute fieldとの差のHuber location (`delta=1.345`, 5 iterations)を
  単一vertical bias`b_prefix`としてfitする。
- prefix biasのfitに未知suffix TVTを使わない。
- field confidenceは次の固定積で定義する。

```text
c_support = clip(ESS / 32, 0, 1)
c_wells   = clip(unique_wells / 24, 0, 1)
c_cond    = clip(log10(1e4) / max(log10(condition), log10(1e4)), 0, 1)
c_surface = exp(-surface_variance / max(train_LOO_surface_variance_p50, 1e-6))
c_field   = c_support * c_wells * c_cond * c_surface
r_final   = c_field * r_vec + (1 - c_field) * r_exp226
```

surface varianceの基準はouter-train leave-one-well-outだけで凍結する。
target truthを使ったcalibrationは行わない。

### 6. path solve

targetは64 ft MD nodeでfieldをqueryし、全rowへ線形展開する。未知suffixの`S`は、
次を同時に満たすbanded weighted least squaresで1回だけ解く。

- known prefixの`S_input`はhard equality。
- absolute term: `S_j ~= S_abs_j + b_prefix`、重みはabsolute inverse variance。
- derivative term: `S_{j+1}-S_j ~= r_final_j * ΔMD`、重みはrate inverse variance。
- curvature term: 二階差分、固定係数`1e-3`。
- exp226は上記`r_final`のfallbackとしてのみ入り、prediction blendは行わない。

solverがnonfinite、rank deficient、またはfield coverage外ならwell全体をexp226へ戻し、
理由を保存する。

## 実験範囲

- 対象実験: `exp383_all_tvt_stratigraphic_vector_drift_field`
- Route: `pf_beam`
- 親: `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction`
- positive reference: `exp287_fold_safe_formation_74_addonly_on_exp264`
- negative references:
  - `exp138_ancc_surface_predictability_audit`
  - `exp150_formation_physical_imputer_revisit`
  - `exp362_segment_local_donor_slope_exact_hmm`
  - `exp376_exp226_formation_conditioned_k16_donor_kernel`
- 変更する変数: donor catalog、surface representation、scalar slopeからvector fieldへの推定、
  prefix calibration、uncertainty shrink、path solver。
- 固定する変数: exp226 outer-fold identity、score rows、保存済みexp226 control、
  6地層列、multiscale window、surface/field kernel、primary median、gate。
- 予定量: 1 physical candidate / 5 reporting folds / fitted ML model 0 /
  HMM 0 / PF 0 / Beam 0 / booster 0 / parent control再実行0。
- Runtime: Kaggle CPU、internet off。実装後に16-well resource preflightを行い、
  full runは別のユーザー承認を必要とする。

## 検証段階

### Stage 0: target-free integrity/support

truth/errorを開く前にfold、surface、donor catalog、field、uncertainty、
prefix availability、raw predictionをlogical content SHAでfreezeする。

hard gate:

- 773 wells / 3,783,989 score rows / 5 folds。
- outer-valid donor/surface reference overlap 0。
- outer-valid/test生Formation read 0、suffix truth read 0。
- surface primary coverage`>=0.98`。
- vector field採用coverage`>=0.95`。
- effective donor数p05`>=10`、unique donor wells p05`>=16`。
- finite prefix calibration coverage`1.0`。
- field/uncertainty/path input finite coverage`1.0`。
- 16-well projected full runtime`<=30,600 sec`、peak RSS`<=25 GB`。

Stage 0 FAILならlate truth join、score、threshold/grid救済を行わない。

### Stage 1: direct physical path

Stage 0 PASS時だけouter-valid TVTをlate joinし、保存済みexp226 OOFと比較する。

- pooled RMSE gain`>=1.0 ft`、すなわちCV`<=8.427109596582213`。
- positive folds`>=4/5`。
- 1000+ RMSE gain`>=0.75 ft`。
- hidden-like spatial/typewell-purged gainが各`>=0.50 ft`。
- near 0--250のdelta`<=+0.05 ft`。
- prediction correlation vs exp226`<=0.999`。
- by-well改善数、p95、worst、`+1/+3/+5 ft`悪化well数は必須報告だが、
  初回のfield識別可能性gateには入れない。

全条件PASSでexp384の実装候補へ昇格する。CV`<=8.0`ならLB 6.5ロードマップの
strong signalと記録する。PASSしてもinference、submission、exp384実行は別承認とする。

## 再現性設計

- seed policy: exp226 outer-fold identityとstable sort。RNGなし。
- stochastic処理: なし。Huber IRLS、local plane、banded solveは固定iteration/order。
- PF/Beam/likelihood-PF/seed bagging: なし。
- 並列処理: 実装時は`(fold, role, well_id, MD, scale)`順へ再整列し、
  neighbor query worker 1をdeterministic controlとする。
- runtime: Kaggle CPU / GPU off / internet off。BLAS/thread数をmanifestへ保存する。
- SHA: raw input manifest、fold、surface point catalog、surface prediction、
  multiscale donor catalog、signature、support、absolute/vector field、
  prefix calibration、path、OOF predictionをschema/logical content SHAで記録する。
- gzip: decompressed content SHAを主証拠にする。
- model manifest: fitted ML modelはないため、物理solver contractとparameter manifestを保存する。
- prediction/submission: OOF prediction SHAを保存する。test/submissionは未承認。
- deterministic anchor: 初回runでは主張せず、rerunでsurface/catalog/prediction SHAが一致した場合だけ再評価する。
- Kaggle bootstrap: 実装承認後にmetadata、bootstrap内config、source SHA、
  CPU/internet/run flagsを照合する。

### 実装時に固定した入出力細部

- targetの29次元signatureで相対深度を作る構造Sは、suffix truthではなく
  保存exp226 OOFの`S_parent=tvt_pred+Z`を使う。
- exp226 fallback rateは同じ保存pathをraw MD上で決定論的に微分して作る。
- vector field bandwidthは24番目unique donor well距離を`[500, 4000] ft`へclipする。
- exp384へ渡す生成物は256 ft donor nodes、query fields、truth-free OOF keys、
  Stage 1後のOOF-with-truth、logical SHA付きmanifestに固定する。
- surface/signature/field/uncertainty/path inputのfinite coverageはStage 0で`1.0`を要求し、
  欠損をpost-hoc補間して救済しない。

## リスク

- leakage: target生Formation、suffix TVT、同fold valid donor混入をrole read ledgerでfail-closedにする。
- availability mismatch: donor covariateもleave-one-well-out surfaceから作る。
- CV/LB: exp226はCV 9.427に対しLB 9.837と悪化したため、1 ft未満の改善では後段へ進めない。
- visible-test overfit: 3 sample wellsのscore/shapeでwindowやgateを選ばない。
- support: exp362の0/12,368退化を繰り返さないよう、full-TVT catalogのnon-degeneracyをtruth前にhard gate化する。
- surface: exp138/150のlong-tail surface誤差をuncertaintyとexp226縮約で扱う。
- field integrability:局所vectorが非保存場になる可能性をabsolute termとbanded path solveで抑える。
- runtime/memory: row全組合せを作らず、well-capped donor nodeと64 ft query nodeを使う。
- post-hoc tuning: window、stride、surface/field donor数、weight、ridge、confidence、
  solver係数、gateを同一OOFで救済しない。
