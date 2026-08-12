# 設計

## 仮説

exp209の遷移、観測分散、state grammarを固定したまま、current-well内で緩やかに変化するGR scale/offsetだけをone-pass affine stateで追跡すれば、identity観測中心よりsuffix HMMを改善できる。

## アプローチ

exp209 exact HMMをidentity observation controlとして固定する。まずcurrent wellのvisible prefixで`x=Type Well GR(TVT_input)`、`y=finite raw horizontal GR`のrobust affineをfitし、初期state `[b, log(a)]`を作る。次にfrozen exp209 posterior mean/stdへType Well GRと局所勾配を対応させ、有限raw GRがあるrowだけdeterministic causal EKFを1回更新する。raw GR欠損rowはprediction stepだけとする。全rowの`a_t,b_t` scheduleを凍結してから、次の観測中心を使うexact HMMを1回だけ再実行する。

```text
mu_GR(state_t) = exp(log_a_t) * GR_typewell(TVT_state_t) + b_t
affine_state_t = [b_t, log_a_t]
```

affine filter内部の観測分散だけは、exp209 `sigma_GR^2`に`Type Well GR局所勾配^2 × base TVT std^2`を加える。variant HMM本体の`sigma_GR`はexp209 zero-fill stdのまま変えない。

## 実験範囲

- 対象実験: `exp345_exp209_time_varying_gr_affine_calibration_hmm`
- Route: `pf_beam`
- 親実験: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- 変更する変数: observation centerの時間変化`a_t,b_t` scheduleだけ。
- 固定する変数: exp209 `sigma_GR`、missing weight 1、GR/typewell preprocessing、Gaussian emission、grid、`sig_r=0.002`、`sig_p=0.02`、position floor、momentum、prior、posterior mean。
- 独立性: exp338と親を共有する兄弟だが、dependencyも自動合流も持たない。

## Affine state契約

- state: `[intercept_b, log_scale_a]`
- transition: local-level random walk
- filter: deterministic causal extended Kalman filter、smootherなし
- 初期pair: visible `TVT_input`へ補間したType Well GRと有限raw horizontal GR
- 最小pair数: 40
- Type Well GR最小標準偏差: 5.0
- slope bounds: `[0.25, 4.0]`
- prefix RMSE上限: 60.0
- robust fit: residual trim quantile 0.90、2 iterations
- prefix RMSE: 固定2回trim後のretained pairsで評価
- process noise: outer-trainだけでfoldごとに1回empirical Bayes推定
- estimator: robust prefix stateの隣接increment二乗median
- state sequence: finite pair 40件ごとのexpanding robust fitと最終prefix fit
- increment scale: state差二乗を対応するraw row差で除した1-row variance
- shrinkage: support `n/(n+100)`でouter-train global medianへ縮約
- shrinkage space: varianceの線形縮約。数値floorは`1e-12`
- initial covariance: retained-pair OLS covarianceを`[b, log(a)]`座標へ変換
- covariance update: Joseph form。各有限raw GR行のposterior stateを当該行scheduleとする
- GR NLL: current-row update前のone-step predictive NLL
- fallback: exp209 identity `a=1,b=0`
- schedule freeze後の再fit、grid、smootherは禁止

## 段階設計

1. Runtime microbenchmark
   - stable SHA256順32 wells。
   - masked parent 32 + variant 32 = 64 HMM runs。
   - full Stage 0外挿が8.5時間を超えたら科学評価へ進まない。
2. Stage 0 prefix mask
   - 各wellのvisible prefixからlast 640 rowsをmaskし、最低160 rowsをvisibleに残す。
   - masked exp209 parent pathを作り、schedule freeze後にvariantを作るため、親773 + variant773 = 1,546 HMM runs。
   - maskしたTVTはschedule/prediction freeze後にだけ接続する。
3. Stage 1 full suffix
   - Stage 0全gate PASSと別承認後だけ有効化する。
   - 保存済みexp209 posterior mean/stdをbase pathにし、新variantだけ773 HMM runs。親controlは再実行しない。

## 判定設計

Stage 0は以下をAND gateにする。

- exp209 masked parent比RMSE gain `>=0.05 ft`
- 改善fold `>=4/5`
- GR Gaussian NLL改善
- affine state boundary jump p95 `<=3 sigma`
- hidden-like非悪化
- worst-well regression `<=+0.25 ft`
- fallback fraction `<=0.50`
- projected runtime `<=8.5 h`

Stage 1はexp209 raw HMM `11.9382872349`比gain `>=0.05 ft`、4/5 folds、1000+、hidden-like spatial、hidden-like typewell-purged、by-well p95非悪化、worst `<=+0.25 ft`をAND gateにする。保存済みLikPFと50:50 blendは診断値として記録するが、科学variantや重みgridにはしない。

## 再現性設計

- seed policy: RNGなし。記録上seed 42、outer fold/well ID/raw row/variant順を固定。
- stochastic処理: なし。
- PF/Beam/likelihood-PF生成: なし。saved LikPFはlate readoutだけ。
- 並列処理: exp209採用のouter workers 2、Numba threads 2を開始点とし、乱数系列は存在しない。
- runtime: Kaggle CPU、GPU off、internet off、上限8.5時間。
- SHA: exp209 HMM/LikPF依存SHA、base path、affine schedule、fallback、process noise、prediction、metricsのschema/content SHAを保存する。gzipはdecompressed content SHAを主証拠にする。
- model manifest: 学習modelなしのため非該当。prediction SHAは必須、submission SHAはinference無効のため非該当。
- Kaggle bootstrap: 実装承認後に正のconfig/notebookからpackageを再生成し、metadataとbootstrap内configの一致をpush前に確認する。
- deterministic anchor: train-side candidate auditでありsubmission anchorとは呼ばない。

## リスク

- base path誤りをaffine stateが吸収する循環: one pass、base stdを使う観測分散、prefix mask、worst gateで制限する。
- exp211/216のstatic affine negative: 本実験を低優先にし、direct gainとhard tail gateを要求する。
- raw GR欠損: exp308 weightを移植せずupdateをskipし、variant HMM観測側はexp209 preprocessingを維持する。
- 実行時間: Stage 0は最大1,546 HMM runsのため32-well runtime gateを先に置く。
- 因果解釈: exp338 well別`sig_r`を混ぜず、独立兄弟として単一変更を保つ。

## 実装状態

2026-07-22にcompact self-contained Jupytext train候補と専用contract testsを実装した。既存正規Notebookはplaceholderのまま上書きせず、Kaggle実行flag、inference、submissionは無効のままとした。
