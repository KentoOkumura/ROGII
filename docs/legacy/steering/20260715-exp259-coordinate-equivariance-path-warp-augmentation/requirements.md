# 要件

## 依頼

`KAGGLE_DIRECTION.md` の `coordinate_equivariance_path_warp_augmentation`
を実験化する。座標系を変えるだけの厳密な対称変換と、実在範囲内の坑井・地質 path
を合成する近似変換を分離し、変換後の GR・trajectory・prefix・周波数特徴を整合的に
再生成できることを先に監査する。

## 制約

- Route: `ensemble`。最終利用先は PF/Beam/HMM candidate を評価する learned
  likelihood / ranker / calibrator だが、初回は prediction path を置換しない。
- 親: `exp251_raw_test_safe_dual_objective_candidate_ranker`。raw-test-safe feature
  contract と固定 candidate bank は将来の train stage の参照に留め、初回 audit では
  model / candidate を再学習・再生成しない。
- 初回 stage は `transform_audit_only`。active variant / LightGBM config / fold /
  booster は `0 / 0 / 0 / 0`。
- 後続の実装済み学習 stage は`train_exact_datum_after_transform_audit`とし、別途学習中の
  exp251 `raw_test_regenerated_copcf` 295列版を固定controlとして参照する。exp259ではcontrolを
  再学習せず、`exact_tvt_datum_shift` 1 variant、2 objectives、5 folds、合計10 CPU boostersだけを
  学習する。
- exp251 selected 295列にはraw X/Yやlocal geometry列がないため、heel translation、reflection、
  yawは学習行を変化させない。これらはinverse監査専用とし、学習にはTVT datum shiftだけを使う。
- TVT datum shiftのraw-well contractは初回監査で確認済みとし、学習時はselected 295列への誘導作用を
  明示する。7個のabsolute TVT featureだけを同じ量だけshiftし、candidate error、within10 label、
  残り288個のrelative/likelihood/prior featureが不変であることをhard guardする。
- 厳密変換は heel-centered XY translation、左右反転、yaw 回転、TVT datum shift。
- 近似変換は MD stretch、TVT shear、XY plane tilt、低周波 spline TVT warp、
  smooth XYZ control-point perturbation。
- 近似変換後は transformed TVT から typewell GR を再 sample し、trajectory 差分、
  prefix 統計、FFT / Haar-DWT 診断を再生成する。
- absolute spatial / KNN feature と local geometry feature を同じ equivariance contract
  に入れない。
- train 分布外の slope / curvature / typewell coverage、非単調 MD、非有限値を reject
  する。
- MTP / CNN / heatmap flip、direct PF/HMM replacement、submission は対象外。
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。

## 受け入れ基準

- 変換 parameter は stable SHA256 key だけから決まり、入力順と thread scheduling に
  依存しない。
- 厳密変換は inverse 後の `MD/X/Y/Z/TVT/TVT_input/GR` 最大絶対誤差を保存し、
  tolerance 内である。
- 近似変換は official-start anchor で連続し、prefix は変更せず、typewell coverage と
  GR/TVT 対応を保存する。
- real train から target-free geometry envelope を作り、変換別 accept/reject と理由を
  well 単位で保存する。
- transform manifest、distribution envelope、transform summary、SHA summary を保存する。
- Jupytext percent 形式の compact train notebook から正規 `.ipynb` を生成し、入力、
  変換、guard、metrics、生成物をセル上で追える。
- inference notebook は audit / train guard 通過前に明示停止する。
- 学習stageはcompleted exp251 version 3のselected 295列schemaとSHAがない場合に停止する。
  exp251 295列clean controlのsummary/metrics/by-wellは学習入力に使わず、並列実行中なら比較を
  `pending`としてexp259 OOFを保存する。control完了後にsaved metrics同士を比較し、exp259内で
  controlを再学習しない。
- `py_compile`、Ruff F821、Jupytext convert/test、unit test、strict `validate-exp` が通る。
- deterministic anchor として扱う場合は、feature content SHA、model SHA、prediction SHA、submission SHA、Kaggle kernel version が記録されている。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
