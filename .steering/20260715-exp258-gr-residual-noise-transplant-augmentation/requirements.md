# 要件

## 依頼

`KAGGLE_DIRECTION.md` の `gr_residual_noise_transplant_augmentation` を新規実験化する。
typewell では説明できない実測 horizontal GR 残差を fold-safe な連続 block donor として
outer-train 内へ移植し、candidate ranker と、その OOF rank-slot 特徴を受け取る最終 TVT
LightGBM を順に再学習する。

ranker だけを最終出力にせず、exp238 と同様に outer-fold safe な selector score / rank / margin
を exp218 feature surface へ add-only し、最終 TVT prediction の clean official-start OOF で採否する。

## 制約

- Route: `ml_model`。最終予測は TVT LightGBM が生成し、固定 PF/Beam/HMM candidate と
  ranker 出力は add-only meta feature として補助利用するため。
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- 親は `exp238_nested_hmm_exp226_selector_rank_slot_addonly_on_exp218` とし、11 candidate、
  outer 5 folds、inner 4 folds、expected-error selector、35 rank-slot feature、exp218 380 base
  feature、final LightGBM 3 config を固定する。
- augmentation は outer/inner train well の raw train data だけから生成する。outer-valid / inner-valid
  well の GR residual、true TVT、error、oracle rank を donor 抽出や augmentation parameter 選択へ使わない。
- clean validation を固定し、validation GR / candidate / feature surface は改変しない。
- residual block は row-wise iid noise に崩さず、連続 block と missing mask を保持する。
- HMM/PF/Beam candidate generation、HMM transition/emission、candidate TVT、direct TVT correction は変更しない。
- historical exp238 no-noise selector/final OOF を control とし、parent/control の再学習は既定で行わない。
- white-noise と shuffled-residual は negative control として実装するが、実測 residual block と
  同時に一括学習せず、stage / variant を分離して実行できること。
- inference / competition submit は train-side guard 通過まで停止する。
- Kaggle train push 前に variant / config / fold / booster 数を `SESSION_NOTES.md` に記録し、
  明示承認を得る。

## 受け入れ基準

- `.steering/`、`config.yaml`、日本語の `README.md` / `SESSION_NOTES.md` / `result.md`、
  `metrics.json`、Jupytext percent 形式の train / inference notebook を持つ。
- Stage 0 residual audit で、affine reconstruction、donor/validation well 非重複、block length、
  missing-run、FFT/DWT/rotation-band、white/shuffled negative control、content SHA を記録する。
- augmentation RNG は immutable key（seed、variant、outer fold、inner fold、recipient well、row id）
  から stable seed を生成し、thread scheduling に依存しない。
- Stage 1 は clean inner-valid / outer-valid に対して historical no-noise、real residual block、
  white noise、shuffled residual の candidate AUC、rank margin、expected-error calibration、
  selector RMSE、near / 1000+ / hidden-like / worst-well を比較できる。
- Stage 2 は real residual block ranker が Stage 1 guard を通過した場合だけ解放され、
  outer-fold safe な 35 rank-slot feature を exp218 380 featureへ加え、3 config x 5 foldsの
  final TVT LightGBMを学習できる。
- final OOF は overall、6 distance bucket、exp115 hidden-like 2群、5 fold、by-well、worst-well、
  feature importance を historical exp238 / exp218 と比較する。
- feature schema、fold manifest、residual donor manifest、selector model、final model、OOF prediction
  の SHA を保存する。
- inference notebook は guard 未通過時に明示停止し、通過後も raw current test はclean入力のまま、
  saved augmented-trained selector / final modelだけを適用する契約にする。
- deterministic anchor として扱う場合は、feature content SHA、model SHA、prediction SHA、submission SHA、Kaggle kernel version が記録されている。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
