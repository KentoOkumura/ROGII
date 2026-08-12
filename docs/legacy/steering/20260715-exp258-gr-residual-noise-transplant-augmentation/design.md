# 設計

## アプローチ

### Stage 0: fold-safe residual donor audit

inner-train well の raw horizontal / typewell data から、robust affine fit

`horizontal_GR ~= gain * typewell_GR(true/known TVT) + bias`

を作り、`residual = horizontal_GR - fitted_typewell_GR` を連続 block として抽出する。
block は GR residual、missing mask、length、donor well、start/end row、gain/bias、residual scale、
spike rate、quantization proxy、FFT dominant / rotation / high-frequency ratio、DWT/detail energyを持つ。

recipient inner-train row の local GR windowは、recipientのtrue TVT/typewell対応から作るclean signalへ、
stable seedで選んだdonor blockを移植して生成する。missing donor位置はrow削除せずmaskとして保持する。
white-noiseはdonor residual scaleに一致させたiid Gaussian、shuffled-residualは同じdonor block内だけを
stable permutationしたnegative controlとし、主variantの連続構造とは分離する。

### Stage 1: nested ranker retraining

exp238のouter 5 x inner 4 fold contractを固定する。各inner modelについて、inner-trainのclean
candidate-long rowsにaugmentation viewを追加し、inner-validとouter-validはcleanのままにする。
selector objectiveはexp238と同じcandidate absolute-error regressor 1本に固定し、candidate bank、
context schema、row cap、LightGBM parameter、early stopping、rank-slot feature定義を変更しない。

no-noise historical exp238をbaselineとし、`real_residual_block`、`white_noise`、
`shuffled_residual`を別stage/versionで実行可能にする。primaryはreal residual block。

### Stage 2: final TVT LightGBM retraining

real residual block selectorが事前guardを通過した場合だけ、そのouter別train/valid scoreから
exp238と同じ35 rank-slot featureを作り、exp218の380 base featureへadd-onlyする。final modelは
exp238と同じ3 LightGBM config x 5 outer folds = 15 GPU boosters。historical exp238 final OOFを
controlとして参照し、base/controlは再学習しない。

inferenceではaugmentationを行わない。raw current-test featureをexp238 parity-safe generatorで
再生成し、augmentationで学習済み20 selectorと15 final LightGBMをfold対応させて適用する。

## 実験範囲

- 対象実験: `exp258_gr_residual_noise_transplant_augmentation`
- Route: `ml_model`
- 親実験: `exp238_nested_hmm_exp226_selector_rank_slot_addonly_on_exp218`
- 変更する変数: selector outer/inner train candidate-longへ追加するGR corruption viewのみ。
- 固定する変数: 11 candidate、HMM/PF/Beam generation、outer/inner fold contract、selector objective、
  clean validation、35 rank-slot feature、exp218 380 base feature、final LightGBM family、metric。

## 再現性設計

- seed policy: global RNGを使わず、SHA256で`seed|variant|outer|inner|well|row-id|view-index`
  からuint64 seedを作り、`np.random.default_rng`へ渡す。
- stochastic 処理の有無: donor/block/recipient view選択、white noise、shuffled permutationにあり。
- PF/Beam / likelihood-PF / seed bagging の有無: 再生成・再学習なし。保存済みfixed candidateを読む。
- 並列処理と乱数の関係: 各recipient keyごとに独立seedを作り、処理順・thread数で結果を変えない。
- CPU/GPU runtime と deterministic flags: selectorはCPU LightGBM、finalはexp218/238と同じ
  `gpu_use_dp=true`、`deterministic=true`、`force_col_wise=true`、固定8 threads。
- train cache / test feature regeneration の SHA 記録方針: residual donor manifest、augmentation inventory、
  clean/augmented feature sample、schema、nested scoreをdecompressed content SHAで記録する。
- model manifest / prediction / submission SHA 記録方針: selector 20 model/variant、final 15 model、
  OOF prediction、inference prediction、submissionのSHAを分離する。
- Kaggle package bootstrap 確認方針: prepare後のbootstrapからconfigを復元し、stage、variant、seed、
  GPU/internet、kernel sourceを正規configと比較してからpushする。

## リスク

- リークリスク: validation well residualをdonorへ混ぜる、recipient true TVT由来synthetic GRを
  validationへ生成する、nested ranker scoreのouter-foldを取り違えること。fold manifestとhard assertで停止する。
- CV/LB 不一致リスク: exp221/238でtrain-side gainとLB/worst-wellが一致しなかった。overallだけでなく
  near、1000+、hidden-like、worst-well、5-foldを全guardする。
- ランタイム/メモリリスク: raw GR blockをcandidate-long全件へ複製するとOOMになる。row cap、
  deterministic sampling、streaming/memmap、variant別実行を使い、full augmented tableを保持しない。
- 再現性リスク: stochastic donor選択とGPU LightGBM。stable per-key seed、feature content SHA、
  deterministic flags、model/prediction SHAで追跡する。
