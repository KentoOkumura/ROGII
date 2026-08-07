# 設計

## アプローチ

exp295のprefix-conditioned horizontal/Type Well GR encoderを固定し、row `i`とType Well TVT state `k`のunary logitを`u_theta(i,k)`とする。正解TVTに最も近いgrid stateを`k_i*`として、学習はrow独立の

`L_CE = -mean_i log softmax(u_theta(i,:))[k_i*]`

だけで行う。state間transition、forward-backward、posteriorは学習lossにもearly stoppingにも使わない。

model freeze後だけ、unaryをexp209固定grammarへ入力し、known prefixをhard clampしたexact log-space forward-backwardから`E[TVT_i | unary, trajectory, prefix]`を得る。これをofficial hidden suffix全rowの主評価値とする。local argmaxは診断のみで、最終候補にはしない。

## 固定する科学契約

- input: 対象well自身の`MD/X/Y/Z/GR/TVT_input`と対応Type Wellの`TVT/GR`のみ。
- neighbor horizontal well、same-typewell donor、candidate bank、既存ML/PF/Beam predictionを使わない。
- preprocessing/architecture: exp295のrobust median/MAD、missing mask、derivative channels、embedding 64、dilation`[1,2,4,8,16]`、GroupNorm、dropout 0.10、prefix context/FiLM、normalized bilinear cosine、temperature範囲を固定する。
- training views: outer-train wellsのofficial/256/512 pseudo-cut最大3 views、全suffix row、row chunk 256。pseudo-cutはaugmentationだけでvalid評価へ使わない。
- decoder: exp209の`step=0.35`、41 rate states、rate span 0.10、`sig_r=0.002`、`sig_p=0.02`、`start_sig=0.75`、`r0_sig=0.01`、`band_pad=100`、`mom=0.998`を固定する。
- controls: 同一trained modelのType Well GR circular shuffle、zero-GR unary geometry-only。追加modelは学習しない。

## 計算量gate

full foldの前に、fold 0 outer-trainからsuffix長quartileごと4件をstable SHA256順に選んだ固定16 viewsでT4 microbenchmarkを行う。次を別々に測る。

1. CE forward/backward、optimizer、AMP、全row chunk処理のrows/secとpeak GPU memory。
2. model freeze後のreal/shuffle/geometry-only exact SSM decodeのcells/secとwell runtime。
3. 556 fit wells、最大3 views、最大8 epochs、fold 0 valid/control decodeへ線形外挿したp50と、固定16-view throughputのp10を使う保守的upper estimate。training/forward rateはCSV読み込み・preprocessingを含むend-to-end時間で計算する。

peak GPU memory`<=14 GB`かつfold 0総時間の保守的外挿`<=8.5 h`だけをPASSとする。FAIL時はview、epoch、architecture、state bandを同じexpで削らずbranch closeする。

## 検証設計

- fold: exp295/exp202と同じ5-fold complete-well GroupKFold。
- score: organizer-equivalent official `TVT_input` hidden suffix全rowのpooled/fold RMSE。
- diagnostics: local CE/NLL、nearest-state accuracy、real/control posterior mass within10/20、target-in-grid、grid-edge mass、1000+、hidden-like spatial/typewell-purged、well p95/worst、runtime/memory。
- truth-late: outer-valid suffix TVTはmodel/unary/posterior/controls/SHAのfreeze後だけjoinする。

## 段階実行

### Stage 0: implementation and fixed microbenchmark

- compact self-contained train候補、fail-closed inference候補、専用contract testsは実装済み。
- T4 microbenchmarkのKaggle package/push/runは未承認であり、別承認を必要とする。
- benchmark PASS前にfull fold trainingを開始しない。

### Stage A: fold 0

- `1 architecture × 1 fold × 1 seed = 1 neural model`。
- optimizerはAdamW、lr`3e-4`、weight decay`1e-4`、最大8 epochs、gradient clip 1.0、AMP、1 well/batch、gradient accumulation 4、worker 0。
- early stoppingはstable outer-train holdout local CE、patience 2だけ。
- PASS: finite prediction 1.0、target-in-grid`>=0.995`、prefix clamp誤差`<=1e-6 ft`、real NLLがshuffleより`>=0.05 nats/token`良い、within10 mass差`>=0.03`、posterior RMSEがgeometry-onlyと同fold exp209より各`>=0.25 ft`改善、exp209比well p95非悪化、worst regression`<=10 ft`、runtime/memory gate PASS。
- 1条件でもFAILならStage Bへ進まず、同一expでrescue gridを行わない。

### Stage B: full OOF

Stage A全PASSと別承認後だけfold 0 modelを再利用し、fold 1--4の4 modelsを追加する。realがshuffle/geometry-only双方をpooled`>=0.50 ft`かつ5/5 foldsで改善し、pooled OOF`<=6.0 ft`、exp221を5/5 folds・1000+・hidden-like 2面で改善、p95非悪化、worst regression`<=5 ft`をpromotion PASSとする。`6.0 < OOF <=6.75`かつattribution PASSは別architecture実験の根拠だけにし、`>6.75`またはattribution FAILはcloseする。

### Stage C: inference

Stage B全PASSと別承認後だけ同じexp内でcurrent-test inferenceを実装する。5 fold modelsをそれぞれ独立にfull-well exact decodeし、5本のposterior-mean TVTをrow-wise等重み算術平均する。unary平均、fold weighting、既存候補とのblendは行わない。raw-test parity、model manifest、fold別prediction SHA、平均prediction SHA、submit-check前にsubmissionを作らない。

## 実験範囲

- 対象: `exp331_prefix_gr_unary_local_ce_exact_ssm`
- Route: `ensemble`
- 親: `exp295_prefix_anchored_wholewell_gr_alignment_ssm`
- 変更する変数: training objectiveをwhole-well structured NLLからrow-independent local CEへ変更し、exact SSMをmodel-freeze後へ移す。
- 固定する変数: input、fold、architecture、preprocessing、decoder、controls、evaluation/promotion gate。

## 再現性設計

- seed: global 42 + stable SHA256 per well/fold/view/control。
- stochastic: PyTorch CUDA convolution、AdamW、dropout、seeded dataloader order。
- worker 0、global RNGをthread/jobへ共有しない。CuDNN benchmark false、deterministic algorithms`warn_only=True`、AMP有効。
- deterministic anchorにはしない。
- input/fold/view/preprocessing schema+content、model state+manifest、unary/posterior/prediction、Kaggle package/kernel versionを記録する。gzipはdecompressed content SHAを主証拠にする。

## リスク

- local CEはglobal pathを直接最適化しないため、row-wise ambiguityをSSMが評価時だけで回収できない可能性がある。
- row×TVT logits自体は大きく、DPを外しても8.5時間を超える可能性がある。
- nearest-state hard labelは0.35 ft量子化誤差を持つが、label smoothing/soft CEを同じexpで追加しない。
- valid suffixをearly stopping、temperature、band、architecture選択へ使うleakageを禁止する。
