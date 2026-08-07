# 設計

## 目的と位置付け

exp295はGR matchingをconfidence featureやcandidate selectorへ回収する実験ではない。各horizontal rowがType Well TVT gridのどこに対応するかをlearned unaryとして直接推定し、complete-well state-space posteriorからTVTを出力する。

primary parentは、real GR heatmapにsignalがある一方でlocal argmax/MTP pathが弱かった`exp202_heatmap_mdn_candidate_generator_probe`とする。`exp178`のpair AUC、`exp221`のexact HMM改善をpositive evidence、`exp197`のlocal CNN弱信号、`exp244`のlocal-linear pseudo-start失敗、`exp292`のhand-crafted warp score識別性FAILをnegative evidenceとして固定参照する。

`exp293` Stage 2/3とは役割が異なる。exp293は固定物理candidate bankの離散選択をhand-crafted GR evidenceで行うphysical-only route、exp295はcandidate bankを使わずcontinuous TVT stateのunaryをouter-train wellsから学習するrouteである。

## 数理モデル

horizontal row `i`の潜在状態をType Well TVT位置`q_i`とrate state`r_i`の組`z_i=(q_i,r_i)`とする。posteriorは次で定義する。

\[
p(z_{1:L}\mid x_w,r_w,a_w) \propto
\exp\left(\sum_i u_\theta(i,q_i;c_w)
+\sum_{i>1}\log A_{\mathrm{exp209}}(z_{i-1},z_i)
+\log p_{\mathrm{exp209}}(z_1\mid a_w)\right)
\]

- `x_w`: 対象井自身のhorizontal GR/missing mask/GR derivative。
- `r_w`: 対象井に対応するType Well GR/missing mask/GR derivative。
- `a_w`: known `TVT_input` prefixと対象井自身の`MD/X/Y/Z`。
- `c_w`: visible prefixのmatched horizontal/Type Well pairをmasked-attention poolingして得る32次元well context。
- `u_theta`: shared multi-scale 1D encoder、FiLM conditioning、bilinear cosine similarityで作るrow x TVT unary logit。
- `A_exp209`: exp209のTVT/rate grid、transition、initial-rate grammarを変更せず再利用する。

known prefixのfinite `TVT_input` stateはhard clampする。hidden suffixではexact log-space forward-backwardを実行し、RMSE目的の出力を`E[q_i | inputs]`とする。Viterbi、marginal MAP、credible interval、entropyは診断だけに保存する。

## 入力とpreprocessing

- Type Well GRはexp209と同じTVT state gridへ範囲内補間し、外挿しない。
- horizontal/Type Well GRは各curveのfinite median/MADでrobust normalizeし、missingは0埋めとmask channelを併用する。
- visible prefixに32以上のfinite matched pairがあればdeterministic Huber affine summaryを`c_w`へ入力する。未満ではneutral summaryへfallbackし、wellを除外しない。
- horizontal encoder channelは`GR`, `GR_missing`, `dGR/dMD`、Type Well encoder channelは`GR`, `GR_missing`, `dGR/dTVT`に限定する。
- `MD/X/Y/Z`はemission encoderへ入れず、固定transition/initial priorとrow identityだけに使う。これによりGR attributionを分離する。
- Type Well/context生成に他horizontal wellを使用しない。

## Architecture contract

- sample unit: complete well、batch size 1 well。
- encoder: shared-family multi-scale 1D residual encoders、embedding 64、dilation `[1,2,4,8,16]`、GroupNorm、dropout 0.10。
- prefix context: masked attention pooling、context dim 32、FiLMは各encoder blockのscale/biasだけを生成する。
- emission: normalized bilinear similarity + prefix-conditioned positive temperature。temperatureはclampし、outer-validでfitしない。
- decoder grid/transition: exp209固定値 `step=0.35`, `n_rates=41`, `rate_span=0.10`, `sig_r=0.002`, `sig_p=0.02`, `start_sig=0.75`, `r0_sig=0.01`, `band_pad=100.0`。
- training loss: fixed decoder上のGaussian soft-label structured negative log likelihood `1.0`（label observation `sigma=0.35 ft`）+ local true-state cross entropy `0.25`。loss weight / sigma gridは禁止する。
- optimizer: AdamW、learning rate `3e-4`、weight decay `1e-4`、最大8 epochs、gradient clip 1.0、AMP有効。
- early stopping: outer-train wells内のstable holdoutだけを使用する。outer-valid suffixを使用しない。
- pseudo-cut training views: outer-train wellsだけでofficial start、official startより256/512 rows前の最大3 views。cut後のtruthはlabelだけに使う。
- test-time adaptation: gradient updateなし。`c_w`を1 forward passで生成するだけとする。

より詳しい不変条件とstage分岐は実験側`architecture_contract.md`を正とする。

## 検証設計

- fold: 5-fold GroupKFold by complete well。fold mapを最初にcontent freezeする。
- valid input: organizerと同じofficial `TVT_input` visible-prefix/hidden-suffix mask。
- score rows: official hidden suffixの全row。
- outer-train pseudo-cutsはdata augmentationであり、valid scoreやconfidence gateに使わない。
- baseline/controlを再学習しない。保存済みexp209 exact HMM、exp221 HMM+LGB、exp202/178 diagnosticsを参照する。
- negative controlsは同じtrained modelを使い、Type Well GRだけをstable within-well circular shuffleしたdecode、GR unaryを0にしたgeometry-only decodeを実行する。
- primary metrics: pooled/fold OOF RMSE、real-vs-control RMSE/NLL、posterior mass within10/20、target-in-grid、grid-edge mass、1000+、hidden-like spatial/typewell-purged、by-well p95/worst、continuity、runtime/memory。

## 段階実行

### Stage A: fold 0 complete-well smoke

- 実装とGPU pushは別承認。
- active architecture 1、seed 1、trained fold 1、neural model 1。
- LightGBM config / trained fold / booster `0/0/0`、PF/HMM parent regeneration 0、control retraining 0。
- fold 0 modelは最終固定configで学習し、Stage Bで再学習せず再利用する。

2026-07-20のKaggle version 2では、hard truth path likelihoodが疎なtruth jumpを固定exp209 grammar内の単一路として表せず、epoch 1完了前にruntime failureした。ユーザー承認により、decoder/state grid/transitionは変更せず、truthを直接Gaussian label observationとして条件付けたpartitionとの差へobjectiveを修復する。これはvalid scoreを見たrescue gridではなく、学習可能性を回復する単一のdata-contract修正であり、`sigma=step=0.35 ft`を固定して比較探索しない。

PASSは次をすべて満たすこと。

- technical: finite prediction 1.0、valid truth target-in-grid 99.5%以上、prefix clamp parity最大絶対差`1e-6 ft`以下、pre-freeze valid truth access 0。
- GR evidence: real true-state NLLがcircular shuffleより0.05 nats/token以上良く、within10 posterior massが0.03以上高い。
- TVT: real posterior mean RMSEがgeometry-onlyより0.25 ft以上、同fold exp209 exact HMMより0.25 ft以上改善する。
- safety: exp209比のwell RMSE p95非悪化、最大well回帰10 ft以下。
- cost: peak GPU memory 14 GB以下、fold runtime 8.5時間以下、5-fold推定がKaggle 12時間runへ分割可能。

1条件でもFAILならStage Bへ進まず、同一OOFでarchitecture/loss/band/temperature救済gridを行わない。

### Stage B: full 5-fold OOF

- Stage A全PASSと別承認後、fold 0 modelを固定再利用してfold 1-4の4 modelsだけを追加学習する。
- active architecture 1、合計5 fold models、追加4 models、seed 1。
- negative-control modelは学習しない。

GR attribution PASSは、realがcircular-shuffleとgeometry-onlyの双方をpooled RMSEで0.50 ft以上、5/5 foldsで改善し、real true-state NLLも5/5 foldsで良いこととする。

LB 5.x promotion PASSは次をすべて満たすこと。

- pooled OOF RMSE `<=6.0 ft`。stretchは`<=5.0 ft`。
- exp221 HMM+LGBを5/5 foldsで改善する。
- 1000+、hidden-like spatial、hidden-like typewell-purgedでexp221をすべて改善する。
- well RMSE p95がexp221非悪化、exp221比最大well回帰`<=5.0 ft`。
- GR attribution PASS、finite coverage 1.0、transition/continuity guard PASS。

`6.0 < OOF <=6.75 ft`かつGR attribution PASSなら、learned whole-well alignment仮説だけを支持する。exp295内で救済せず、別expのarchitecture iteration候補として記録し、inferenceへ進まない。`OOF >6.75 ft`またはGR attribution FAILならbranchを閉じる。

### Stage C: inference

Stage B LB 5.x promotion PASSと別承認後だけ、同じexp内で5 fold modelsによるcurrent-test posterior meanを実装する。raw-test input parity、model manifest、prediction SHA、runtime、submit-checkを通過するまでsubmissionを作らない。候補bankや既存ML/PFとのblendはStage Cにも含めない。

## 実験範囲

- 対象実験: `exp295_prefix_anchored_wholewell_gr_alignment_ssm`
- Route: `ensemble`
- primary parent: `exp202_heatmap_mdn_candidate_generator_probe`
- transition parent: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- decoder reference: `exp221_lgb_oof_gaussian_emission_hmm_on_exp148`
- signal reference: `exp178_supervised_gr_window_matcher_from_known_tvt_prefix`
- negative references: `exp197_cnn_pf_likelihood_probe`, `exp244_bidirectional_prediction_start_pseudotail_augmentation`, `exp292_typewell_gr_warp_rate_identifiability_audit`
- 変更する変数: GR emissionをhand-crafted/local/candidate scoreからprefix-conditioned learned complete-well unaryへ変更する。
- 固定する変数: well split、official mask、exp209 state grid/transition、posterior mean output、neighbor-free入力、success gate。
- 2026-07-19追加実装範囲: Stage Aの別名compact self-contained train候補、fail-closed inference候補、contract tests。canonical採用、package、GPU実行、Stage B/C、submissionは含めない。

## 再現性設計

- seed policy: fixed global seed 42 + stable SHA256 per-well sample/pseudo-cut/control-shuffle key。
- stochastic 処理: PyTorch CUDA convolution、AdamW、dropout、seeded dataloader order。
- PF/Beam / likelihood-PF / seed bagging: すべて0。exp209 grammarのdeterministic exact decoderだけをStage A compact候補内へ再実装済み。
- 並列処理: DataLoader worker 0、1 well/batch、gradient accumulation 4。global RNGをworkerへ共有しない。
- GPU flags: CuDNN benchmark false、deterministic algorithms `warn_only=True`、AMP有効。deterministic submission anchorとは扱わない。
- SHA: raw input、fold map、pseudo-cut manifest、preprocessing schema/content、model state/manifest、emission/posterior/predictionのcontent SHAをfold別に記録する。
- gzip: decompressed content SHAを主証拠、raw gzip SHAを補助証拠にする。
- inference/submission: Stage C承認後だけprediction/submission SHAを記録する。
- Kaggle bootstrap: loose/package/bootstrap内configとsource SHA、kernel metadata、kernel versionを照合する。

## リスク

- リークリスク: complete-well encoderへouter-valid suffix TVTやsame-well pseudo-labelが入力される危険。input schema allowlist、mask-first loader、truth-late readoutを必須にする。
- neighbor leakage: validation wellと同じtypewell/XY groupのhorizontal pathsをcontextへ混ぜる危険。inference-side horizontal source countを常に1にassertする。
- band coverage: exp209 grid外のtrue pathはlearned emissionで回収できない。Stage Aでtarget-in-gridを測り、失敗後にvalid truthを見てbandを拡張しない。
- shortcut learning: trajectoryやknown prefix SDFだけでGRを無視する危険。trajectoryをemissionから分離し、real/shuffle/zero-GRを同一modelで必須比較する。
- compute/memory: row x TVT unaryとstructured lossが大きい。1 well/batch、streamed unary、gradient accumulationを固定し、Stage A cost gateを通す。
- CV/LB不一致: public testは3 wellsで分散が大きい。absolute CV gate、hidden-like、worst-wellを満たすまでinferenceへ進まない。
- 再現性: CUDA/AMPはbyte deterministicでない可能性がある。seed、manifest、prediction差を記録しdeterministic anchorを名乗らない。
