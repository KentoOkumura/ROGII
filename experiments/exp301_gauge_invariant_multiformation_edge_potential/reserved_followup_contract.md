# exp301 reserved follow-up contract

## この文書の役割

この文書は案2と案3のsingle source of truthである。両案はexp301のvariantではなく、必要条件を満たした後に
別exp番号で作る後続実験である。現時点では番号、実装、Kaggle実行を予約しない。別セッションで設計を再開する場合も、
ここに書かれた入力、自由度、gate、禁止事項を暗黙に変更しない。変更が必要なら、実装前にこの文書とsteeringを明示改訂する。

## 案2: implicit_horizon_evidence_on_exp293_bank

### 開始条件

- exp301 Stage 0 technical PASSかつStage 1 direct quality/candidate noveltyの両方がPASSしている。
- exp301 OOF potential prediction、component/support、calibrated residual scale、SHA manifestが保存されている。
- exp293 fixed deployable12 candidate bank、順序、H512 block assignment、SHAがそのまま再利用できる。
- いずれかが欠ければ案2を開始しない。exp301 FAILを案2のselectorで救済しない。

### 固定する設計

- Route: `pf_beam`。
- 候補bank: exp293 Stage 1 deployable12を値・formula・順序ごと固定。追加・削除・再生成しない。
- 粒度: exp293と同じprediction suffix起点の非重複H512、末尾short blockを含む。
- evidence: 各candidateの`U_candidate=TVT_candidate+Z`とexp301 posterior fieldの差に対する、
  exp301 variance-calibrated Gaussian negative log densityのblock和だけを使う。
- selection: block energy最小の1候補。tieはexp293 candidate order。
- exp301 direct predictionを13番目候補として混ぜない。GR、ML、true TVT、oracle ID、existing errorはscore入力にしない。
- weight/temperature/threshold/gridは持たない。variance calibrationはexp301 outer-train residualだけで固定する。
- control: saved exp263/exp293 fixed anchor `exp226_w500_50_50`。parent/controlは再学習・再生成しない。
- negative control: well/component対応をouter-valid truthを見る前にstable permutationしたpotential evidenceを1本だけ評価する。

### 成功条件

- selected H512 pooled RMSEがanchor 8.2383315465から`>=0.20 ft`改善する。
- anchorより5/5 foldsで改善する。
- distance 1000+、hidden-like spatial/typewell-purged、by-well p95が非悪化、worst delta`<=+0.25 ft`。
- real evidenceがstable-permutation controlよりpooledかつ5/5 foldsで良い。
- 1項目でもFAILならselector weight、variance、horizon、候補subset、thresholdを同じOOFで救済せず閉じる。

### 禁止事項

- candidate生成、GR evidence、learned selector、oracle feature、posterior直接補正、rowwise選択、softmax blend。
- exp301 FAIL時の開始、exp293 bank変更、H128/H256/whole-wellへの事後primary変更。
- inference/submissionへの自動進行。PASS後も別承認を必要とする。

## 案3: implicit_horizon_prior_redecode_on_exp295

### 開始条件

- exp301が案1の全Stage 1 gateをPASSしている。
- exp295が自身の事前登録Stage B promotion gateをPASSし、5-fold OOF neural unary、state grid、transition、
  decoder posterior、model/input SHAを保存している。
- exp295がFAILまたは未完了なら案3を開始しない。exp295をexp301 priorで救済しない。

### 固定する設計

- Route: `ensemble`。exp295 learned GR unaryとexp301 physical priorが予測生成へ本質的に寄与するため。
- exp295の学習済みneural emission、state grid、known-prefix hard clamp、transition、decoderを完全固定し再学習しない。
- exp301 priorは各stateの`U_state`とexp301 field meanのvariance-calibrated Gaussian NLLとして加算する。
- prior varianceはexp301 outer-train calibration artifactから一意に決め、alpha、temperature、clip、thresholdを追加しない。
- decoderはexp295と同じfixed exact forward-backward posterior meanを1回再実行する。Viterbi/top-k/candidate bankへ変更しない。
- controlはsaved exp295 OOF。fold 0を含めneural model再学習0、LightGBM/PF/Beam 0。

### 成功条件

- pooled RMSEがsaved exp295 OOFから`>=0.20 ft`改善する。
- exp295より5/5 foldsで改善する。
- distance 1000+、hidden-like spatial/typewell-purged、by-well p95が非悪化、worst delta`<=+0.25 ft`。
- GR attributionはexp295 Stage Bで既にPASS済みであることを前提とし、案3でgeometry-only/shuffleとの差を悪化させない。
- 1項目でもFAILならprior weight、variance、state band、transition、decoder、blendを同じOOFで救済せず閉じる。

### 禁止事項

- exp295再学習、architecture/loss/sigma/state-grid変更、candidate bank接続、ML blend、test-time backprop。
- exp301 direct predictionとのposthoc平均、oracle row selection、案2scoreの流用。
- exp301またはexp295のFAILを覆すための開始。
- inference/submissionへの自動進行。PASS後も別承認を必要とする。

## 分岐順序

1. exp301 Stage 0を実装・実行しtechnical判定する。
2. Stage 0 PASS時だけexp301 Stage 1を実装・実行し、direct qualityとcandidate noveltyを判定する。
3. exp301 PASS後、案2はexp293 fixed bank側の独立分岐として設計可能になる。
4. exp301 PASSかつexp295 Stage B PASS後、案3はexp295 decoder側の独立分岐として設計可能になる。
5. 案2と案3は互いの先行条件ではなく、同一expへ統合しない。

