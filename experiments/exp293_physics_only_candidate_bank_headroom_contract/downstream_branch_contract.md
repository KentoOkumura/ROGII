# exp293 後続分岐契約

## 位置付け

この文書は、物理モデル単体でPublic LB 6.5を目指すexp293系列の第2・第3・第4段階について、
別セッションが目的、開始条件、候補集合、出力の意味を独自変更しないための正規契約である。

変更する場合は、exp293の実行結果を根拠にユーザーの明示承認を得て、この文書、対象実験のrequirements.md、config、
`backlog/KAGGLE_DIRECTION.md`を同時に更新する。実装都合だけで分岐や合格条件を変更しない。

## 共通不変条件

- Routeは`pf_beam`。ML predictor、ML selector、学習済みTVT model、MLとのblendを使わない。
- horizontal GRは推論時に観測可能な入力であるというAssumptionを置く。
- true suffix TVT、error、oracleはtarget-free candidate/score/policyのSHA freeze後の評価だけに使う。
- safe pathまたはexp293 fixed anchorを常に保持する。
- Type Well registration offsetは観測側潜在変数であり、最終TVTへ直接加えない。
- hard top1、row-wise switch、oracle selection、LBを見たparameter選択を行わない。
- inference、submissionは各段階のfull guard通過後に別承認を得る。

## 分岐図

```text
exp293 support PASS
  -> Stage 2 latent-registration GR evidence
       -> PASS: Stage 3 joint physical smoother
       -> FAIL: stop。Stage 4へ自動分岐しない

exp293 support FAIL
  -> Stage 4 candidate birth
       -> exp293と同じheadroom contractを新bankで再監査
            -> support PASS: Stage 2
            -> support FAIL: stopまたは別仮説をユーザー判断
```

## 2026-07-26 ユーザー承認による独立分岐

exp297でStage 2は科学FAILとなり、上記の元契約どおりStage 3へは進まず閉じた。
その後、exp399でdocking依存のgeometry復帰がwrong modeで弱くなる構造、
exp370でtrigger resetの識別不能が確認された。これらの新しい失敗証拠を受け、
ユーザーは元Stage 3の自動継続ではない次の独立分岐を明示承認した。

```text
exp405 geometry-reinjected interval semi-Markov
  input: exp293 saved deployable12 only
  geometry floor: docking/trigger independent
  PASS: same exp405 current-test implementation eligible
  technically valid scientific FAIL:
        exp406 GR-first loop-closed multi-well RGT fixed16 Stage 0
  technical ERROR:
        exp405のtechnical issueだけを修正して同じcontractを再実行
```

- exp405はexp297のFAILをPASSへ変更せず、元Stage 2 evidenceも再利用しない。
- exp405は元Stage 3の`candidate × registration × reliability` stateを実装せず、
  H256/H512 interval candidate posteriorとunconditional geometry floorを検証する
  独立仮説である。
- exp406はexp386の閾値救済ではなく、Formation-derived graphを使わない
  horizontal-GR-first topology familyである。
- 正規仕様はexp405/406それぞれの現行実験記録を正とし、廃止前の契約確認が必要な場合だけ`docs/legacy/steering/`を参照する。
- current-test、inference、submissionは各gate PASSと別承認が必要である。

## Stage 2: prefix_calibrated_latent_registration_gr_evidence

### 仮説

exp293の物理candidate pathを変更せず、Type Well/horizontal GRの局所登録ずれとGR不一致を観測側へ
分離すれば、H256/H512 blockでtruth-best candidateをtarget-freeに順位付けできる。

### 固定入力と観測モデル

- candidate bank: exp293 support PASSに使ったprimary 12候補。その後の候補追加は禁止。
- known-prefix calibration: 最後の最大512 rows、finite pair最小64。
- affine: Type Well GRからhorizontal GRへのHuber IRLS、2回、slope clip `[0.25,4.0]`。
- residual scale: `1.4826 * MAD`、clip `[10,60]`。
- registration grid: `[-20,20] ft`、2 ft刻み、21 states。範囲外不一致はoffset拡張せずoutlier componentで扱う。
- horizon: H128/H256/H512、primary H256、H512を継続性guardにする。
- score成分: Student-t raw residual (`df=4`)、NCC、chain-rule derivative residualの3成分。
- component統合: candidate×registration内median/MAD z-score後に等重み`1/3`。
- registration prior: `q(delta) proportional exp(-abs(delta)/10)`。
- GR reliability: inlier/outlier 2成分。outlierはcandidate間で同一のbroad likelihoodとし、
  GR不一致時にcandidateを強制移動させない。
- prior: reliable stateではcandidate 12本を一様、registrationは
  `q(delta) proportional exp(-abs(delta)/10)`、reliable priorは`0.9`に固定する。
  unreliable stateは`exp226_w500_50_50`だけに質量を置くsafe fallbackとする。
- prefix finite pairが64未満、affine/scaleが非finite、Type Well局所分散または勾配が不足する場合は、
  そのwell/blockをunreliableとしてsafe fallbackへ固定する。閾値はexp292のfixed contractを継承しgrid化しない。
- negative control: stable within-well circular-shuffle。real scoreと同じ候補数・horizon・maskを使う。
- 出力: candidate/block evidence、registration posterior、entropy、mode gap、reliabilityだけ。
  selected TVT prediction、補正TVT、submissionは作らない。

### Stage 2のPASS判定用readout

- componentのmedian/MAD標準化は、各well/block/componentの12 candidate×21 registration state集合内で行う。
- target-free scoreをSHA freezeした後、registrationとreliabilityを周辺化してblockごとの`p_b(c)`を得る。
- truth join後の評価用expected candidate SSEを
  `E[SSE_b] = sum_c p_b(c) * SSE_b(c)`と定義する。これはcandidate TVTの平均やselected predictionではなく、
  固定evidenceがtruth-good candidateへ置いた確率質量のreadoutである。
- H256 headroom recoveryは
  `(SSE_anchor - sum_b E[SSE_b]) / (SSE_anchor - SSE_H256_oracle)`とする。
  denominatorが非正または非finiteならFAILとする。
- 1000+、hidden-like、shuffle比較も同じexpected candidate SSE/readoutで評価する。
- hard top1、posterior mean TVT、row predictionを生成・保存しない。safe fallback以外のcandidate tieはexp293順とする。

### Stage 2 PASS

- H256でoracle SSE headroom recovery `>=0.35`。
- recoveryが5 foldsすべてで正。
- realがcircular-shuffleをpooledかつ5/5 foldsで上回る。
- H512 recoveryがH256から`0.05`を超えて悪化しない。
- exp293で記録した1000+、hidden-like spatial/typewell-purged risk flag各面でanchor非悪化。
- feature/score/registration posterior SHA freeze前のtruth access 0。

1条件でもFAILならStage 3へ進まない。candidate不足ではないためStage 4へ自動分岐しない。

## Stage 3: joint_physics_candidate_registration_semimarkov_smoother

### 仮説

Stage 2のcandidate evidenceを、candidate identity、観測registration、GR reliabilityのjoint posteriorとして
時間統合すれば、誤ったregistrationをTVTへ転写せず、物理candidate bankのoracle headroomを回収できる。

### 固定状態と推論

- candidate states: Stage 2と同じ12候補。
- registration states: Stage 2と同じ21 states。
- reliability states: `reliable / unreliable`の2 states。
- base block: 128 rows。candidate minimum durationは2 blocks=256 rows。
- candidate transition: TVT endpoint continuity、`dTVT/dMD`差、曲率差だけを使う。
- registration transition: same/adjacent stateを基本とし、candidate TVTへregistrationを加えない。
- reliability transition: persistence priorをknown-prefix pseudo-cutだけから推定する。
- hyperparameter calibration: outer-valid suffix truthを使わず、known-prefix内H128/H256/H512 pseudo-cutだけを使う。
- solver: exact log-space forward-backwardまたは同値なsemi-Markov posterior。Viterbi/hard top1は禁止。
- final output: candidate TVTのposterior mean。registrationは周辺化し、出力へ加えない。

### Stage 3 PASS

- pooled OOF RMSE `<=6.9 ft`。
- exp263 fixed anchorを5/5 foldsで改善。
- 1000+、hidden-like spatial、hidden-like typewell-purgedをすべて改善。
- well RMSE p95がanchor非悪化、worst-well regression `<=5.0 ft`。
- prediction finite coverage 1.0、physical continuity guard PASS。
- raw-testに必要なcandidate/GR/typewell/geometry inputがすべて再生成可能。

PASSしてもinference/submissionは別承認とする。parameter grid、candidate追加、ML接続でFAILを救済しない。

## Stage 4: mode_loss_triggered_candidate_birth_beam

### 開始条件

exp293 scientific support FAILの場合だけ開始する。Stage 2 FAIL、Stage 3 FAIL、単一subgroup risk flagだけを
理由に開始しない。

### 固定役割

- candidate生成だけを検証し、候補選択・score・decoder・submissionを同時に変更しない。
- exp293 primary12を削除せずsafe bankとして保持する。
- exp293のoracle nearest residualをpost-freeze診断し、candidate support不足の形をconstant offset、linear rate、
  curvature/change-pointへ分解する。
- 新規candidate familyは、SSE説明率が最大かつ4/5 foldsで同じfamilyが最大となる1 familyだけを選ぶ。
- family tie順は`offset -> rate -> curvature/change-point`。4/5 foldsで一致しなければStage 4は実装しない。
- offset family: bounded observation-independent path offset branches。Type Well best shiftをtruthやscoreから直接採用しない。
- rate family: exp268の固定`tail30/w32/w64/w128/w256` initial-rate grammarを候補生成へ使う。
- curvature/change-point family: H256境界で最大1回だけrate/curvature regimeを変えるsafe-anchored beam。
- 生成候補はrow/block/whole-well oracleを読む前に全件freezeし、true TVTでbranchを生成・pruneしない。

### Stage 4 PASSと戻り先

- 新bankのH512 oracle RMSE `<=5.5 ft`。
- exp293 primary12 H512 oracleを5/5 foldsで改善。
- new-best candidate rateがpooledで`>=5%`。
- 1000+とhidden-like 2面のoracleを悪化させない。

PASS後も直接Stage 2へ進まず、exp293と同じheadroom contractを新bankで再実行する。
再監査support PASS後だけStage 2を新bankに固定し直す。

## 次の開始判断

exp293実行前はStage 2/3/4をいずれも実装しない。exp293 scientific support判定を唯一の最初の分岐入力とし、
PASSならStage 2だけ、FAILならStage 4だけを新しい実験のrequirements.mdと実験番号へ切り出す。

## 明示的な非目標

- exp264/ML downstream add-only feature化。
- Type Well best offsetのTVT直接補正。
- exp281/290のparameter救済。
- exp291/292の結果を見てhorizon、weight、shift gridを後付け変更すること。
- 複数candidate birth familyの同時投入。
