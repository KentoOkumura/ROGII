# exp298 後続分岐契約

## 位置付け

この文書は、exp298で検証する「exp226は局所形状を捉え、別の物理候補は大局トレンドを補える」仮説の
後続案2・3・4について、別セッションが目的、開始条件、候補集合、演算子、出力の意味を独自変更しないための
正規契約である。exp298の実装ではなく、将来の独立実験の設計境界を固定する。

この契約を変更する場合は、先行実験の結果を根拠にユーザーの明示承認を得て、この文書、exp298のrequirements.md、
`config.yaml`、`backlog/KAGGLE_DIRECTION.md`を同時に更新する。実装都合や単一foldの結果だけで分岐、候補数、
平滑化スケール、合格条件を変更しない。

## 共通不変条件

- Lateフェーズ固有の分岐、特徴量、metric、補正は扱わない。
- true suffix TVT、error、oracleはtarget-free入力、候補、score、policyのSHA freeze後の評価だけに使う。
- exp293 deployable12は削除、置換、再重み付けせず、常にsafeな原候補bankとして保持する。
- exp293とexp297の固定12候補契約および既存成果物を変更しない。後続は必ず別expにする。
- exp295は対象井自身のcomplete-well GRを使うcandidate-free learned SSMという独立枝であり、本枝へ統合しない。
- oracle offset、oracle slope、oracle candidate identityを予測、特徴量、補正値として保存しない。
- hard top1、row-wise switch、true error trigger、直接residual correction、LBを見たparameter選択を行わない。
- inference、submission、親/controlの再学習、Kaggle GPU/CPU実行は各段階で別承認を得る。

## exp298 singleton final block契約

2026-07-20のユーザー承認により、exp293固定block assignmentに含まれる長さ1の最終blockは、exp298の
affine quotient RMSE/rank/block win/strict unique-bestの分母から全候補共通で除外する。technical coverage
1.0はselected row数2以上のaffine-eligible rowsに対して要求する。長さ2以上のinvalid blockはtechnical FAIL、
singleton block/row/well数は必須記録とする。exp293のblock ID、境界、SHA、H128/H256/H512、final short block
自体は変更しない。この除外はStage 2以降のcandidate値・S512・24候補bank・PASS閾値を変更しない。

## 分岐図

```text
exp298 local-shape audit
  -> PASS: Stage 2 local/global hybrid candidate bank
       -> PASS: Stage 3 latent-registration semi-Markov posterior
            -> deployable guards PASS: inference候補化は別承認
            -> insufficient: Stage 4は自動開始しない
                 Stage 2 supportが強く、ユーザーが別途承認した場合だけnested rankerを設計・実装
       -> FAIL: branch close
  -> FAIL: branch close
```

`exp298 FAIL`または`Stage 2 FAIL`から、平滑化幅、混合係数、候補weight、block幅の救済gridへ進まない。
Stage 4はStage 3の自動救済ではなく、Stage 2で候補supportが強いのに物理evidenceだけでは回収できない場合の
高リスクな別実験である。

## Stage 2: exp226_local_shape_global_trend_hybrid_candidate_bank

### 開始条件

exp298のtechnical guardとscientific PASS条件をすべて通過した場合だけ、新しい実験番号で開始する。
exp298の一部metricだけが良い場合、`exp226_post_u`だけが良い場合、H128だけが良い場合は開始しない。

### 仮説

exp226 pre-U pathのH512-scale高周波成分を局所形状として固定し、exp293各候補のH512-scale低周波成分を
大局トレンドとして組み合わせると、原12候補を保持したままH256/H512 block oracle supportを広げられる。

### 固定入力

- 局所source: exp298で固定した`P226_preU = tvt_geop + gr_delta`だけ。
- 大局source: exp293 version 2 support PASS時のdeployable12を同じ順序で使用する。
- 各wellの`T0`: prediction suffix直前にある最後のfinite `TVT_input`。target-free phaseで固定する。
- 行順、fold、well、suffix row、H128/H256/H512 blockはexp293の固定割当を再利用する。
- 原12候補はsafe bankとしてそのまま残し、12本のhybrid候補を追加する。評価bankは合計24本。

### 固定低周波演算子 `S512`

well内suffix relative pathを`r_i = P_i - T0`とする。`S512`は次のtarget-free決定的演算だけを使う。

1. row順を固定し、各行`i`で`j in [max(0,i-256), min(n-1,i+256)]`のfinite `r_j`をfloat64平均して
   `m_i`とする。これは中心半径256、最大513行のboxcarで、端では存在する行だけを使う。
2. `S512(r)_i = m_i - m_0 + r_0`とし、低周波pathの先頭を元pathの先頭へ固定する。
3. 欠損、行重複、非finite平均が1件でもあればfallbackせずtechnical FAILとする。

kernel、boxcar半径、端処理、先頭anchor、dtypeを変更しない。H256/H512は評価horizonであり、`S512`自体を
H256へ差し替えない。平滑化scale、kernel、local/global weightのgridは禁止する。

### 固定候補式

```text
R226 = P226_preU - T0
L226 = R226 - S512(R226)

Rc = Pc - T0
Gc = S512(Rc)

Hc = T0 + Gc + L226
```

`c`はexp293の12候補で、候補名は`hybrid_local226__global_<candidate>`とする。weightは常に1.0/1.0で、
local amplitude、global amplitude、offset、slopeをtruthから再調整しない。

### Stage 2 readoutとPASS

- primary: 24候補bankのH512 block oracle RMSE。
- secondary: H256、whole-well、5 folds、1000+、hidden-like spatial、hidden-like typewell-purged、
  by-well p95/worst、candidate choice count、hybrid unique-best fraction。
- exp293と同様、候補bankとblock assignmentをSHA freezeしてからtruthをjoinし、oracle predictionを保存しない。

次をすべて満たした場合だけPASSとする。

- H512 oracle RMSEがexp293原12候補の`3.683762664246268`から少なくとも`0.05 ft`改善する。
- H512 oracle RMSEが原12候補を5 folds中4 folds以上で改善する。
- H256、whole-well、1000+、hidden-like spatial、hidden-like typewell-purgedのoracle RMSEが原12候補bankから非悪化。
- hybrid候補がstrict unique-bestとなるH512 block比率が`>=0.05`。
- finite coverage 1.0、原12候補の値/content SHA parity、truth freeze、row/block identity guardをすべて通過する。

1条件でもFAILならStage 3へ進まず、本枝を閉じる。original12をhybridで置換する、hybridだけに絞る、
local sourceをpost-Uへ変更する、S512を調整する、候補blendを追加する救済は行わない。

## Stage 3: exp226_hybrid_bank_latent_registration_semimarkov

### 開始条件

Stage 2がfull PASSした場合だけ、新しい実験番号で開始する。Stage 2の24候補bankとその順序、S512、
row/block identityを変更しない。

### 仮説

局所GR evidenceと大局的な候補path継続性をcandidate×registration×reliabilityのjoint stateとして時間統合すれば、
局所形状を保ちながらH256以上で大局トレンド候補を安全に切り替えられる。

### 固定状態と観測

- candidate states: Stage 2の原12 + hybrid12 = 24 states。
- registration states: `[-20,20] ft`を2 ft刻みの21 states。
- reliability states: `reliable / unreliable`の2 states。
- base block: H128。candidate minimum durationは2 blocks=H256。H512は継続性guard。
- local evidence: Type Well/horizontal GRのHuber affine後residual、NCC、chain-rule derivative residual、
  exp226 donor distance、`gr_delta` magnitude/variation。すべてtarget-freeで生成する。
- global evidence: candidate identity persistence、block endpoint continuity、`dTVT/dMD`差、曲率差、
  original/hybrid family transition flagだけを使う。
- registration offsetは観測側の潜在変数であり、candidate TVTへ直接加えない。
- reliabilityが不足するblockは原12の`exp226_w500_50_50`へ固定するsafe stateを持つ。
- calibrationはknown-prefix内pseudo-cutだけで行い、outer-valid suffix truthを使わない。
- solverはexact log-space forward-backwardまたは同値なsemi-Markov posteriorとする。
- 出力はcandidate TVTのposterior mean。Viterbi path、hard top1、registration補正TVTは出力しない。

exp297のtarget-free registration primitiveを再利用する場合も、exp297の原12候補契約や成果物を変更せず、
Stage 3内で24候補に対する新しいscore/manifestを別SHAで固定する。

### Stage 3 PASS

- pooled OOF RMSE `<=6.9 ft`。
- exp263 fixed anchor `8.2383315465`を5/5 foldsで改善する。
- 1000+、hidden-like spatial、hidden-like typewell-purgedをすべてanchorから改善する。
- well RMSE p95がanchor非悪化、worst-well regressionが`<=5.0 ft`。
- prediction finite coverage 1.0、prefix continuity、slope、curvatureのphysical guardをすべて通過する。
- current testで必要なcandidate、GR、Type Well、geometry入力をtarget-freeに再生成できる。

PASSしてもinference/submissionは別承認とする。FAIL時にtransition、duration、registration grid、evidence weight、
temperature、S512、candidate bankを同一OOFで調整しない。

## Stage 4: exp226_hybrid_bank_nested_block_ranker

### 開始条件

Stage 2のcandidate supportはfull PASSしているが、Stage 3がdeployable guardを満たさず、物理evidenceだけが
律速と判断された場合に限る。ユーザーの別途明示承認と新しい実験番号が必須で、自動開始しない。

### 固定nested設計

- Routeは`ensemble`。ML rankerとPF/Beam由来24候補の両方が予測生成に本質的に寄与する。
- candidate bankはStage 2の24本を固定する。
- outer splitは既存5 folds、各outer-train内でinner 4 folds。outer-validのtarget/errorをfit、feature、
  threshold、early stoppingへ使わない。
- 学習単位はcandidate×blockで、H256とH512を別rowとして扱う。
- targetは3つに固定する: block SSE回帰、within10率回帰、candidate間pairwise preference。
- 各objectiveは1固定LightGBM configとし、最大`5 outer × 4 inner × 3 objectives = 60 boosters`。
  親/controlの再学習は0。
- inference featureはStage 3のtarget-free local/global evidence、candidate disagreement、continuity、
  registration posterior、reliability、candidate familyだけ。true error/oracle featureは禁止する。
- 3 objectiveのranker出力は固定等重みでstandardizeし、Stage 3 semi-Markovのcandidate unaryへ加える。
- rankerからhard candidateを選ばず、最終出力はsemi-Markov candidate TVT posterior meanとする。

### Stage 4 PASS

- pooled OOF RMSE `<=6.9 ft`かつexp263 fixed anchorを5/5 foldsで改善する。
- Stage 3 physical posteriorよりpooled RMSEを少なくとも`0.05 ft`改善し、5 folds中3 folds以上で改善する。
- 1000+、hidden-like spatial、hidden-like typewell-purged、p95、worst-well、continuity guardをStage 3から非悪化。
- outer/inner leakage監査、feature schema/content SHA、60 model manifest SHA、OOF prediction SHAをすべて記録する。

PASSしてもinference/submissionは別承認とする。direct row residual correction、hard top1、candidate pruning、
outer-valid targetによるfeature/threshold/grid選択、Stage 2/3 contractの上書きは禁止する。

## 将来実験を作る際の必須参照

Stage 2・3・4を実験化するセッションは、実装前にこの文書を全文参照し、steeringと`config.yaml`の
`lineage.references`へこのpathを記載する。契約をそのまま実装できない場合は作業を止め、ユーザーに確認する。
