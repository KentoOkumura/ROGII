---
name: kaggle-idea-forge
description: Kaggleの停滞時や次実験の発想時に、利用可能なデータ・既存実験・失敗証拠・計算制約から、parameter tuningに偏らない複数の問題表現、情報源、候補生成、融合、data generation、validation案を独立生成し、互いに異なる反証可能な実験案の組へまとめる。上位解法級の非連続なアイデア探索、失敗した情報の別用途での再評価、候補多様性の設計、参照元を隠した過去時点の記録による着想評価に使う。通常の実験優先順位整理だけならkaggle-strategyを使う。
---

# Kaggle Idea Forge

既存backlogの微修正ではなく、taskの表現と情報の使い方を組み替えた実験案を作る。実装や採用判断は行わず、反証可能なidea portfolioを`kaggle-strategy`へ渡す。

backtickで示すfield名や分類値は、このskillのJSON schemaと作業順を管理するためのリポジトリ内の管理用語であり、一般的な手法名ではない。ユーザー向けの回答では、入力、予測対象、モデル出力、損失、推論方法、処理単位、検証条件、計算条件を先に平易に説明し、必要な場合だけ管理用語を括弧内に添える。

## 入力契約

最初に次を確認する。不足しても推測で埋めず、`missing`としてideaのconfidenceを下げる。

- task、metric、提出・推論形式
- train/testで利用可能な入力とhidden-test制約
- evidence cutoff
- trusted baselineとCV split
- 既存signal、予測、OOF、model、cache
- positive evidenceとnegative evidence
- compute、runtime、deadline、submission budget

source-hidden評価が指定された場合は、ユーザーが許可したpacketだけを読む。repo探索、Web、後続実験、writeup、survey、他runの出力を読まない。許可sourceを最終出力に列挙する。

## ワークフロー

### 1. Taskを圧縮する

モデル名を出す前に、次のtask cardを作る。

1. 予測対象とmetricが罰する誤り。
2. 推論時に既知、未知、部分的に既知の量。
3. sample、row、group、sequence、graph、fieldなどの依存単位。
4. current outputが捨てている不確実性。
5. 保存すべきdomain invariantと、許される変形。
6. hiddenで変わり得るavailability、size、distribution、runtime。

詳細な実験履歴を読む前に、task cardだけから6件の`task_first`案を作る。少なくとも2件はcurrent point predictorと既存candidateを使わない。各案はmodel名ではなく`input -> target/objective -> output -> decode`で記述する。後で証拠と矛盾して棄却してよいが、このpassを既存backlogで置き換えない。

### 2. 構造的opportunity probeを実行する

すべてのtaskで次を検討し、該当しない場合も理由を記録する。固有手法を決め打ちせず、task/dataから成立条件を導く。

#### A. Same-entity context

推論対象と同じentity/group/sequenceに、label、観測、履歴、support setが部分的に与えられるか確認する。存在する場合は、単なる特徴集約だけでなく次を比較する。

- global referenceと同じ座標系のentity-specific referenceまたはcalibration domainを作る。
- observation model、matching representation、conditioner、test-time adapterの各roleへ移す。
- entity contextのcoverage内とcoverage外を分け、global referenceまたはanchorへのfallbackを固定する。
- contextが短い、欠損、範囲外、重複する場合のavailability testを作る。

#### B. Imperfect-intermediate training

model、candidate selector、refinerが別modelの予測やscoreを入力に使う場合、training時の中間入力が推論時より過度に正確でないか確認する。該当する場合は1枚の閉じた案として次を接続する。

1. OOFで中間予測の誤差振幅、自己相関、bias、欠損、mode inversionを測る。
2. ground truthまたはclean intermediateから、それらを再現するcorrupted conditioningを生成する。
3. corrupted conditioningから修正するmodelをpretrainする。
4. real OOF-like conditioningで短くfine-tuneし、clean held-groupだけで評価する。
5. copy-through、corruption mismatch、推論時runtimeをkill条件にする。

通常のinput noise、selector score augmentation、residual stackのいずれか一つだけではこのprobeを完了したと数えない。

#### C. Invariant discovery

既知の座標、保存則、単位、boundary condition、symmetryから、targetと既知量の組合せ候補を列挙する。名称だけで仮定せず、trainでvariance、微分、boundary誤差を測るcheap EDAを先に置く。支持された関係だけを次へ使う。

- invariantを保存するaugmentation。
- invariant空間のtargetまたはstate。
- constraint loss、projection、candidate prior。
- 観測だけを意図的に壊すcalibration/shift corruption。

`preserved_invariants`には、保存する量、意図的に壊す量、再計算が必要な入力を明記する。

### 3. Negative evidenceの範囲を限定する

各失敗を次のtupleで記録する。

`(signal, representation, role, fusion, validation regime, compute regime)`

必ず次を分ける。

- 実装した具体案だけを棄却できる証拠。
- 同じ使い方をした複数の実装まで棄却できる証拠。
- 異なる使い方での独立検証または既知の制約との矛盾により、情報や仕組み自体を棄却できる証拠。

negative result内のpositive submetric、oracle headroom、特定bucket、coverage、誤差非相関性を抽出する。一つでも残ればfamily全体を閉じない。

### 4. 独立した発想passを作る

利用可能ならfresh subagentを使い、各agentへtask cardと必要最小限の証拠だけを渡す。互いの案、期待解、上位解法、main agentの結論は渡さない。subagentを使えない場合も、以下を順番に独立生成し、前passの順位を次passへ見せない。

#### Pass A: representation

- current point targetをdistribution、set、ranking、structured object、latent state、pairwise relationへ置き換えられないか。
- local predictionをgroup全体のjoint predictionへ変えたとき保持できる情報は何か。
- model brandを選ぶ前に、input tensor、target、loss、decodeを定義する。
- 少なくとも5案を出し、parameter変更は禁止する。

#### Pass B: information and invariance

- 各signalを`target / observation / reference / candidate / prior / feature / conditioner / augmenter / gate / calibrator`の別roleへ移す。
- direct predictorとして弱いsignalを捨てず、どの条件で情報を持つかを問う。
- same-entity contextがある場合、entity-specific referenceとcoverage fallbackを少なくとも1案作る。
- domain invariantを保つsynthetic exampleと、deploymentで実際に起きるerrorを再現するcorruptionを作る。
- model-generated intermediateを下流modelが使う場合、corrupted conditioningからtrustを学ぶpretrain/refine案を少なくとも1案作る。
- 少なくとも5案を出す。

#### Pass C: candidate and uncertainty

- point estimate前の候補集合、score、posterior、member disagreementを保持できないか。
- candidate単体精度、truth bracketing、oracle coverage、residual correlation、target-free selectabilityを分離する。
- hard top-1、soft fusion、conditional gateを別案として比較する。
- 候補の多様性を`observation / reference / representation / dynamics / decoder / seed`で記録し、seedだけが違う集合を多様と数えない。
- 弱いが非相関な候補を残す基準と、point化前にsoftに融合する案を作る。
- 少なくとも5案を出す。

#### Pass D: validation and compute

- CVが本番と異なり得るgroup size、availability、distance、time、tail、domainを列挙する。
- oracleで到達不能なparameterizationを実装前に落とす。
- computeを増やす案だけでなく、高速化によって初めて探索可能になるalgorithmを考える。
- hidden unitあたりruntime、peak memory、determinism、offline availabilityを扱う。
- 少なくとも4案を出す。

### 5. Cross-pollinationする

独立passを匿名化してから、次の問いで組み合わせる。

- 強いrepresentationに、別passのobservationまたはpriorを入れると何が増えるか。
- oracle coverageがあるがselectabilityが弱い候補を、point化せずconditionerやuncertaintyとして使えないか。
- 平均では弱いsignalを、disagreementまたはavailabilityで条件付けできないか。
- 現在の100倍速で実行できるなら、どの近似を外せるか。
- current bestを利用禁止にした場合、別の推論objectをどう構成するか。

類似案を統合して10–14枚のidea cardにする。少なくとも4 mechanism familiesを残し、parameter-only案は最大2枚とする。

### 6. Adversarial gateを通す

ideatorと異なるcontextまたはroleで、各cardを次から壊す。

- leakage、same-OOF selection、train-only input
- public/example固有artifact、hidden cardinality
- CV splitと本番availability/domainの不一致
- oracle coverageとtarget-free selectabilityの混同
- tail、worst group、fold variance
- runtime、memory、offline dependency、stochastic reproducibility
- closest past failureとの差がparameterだけ

危険案は採点せずrejectする。family全体をrejectせず、棄却されたtupleとreopen条件を返す。

### 7. Portfolioを選ぶ

top 5は次の5枠をすべて覆う。弱い案をquotaだけで採用せず、hard gateを通る案がない枠は未解決として報告して発想passへ戻る。

1. `representation`: joint object、distribution、set、latent stateなどへの表現変更。
2. `information`: same-entity/global reference、observation、conditionerなどのrole変更。
3. `data_generation`: invariant-preserving syntheticまたはimperfect-intermediate training。
4. `candidate_generation`または`fusion_uncertainty`: 多様な候補、soft fusion、conditional uncertainty。
5. `validation`または`compute_enabler`: hidden shift、causal ablation、探索を解禁する高速化。

top 5の少なくとも1件は`task_first`、少なくとも1件は`representation_change`とする。平均scoreだけで一列にせず、次のslotを使う。

- `safe`: 既存証拠が強く、cheap testが明確。
- `exploration`: confidenceは低いが構造的upsideが大きい。
- `orthogonal`: anchorとの誤差非相関性を狙う。
- `compute_enabler`: 後続探索を解禁する。

各案を`cheap proxy -> full OOF -> inference smoke`の順にし、各stageのkill criterionを事前固定する。compute案は、解禁する下流algorithmとend-to-end accuracy runをaccept条件へ結び付ける。

## 出力

[portfolio-schema.md](references/portfolio-schema.md)のJSON schemaに従い、指定先へ`idea_portfolio.json`を保存する。人間向け回答ではtop 5比較、reject理由、未解決入力を短く示す。

保存後は次を実行する。

```bash
uv run python .agents/skills/kaggle-idea-forge/scripts/validate_portfolio.py idea_portfolio.json
```

validatorの構造PASSはideaの科学的妥当性を証明しない。source-hidden評価では、agentに期待解や採点rubricを渡さず、別のjudgeが後からmechanism recallと安全性を採点する。

idea portfolioの作成だけでは、候補を採用、実験化、またはバックログへ追加しない。このskillでは `KAGGLE_DIRECTION.md` の「検証中の仮説」「アイデアバックログ」節と `docs/backlog/` を作成・更新・削除しない。ユーザーが選んだ候補の「バックログ化」「バックログへ追加」を明示的に依頼した場合は、同じターンで `kaggle-strategy` を使い、選択したidea card、根拠、reject理由、未解決入力を引き渡す。portfolioに不足項目があれば推測せず、未決事項として引き渡す。採番、steering、実装、Kaggle実行は別のユーザー承認を必要とする。

## 停止条件

- taskまたは推論時availabilityが不明で、leakage判定ができない。
- trusted baseline/CVがなく、改善仮説を比較できない。
- source-hidden指定なのに許可sourceの境界が曖昧。
- 全案がparameter-only、または同一mechanism familyに偏る。
- top案にcheap test、kill criterion、hidden inference contractがない。

停止時は不足入力だけを返し、実装、実験作成、push、submissionを行わない。
