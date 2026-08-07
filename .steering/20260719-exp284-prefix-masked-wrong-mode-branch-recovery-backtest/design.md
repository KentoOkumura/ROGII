# 設計

## 結論

real unknown suffixのoracle failure条件を使わず、known prefix末尾をmaskしたcontrolled wrong-mode
injectionでbranch-and-verifyの回復能力とfalse switchを測る。exp283のproposal/evidence契約を変えず、
safe baseを常に残す。

## 実験範囲

- 対象: `exp284_prefix_masked_wrong_mode_branch_recovery_backtest`
- Route: `pf_beam`
- 親contract: exp283固定proposal/evidence仕様をself-containedに再実装
- Kaggle実行: exp283非依存のstandalone backtestとしてユーザー明示override承認済み
- geometry / emission: exp226 `tvt_geop` increment / exp209 raw-GR Gaussian likelihood
- 変更する変数: true unknown suffixではなくmasked known-prefix suffix、wrong mode injection、paired control。
- 固定する変数: exp283 K=3、GR前処理、proposal rank、H=256 verifier、geometry veto、freeze順序。
- 対象外: decoder update、HMM/PF再生成、model fit、current-test生成、submission。

## Stage 0: pseudo cutとmask

well内のcontiguous finite `TVT_input` prefix終端を`k`とする。`cut = k - 640`を1 well 1 cutに固定し、
`cut+1 ... k`の`TVT_input`をloader直後にNaNへ置換する。cut以前が512行未満、mask 640行が
揃わない、GR/MD/geometry identityが欠けるwellは固定理由付きでineligibleとする。

truth-aware raw frameとtarget-free masked frameを別objectにし、generation APIはmasked frameしか
受け取らない。mask manifestを保存・hashし、post-cut TVT access counterを0に固定する。

## Stage 1: GR-supported wrong mode injection

cut以前の最後128 visible rowsに対して、safe geometry pathへexp280 fixed bank
`[-80,-40,-20,-10,-5,-2,0,2,5,10,20,40,80] ft`を加え、exp209 typewell log-likelihoodを集計する。

- reference mode: shift 0。
- alternative eligible: `|shift| >= 10 ft`。
- selected wrong mode: eligible local maximumの尤度最大。なければeligible全体の尤度最大。
- tie: shift-bank order。

selected shiftはvisible cut anchorに対する意図的な誤modeなので、truthを見ずに確実に10ft以上離れる。
safe baseとwrong activeは次で作る。

`safe[r] = TVT_input[cut] + tvt_geop[r] - tvt_geop[cut]`

`wrong[r] = safe[r] + selected_shift`

wrong branchをcutから128行activeとして保持し、eventを`cut+128`に置く。decoderを実際に更新する
のではなく、fixed candidate pathsを生成するだけである。

## Stage 2: self-GR proposal

eventまでのmasked frameをexp283 proposal APIへ渡す。known donorはcut以前だけ、prediction donorは
eventより256行前までだが、このcut設定では通常空bankとなる。空bankは失敗ではなくsource coverageへ
記録する。17/31/51行、forward/reverse、dedup、global top-3、tie-breakを変更しない。

alternative branchはmatched visible donor TVTをanchorとし、event以後をexp226 incrementで延長する。
stable shuffled controlはexp283と同じper-event source-local seedでdonorだけを置換する。

## Stage 3: future evidenceとpolicy readout

event後128/256/512行でbranchごとのexp209 cumulative mean log-likelihoodを計算し、exp283 geometry vetoを
適用する。primary selectionはH256 score最大、全alternative veto時はsafe baseへ戻る。

固定5 policies:

1. wrong active only
2. safe + wrong
3. safe + wrong + real self-GR top-3
4. safe + wrong + shuffled self-GR top-3
5. no-injection safe base + real self-GR top-3

policy 5はactive branchをsafeにしたpaired safety controlであり、同じcut/event/evidenceを使う。
全target-free branch/evidence tableとcontent SHAを固定してからtruth frameをidentity joinする。

## Stage 4: post-freeze評価

- injection: selected shift、likelihood margin、wrong-active H128/H256/H512 RMSE。
- pair verifier: `LL_safe - LL_wrong`で`safe RMSE < wrong RMSE`を予測するAUC/accuracy。
- recovery: policy別selected RMSE、wrong-only gain、pair-onlyに対するself-GR incremental gain。
- persistence: H128/H256/H512の選択列、earliest recovery、再slip率。
- safety: no-injectionでbase unique-bestなのにalternateへ移るfalse switch。
- controls/scopes: real vs shuffled self-GR、fold、hidden-like、shift sign/magnitude、source/orientation、by-well/worst。

post-freeze `truth-best`は評価labelだけであり、branch生成・score・policy選択には戻さない。

## 固定guardと判断

- eligible >=100 wells / all 5 folds、finite/identity 1.0、post-cut truth-before-freeze 0。
- pair AUC >=0.60 in 5/5、safe choice pooled >=0.60かつ各fold >0.50。
- full H256 gain vs wrong-only >=0.10ft / 5 folds改善。
- full H256 gain vs safe+wrong >=0.02ft / 3 folds改善。
- H512でH256 gain非消失、no-injection false switch <=5%。
- real self-GRがshuffledより良く5/5 fold非悪化。

全PASSだけがtriggered decoder backlogを実装候補へ上げる。partial PASSは許可にしない。

## 実行契約

- active backtest variants: 1
- fixed policies: 5（学習variantではない）
- model / LightGBM config / trained fold / booster: 0 / 0 / 0 / 0
- HMM / PF regeneration: 0 / 0
- parent/control retraining: 0
- runtime: Kaggle CPU、GPU/internet off、single process
- inference/submission: disabled / disabled
- Kaggle push approval: true。1 variant / 5 policies / 0 config / 0 fold / 0 boosterを確認済み。
- dependency status: `user_authorized_independent_execution`。元のexp283 gateは履歴として残す。
- monitoring: push後の定期監視は行わず、ユーザーの完了連絡後にlogs / guardを記録する。

## 再現性設計

- pseudo cut、shift bank、tie、event、branch orderは決定的。
- real pipeline RNGなし、shuffled donorだけstable SHA256 local RNG。
- mask manifest、injection table、proposal table、evidence tableのschema/content SHAを別保存する。
- gzipはdecompressed content SHAを主証拠とする。
- post-cut TVT access counterとtruth attachment timestamp/orderをcontractへ保存する。
- model/prediction/submissionは作らないためSHA対象外。
- diagnosticでありdeterministic prediction anchorとは呼ばない。

## 実装時の境界明確化

- 保存済みexp226 OOFは元のunknown suffixだけを持つため、masked prefixへ直接joinしない。保存済みfold
  assignment / kappaと、held-out foldを除いたsource-well geometry fieldを使い、pseudo cutをanchorとして
  exp226 `tvt_geop`増分を再生する。
- wrong shiftのcut直前128行score referenceは、その区間でtest-timeに観測可能な`TVT_input`とする。
  pseudo cutより前のexp226 pathは保存済みOOFに存在しないため、暗黙のbackcastは行わない。
- proposal preprocessingはeventでframeをtruncateしてからGR補間とtrailing rolling mean 5を行い、
  centered smoothingによる`event+1`以降の混入を禁止する。
- source-well true TVTはexp226のfold-safe geometry field構築にだけ使う。現在held-out foldのtarget readerは
  `TVT`をloadせず、全target-free table freeze前のheld-out post-cut truth accessを0にする。

## リスク

- masked known prefixはofficial suffixより簡単な可能性がある。hidden-like、prefix length、geometry spanを
  readoutし、成功をそのままLB改善と解釈しない。
- safe baseが強すぎるとself-GR incremental valueが出ない。その場合はdecoderへ進まない。
- typewell evidenceはwrong modeの生成にも使うが、生成はcut前、検証はevent後でrowを分離する。
- prediction donor bankが空になりやすい。exp283契約を変えてgapを縮めず、known-source backtestとして記録する。
- cut/mask/horizon gridは同一truthへの適応になるため禁止する。
