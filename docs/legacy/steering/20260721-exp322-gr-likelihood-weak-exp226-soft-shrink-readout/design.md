# 設計

## 2026-07-21 実装反映

凍結設計をcompact self-contained train sourceへ反映した。exp280互換性のためGaussian row scoreはexp280と同じ`-0.5*min(zscore^2, 600)`で計算し、configの`row_log_likelihood_clip: [-600, 0]`は承認済みlegacy contract labelとしてhard guardする。canonical train/inference Notebookは上書きせず、compact候補Notebookを別名で生成する。生成物は以下の契約9点だけとし、outer-train thresholdはblock gate表、raw per-well SHAはinput manifestのaggregate evidenceへ含める。

## アプローチ

保存済みexp263固定blendを基準にし、exp226 K16を中心とするGR emission likelihood landscapeだけを新しく計算する。GRがshiftを識別できず、かつshift 0が尤度上棄却されていないH512 blockだけ、exp263からexp226方向へ固定量のsoft shrinkを行う。

```text
p_base(i) = 0.50*p226(i) + 0.25*p_likpf(i) + 0.25*p_hmm(i)

weak(b) = [margin(b) <= Q20_outer_train(margin)]
          AND [entropy_norm(b) >= Q80_outer_train(entropy_norm)]

admissible226(b) = [zero_rank(b) <= 3]
                   OR [zero_gap(b) <= Q20_outer_train(zero_gap)]

gate(b) = weak(b)
          AND admissible226(b)
          AND [observed_gr_share(b) >= 0.80]

p_new(i) = p_base(i)
           + gate(block(i))
           * 1[md_since_last_known(i) >= 250]
           * clip(0.25 * (p226(i) - p_base(i)), -10, 10)
```

`Q20/Q80_outer_train`は各評価foldについて残り4 foldsのtarget-free block分布だけから計算する。test inferenceへ進む場合は同じexp322内で設計を更新し、full trainのtarget-free分布からthresholdを作るが、今回はinferenceを設計・実装しない。

## 実験範囲

- 対象実験: `exp322_gr_likelihood_weak_exp226_soft_shrink_readout`
- Route: `pf_beam`
- 親実験: `exp263_last_anchor_better_candidate_confidence_pair_cache`
- anchor / shrink先: `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction`
- likelihood parity親: `exp280_exp226_shift_likelihood_separability_readout`
- 変更する変数: exp226 K16中心のGR弱区間gateを通した固定bounded shrinkの有無だけ。
- 固定する変数: exp263 formula、exp226 prediction、exp263 readout fold identity、exp226元OOF source fold identity、shift bank、H512、emission、threshold quantile、admissibility、coverage、near veto、alpha、clip、control、評価scope。

## データフロー

```text
exp263 Stage 0 cache ── materialize exp226_w500_50_50 ── p_base ──┐
                                                                  ├─ fixed shrink ─ target-free candidate freeze
exp226 saved OOF ───── exp226_k16 / source-fold audit / identity ─┤
             └──────── 13 shifted paths ─ exp280-parity GR score ─ gate ┘

raw train/typewell ─── GR emission only
hidden-like assignment ────────────────────────────────┐
true suffix TVT ── late readout after SHA freeze ──────┴─ metrics / decision
```

readoutのfoldは親exp263 cacheの保存済み`outer_fold`を使う。exp226 OOFは別のgroup splitで生成されているため、元`fold`は各wellで一意かつ`[0,1,2,3,4]`を満たすsource identityとして監査するが、exp263 foldとの一致は要求しない。exp226予測がexp263 cache内`exp226_k16`と`1e-5 ft`以内で一致することを別guardにし、新しいsplitやrefitは行わない。

## GR尤度と弱区間の定義

### 固定likelihood bank

- 中心: exp226 final K16 prediction `p226`。exp280の`tvt_geop`中心scoreは入力に流用せず、実装parityの技術参照に限定する。
- shift bank: `[-80,-40,-20,-10,-5,-2,0,2,5,10,20,40,80] ft`。
- block: 未知suffix先頭から非重複512行、short tail保持。
- emission: exp209/exp280 Gaussian raw-GR/typewell emission。
- sigma: known-prefix residual std、clip `[10,60]`。
- missing GR: exp280と同じboth-direction interpolation後typewell mean fallback。
- score: row log-likelihoodを`[-600,0]`相当へclipし、block内meanを取る。
- tie: config上のshift順。

### block特徴

shift別block scoreを`ell_j`とする。

```text
margin = ell_top1 - ell_top2
prob_j = softmax(ell_j - max_j ell_j)
entropy_norm = -sum_j(prob_j * log(prob_j)) / log(13)
zero_gap = ell_top1 - ell_shift0
zero_rank = fixed tie policy下のshift0順位
observed_gr_share = raw入力で有限だったGR行率
```

絶対`ell_top1`はwellごとのsigma、欠損、block長の影響が大きいためgateに使わない。低marginだけでは隣接shiftの平坦さを過剰検出するため、高entropyとのANDに固定する。

## Gateと補正

- ambiguity thresholdは各foldのouter-train blockだけから`margin q20`、`entropy q80`を計算する。
- exp226 admissibilityは`zero_rank<=3 OR zero_gap<=outer-train q20`。
- `observed_gr_share<0.80`は「GRが弱い」ではなく「観測不足」として非発火。
- block gateはhardだが、予測変更は固定`alpha=0.25`のsoft shrink。1行の移動は`10 ft`でclipする。
- `md_since_last_known<250 ft`は必ずbase parityを保つ。
- H512境界taperは入れない。境界不連続はdiagnosticとして記録し、同一OOFでtaperを救済追加しない。

## Negative control

各wellでblock gate列を非zero circular shiftする。block数`B>1`なら、

```text
offset = 1 + int(sha256("exp322|" + well_id)[:8], 16) mod (B - 1)
```

とし、同一well内の発火block数を保存する。`B=1`のwellはcontrolでもreal gateを維持して別集計し、差分がないことを明示する。controlは候補選択や提出に使わず、GR landscapeと場所の対応が真に効いたかだけを測る。

## Truth境界

1. raw identity、exp263 readout fold、exp226元OOF source fold、exp263 base、exp226 anchorを読み、禁止列がないことを確認する。
2. 13 shift scoreとblock特徴を作り、outer-train threshold、real/control gate、候補予測を作る。
3. input manifest、score contract、block table、candidate tableのschema/content SHAを確定する。
4. 別late-readout関数だけがtrue TVTとhidden-like roleを結合する。
5. suffix truth、error、truth-nearest shift、oracle improvementは1--3の入力に戻さない。

## 評価と判定

### 技術guard

- 3,783,989 expected rows / 773 wells / folds `[0,1,2,3,4]`、identity coverage 1.0。
- exp263 fixed formula parity最大絶対差`<=1e-5 ft`。
- finite score / finite prediction / shift-bank coverage 1.0。
- target-free contract SHAとlate-readout側SHAが一致。
- changed row率`1%--25%`、changed wells`>=50`、changed folds`>=4`。満たさなければ科学的PASSではなく`INCONCLUSIVE_COVERAGE`。

### 科学的PASS

- overall RMSE gain vs exp263 `>=0.02 ft`。
- fold改善`>=4/5`。
- activated subset RMSE gain`>=0.10 ft`。
- near `0--250 ft` prediction bitwise parity。
- 1000+、hidden-like spatial、hidden-like typewell-purgedのRMSE delta`<=0.00 ft`。
- by-well RMSE delta p95、worst deltaとも`<=0.00 ft`。
- `real overall gain - circular-control overall gain >=0.02 ft`。

1つでもFAILならbranchを閉じ、alpha、quantile、block、clip、admissibility、emissionのsame-OOF rescueはしない。PASSは同じexp322でinference設計へ進む許可条件にすぎず、自動的な推論・提出承認ではない。

## 生成物契約

将来の実装では次だけを生成する。

- `exp322_input_manifest.csv`
- `exp322_target_free_shift_scores.csv.gz`
- `exp322_target_free_block_gate.csv.gz`
- `exp322_target_free_predictions.csv.gz`
- `exp322_score_contract.json`
- `exp322_fold_metrics.csv`
- `exp322_scope_metrics.csv`
- `exp322_by_well_metrics.csv`
- `exp322_summary.json`

model、booster、test prediction、submissionは生成しない。

## 再現性設計

- seed policy: real likelihood/gate/shrinkはRNGなし。negative controlだけstable SHA256 per well local決定。
- stochastic 処理: 実質なし。stable circular shiftはhash決定で再現可能。
- PF/Beam / likelihood-PF: 保存済みexp263 primitive/formulaの値だけを読む。PF/HMMを再実行しない。
- 並列処理: well単位並列を許す場合もglobal RNGを使わず、出力をcanonical `(fold, well_id, row_idx)`順へsortしてhashする。
- runtime: Kaggle private CPU、GPU/TPU/internet off。0 booster。
- SHA: exp263 cache manifest、exp226 OOF、raw/typewell、hidden-like input、feature schema/content、候補predictionを記録する。gzipはdecompressed content SHAを主証拠とする。
- model/submission SHA: 非該当。
- Kaggle bootstrap: 実装承認後のpackage時にNotebook、config、必要helper、settings/projectを列挙して照合する。現時点ではpackageしない。
- deterministic anchor: いいえ。train-side readoutであり、PASSしてもinference再現性確認までは提出anchorにしない。

## リスク

- リークリスク: feature quantileはtarget-freeだがfold全体を混ぜるとtransductiveになるためouter-train-onlyに固定する。truthはSHA freeze後にlate joinする。
- 科学的リスク: ambiguity自体がbad regimeを意味しないことはexp133で確認済み。zero admissibilityとmatched controlで「曖昧だからexp226」という飛躍を監査する。
- tailリスク: exp281のように一部wellで大幅悪化しうるため、p95/worst非悪化を必須にする。
- CV/LBリスク: Public LB 3 wellsとOOF 773 wellsは分布が異なる。PASS後もCVだけでsubmitせず、raw-test再生成・SHA設計を同じexp内で別承認する。
- runtimeリスク: exp280相当の13-shift scoringはCPU数分規模を想定するが、入力解決とtypewell補間のメモリを事前計測する。
- 再現性リスク: exp263 formulaとexp226 anchorのversion取り違えをSHA hard guardで拒否する。
