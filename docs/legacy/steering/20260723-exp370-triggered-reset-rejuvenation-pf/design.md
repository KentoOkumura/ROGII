# 設計

## アプローチ

Stage 0はexp072互換PFを500 particles / seed index 0で全773 wellsへ流し、row ESSを記録する。
visible prefixでrobust calibrationしたGR changeがq99.5以上、かつESS/N`<=0.20`のANDをtriggerとし、
trigger後512行は再triggerしない。各outer foldのatlasはouter-train wellsだけで構築し、
trigger前後256行GRのZNCCでtop3を選ぶ。candidate TVT間は10 ft以上離す。

trigger、atlas manifest、top3 proposal、scoreのSHAをfreeze後にtruthをjoinする。triggerのbad10 AUCと
circular control、atlas top3のwithin10 coverageとbase likpf比coverage gainを評価する。
exp231結果は流用せず、独立Stage 0の全gateが必要。

実装時のatlas内部表現はexp231のfold-safe prototype方式を参照し、256行のGR patchを
32点へ圧縮してZNCCを計算する。source centerは32行stride、TVTは2 ft bin、
well/bin最大6 patch、bin当たりouter-train 2 wells以上とする。queryはtrigger中心の
前後128行で、同点はZNCC降順・TVT昇順、10 ft未満の候補を除外してtop3を選ぶ。

bad-eventは保存済みexp072 `likpf_mean`のtrigger起点128行RMSEが10 ft以上と定義する。
atlas coverageのbase比較も保存済みexp072を使う。1-seed PF predictionはESSを得る
診断系列であり、saved base controlを置換しない。

Stage 1は各seedのtrigger時に通常resamplingを450粒子へ行い、残り50粒子をatlas top3へ
`17/17/16`配分する。position jitterは0.10 ft、rateは現posterior weighted medianを中心に
sigma 0.001。rejuvenated source ageを128行追跡するがdynamicsには作用させない。
総粒子500、128 seeds、exp072 dynamics/likelihood/mean aggregationは固定する。

## 実験範囲

- 対象: `exp370_triggered_reset_rejuvenation_pf`
- Route: `pf_beam`
- 親: `exp072_exp063_full_replay_feature_cache`
- 変更: target-free trigger時の10% atlas proposal rejuvenationだけ。
- 固定: exp072 PF本体、500 particles、128 seeds、GR likelihood、noise、mean aggregation。
- Stage 0 gate: trigger AUC`>=0.60`、circular差`>=0.05`、trigger率`[0.001,0.10]`、
  top3 within10 coverage`>=0.60`、base比coverage gain`>=0.10`、4/5 folds、
  hidden-like 2面正方向。
- Stage 1 gate: exp072比`>=0.05 ft`、4/5 folds、1000+/hidden-like/p95回帰`<=0.02 ft`、
  worst`<=0.25 ft`。

## 再現性設計

- seed: `SHA256(experiment|fold|well|family|seed_index)`からlocal RNG。
- global RNG、thread schedule依存、fold間atlas共有は禁止。
- stochastic: baseline PF、systematic resampling、normal/atlas jitter。
- CPU single worker、GPU off、上限30,600秒。
- raw trainはouter-fold atlas、raw testは全train atlasとして別生成する。
- atlas/trigger/proposal/predictionのcontent SHA、gzipのdecompressed SHAを記録する。
- Stage 0はouter-train donor TVTをatlas構築に使用できるが、各foldのvalidation wellは
  source集合から必ず除外し、target well truthはSHA freeze後だけ読む。

## リスク

- leakage: valid donor、truth/error trigger、oracle candidate選択。fold exclusionとfreezeで防ぐ。
- CV/LB不一致: atlas coverageとtrigger頻度差。
- runtime: Stage 0でも773 PF seed-well runs、Stage 1は98,944。
- reproducibility: per-seed ESS triggerとproposal配分順を固定する。
- science: exp231がatlas evidenceを支持せず、現時点は保留。

## 2026-07-25 Stage 0結果

- Kaggle private CPU version 2で773 / 773 diagnostic seed-well runsを完了。
- technical gateはPASS。target truth / hidden roleのfreeze前accessは0、
  donor-fold leakageも0。
- accepted triggerは13 / 3,685,818 eligible rows、率`3.527e-6`。
  AUCは`0.499998`、circular差は`-3.76e-12`。
- atlas top3 within10 coverageは`0.076923`、saved likPFは`0.846154`、
  coverage gainは`-0.769231`。passing foldsは`0/5`。
- scientific gateはFAILし、Stage 1は不適格。同じtrigger / atlas /
  rejuvenation設定のthreshold・top-k救済を行わずbranchを閉じる。
