# 設計

## 結論

exp111 score系27列は取り込み可能だが、保存済みfold0モデルを使う方式では取り込まない。
downstream TVTの各outer fold内でexp111の2 scorerを4-fold nested学習し、outer-trainにはinner OOF、
outer-validにはouter-trainだけで学習した4 model平均を渡す。これで27列だけをexp287の421特徴へ
add-onlyし、予測価値をleakageなしで評価する。

## 仮説

exp111のwithin10 probability / expected absolute errorは、5つのPF/Beam系候補の局所的な信頼度を
表し、既存のraw candidate値やtarget-free compact特徴だけでは表現しきれない誤差構造を補える。
過去の27列が無効だった理由はfeature familyではなく、fold0 scorerを全trainへ適用したnon-OOF生成
なので、strict nested化すればexp287からglobal RMSEを改善しつつtail guardを回復できる可能性がある。

## 実験範囲

- 対象実験: `exp396_fold_safe_exp111_score_27_addonly_on_exp287`
- Route: `ml_model`
- 親実験: `exp287_fold_safe_formation_74_addonly_on_exp264`
- clean tail control: `exp264_exp263_candidate_confidence_dual_selector`
- 変更する変数: strict nestedで再生成したexp111 score系27列のadd-only
- 固定する変数:
  - exp287のclean 273 + nested compact 74 + fold-safe formation 74 = 421特徴
  - exp287のouter 5-fold group split、score rows、row identity、target residual
  - exp287のLightGBM configs `[0, 1, 2]`、early stopping `250`、seed `42`
  - exp111の5 candidates、48入力特徴、binary / L1の2目的、model hyperparameters
  - valid/OOF/scope/by-wellの非加重RMSE

Routeは `ml_model` とする。PF/Beam由来値は補助特徴であり、最終TVTはdownstream LightGBMが生成する。

## 比較対象

| 役割 | 成果物 | 用途 |
| --- | --- | --- |
| 主control | exp287保存済みOOF / fold / scope / by-well | 421特徴からの純増分を比較 |
| clean tail control | corrected exp264保存済みOOF / by-well | exp287で悪化したtail guardを判定 |
| scorer仕様参照 | exp111 source / config / schema | 候補、48入力特徴、2目的、model設定だけを固定 |
| new variant | exp396 fold-safe score 27 add-only | Stage Bで448特徴を評価 |

- exp287 OOF RMSE: `8.136708220359452`
- exp287 OOF SHA256: `8f026c5c5f6508fb142981832994c6ba9cded4940168c648a9df9f3e698c3913`
- exp287 model manifest SHA256: `419dbdf83dd6bc343f0265aca56dd690ba1f231ee419e7cf0ff456ffdb797590`
- exp287 metrics SHA256: `435434342494aaa62cee6e627809363ac34f16174973f4b81301d2923f780862`
- exp287 fold metrics SHA256: `864eca0452eea578c96baa653d25c4f2ae241c84b8e5d659b277407b5e427141`
- exp287 by-well SHA256: `3562cec13abe3c3df496e57d71b46aeb592ea2022c7bf0b9b5df1e062c21024d`
- corrected exp264 OOF SHA256: `b11c5005ca566f76588f4e1735386c15b8f016b874701a82e1c0741c8b839ae2`

入力SHAが一致しない場合はfit前に停止し、control boosterは再学習しない。

## 固定27列

列順も契約に含める。

1. `ll_learned_prob_top1_index`
2. `ll_learned_error_top1_index`
3. `ll_learned_prob_top1_value`
4. `ll_learned_prob_top2_value`
5. `ll_learned_prob_margin_top1_top2`
6. `ll_learned_prob_entropy`
7. `ll_learned_error_top1_value`
8. `ll_learned_error_top2_value`
9. `ll_learned_error_margin_top2_top1`
10. `ll_learned_prob_likpf_rank`
11. `ll_learned_error_likpf_rank`
12. `ll_learned_prob_top3_contains_likpf`
13. `ll_learned_error_top3_contains_likpf`
14. `ll_learned_prob_pf_ancc`
15. `ll_learned_pred_abs_error_pf_ancc`
16. `ll_learned_prob_beam_mean`
17. `ll_learned_pred_abs_error_beam_mean`
18. `ll_learned_prob_likpf_mean`
19. `ll_learned_pred_abs_error_likpf_mean`
20. `ll_learned_prob_sc_ens`
21. `ll_learned_pred_abs_error_sc_ens`
22. `ll_learned_prob_hyb`
23. `ll_learned_pred_abs_error_hyb`
24. `ll_learned_prob_weighted_tvt_minus_last_known_tvt`
25. `ll_learned_prob_weighted_tvt_minus_likpf_mean_tvt`
26. `ll_learned_error_weighted_tvt_minus_last_known_tvt`
27. `ll_learned_error_weighted_tvt_minus_likpf_mean_tvt`

既存clean 273に含まれるraw candidates、`candidate_tvt_std/range`、multi-observation列は重複追加しない。
exp264監査でこの27列に依存するとされたGRWR 6列も追加しない。

### 27列の導出式

candidate順を `pf_ancc, beam_mean, likpf_mean, sc_ens, hyb`、scorer出力を
`p_j = within10 probability`、`e_j = max(predicted abs error, 0)`、candidate TVTを `t_j` とする。
rank/top-kの同値時は固定candidate順を優先する。

- probability側のtop1/top2、margin、likpf rank、top3 flagは `p_j` の降順。
- error側のtop1/top2、margin、likpf rank、top3 flagは `e_j` の昇順。
- entropyは `-sum_j clip(p_j, 1e-6, 1) * log(clip(p_j, 1e-6, 1))`。
- `learned_prob_weighted_tvt = sum_j(t_j*p_j) / sum_j(p_j)`。
  分母が `<= 1e-6` のときだけ分母を1とする。
- `learned_error_weighted_tvt = sum_j(t_j*w_j) / sum_j(w_j)`、
  `w_j = 1 / max(e_j, 1e-3)`。
- 最後の4列は上記2 weighted TVTから `last_known_tvt` と
  `last_known_tvt + likpf_mean_d` をそれぞれ引く。
- 13 summary列、候補別probability/error 10列、weighted差分4列をこの順でfloat32化する。

この式はexp145の生成契約を固定化したものだが、exp145のbatch medianとexp111保存modelは継承しない。

## Stage A: strict nested score生成

### Fold契約

downstream outer foldを `o=0..4`、そのouter-train内のinner foldを `i=0..3` とする。

1. outer-valid wellsを除いたouter-train wellsだけを4 GroupKFoldへ割り当てる。
2. 各 `(o, i)` でinner-trainだけからscorer入力48列のmedianをfitする。
3. 同じinner-trainでwithin10 classifierとabsolute-error regressorを1個ずつ学習する。
4. inner-validへそのmodel pairを適用し、outer-trainのscore coreをinner OOFで埋める。
5. outer-validへ4個のinner model pairを適用し、candidateごとの出力を単純平均する。
6. 10個のscore core（5 candidates × probability/error）から固定式で27列を導出する。

outerごとに `4 inner × 2 objectives = 8`、全体で40 CPU boostersとなる。全trainでfitしたmodel、
保存済みexp111 fold0 model、outer-validを学習に含むmodel、batch medianはいずれも禁止する。

### Current-test契約

Stage Bが全gateをPASSし、別途inference実装が承認された場合だけ生成する。downstream outer fold `o`
のTVT modelへ渡すcurrent-test scoreは、そのouter foldで保存した4 inner model pairの出力平均とする。
full-train scorerの追加fitは行わず、outer foldごとのtrain時のfit境界をそのまま維持する。

### 保存単位

- 保存の物理coreはouter-role別の10 score値でよい。27列は固定式で再構成しlogical SHAを記録する。
- 40 model、各modelの48-feature schema/order、48 medians、best iteration、model SHAをmanifestへ保存する。
- outer/inner train-valid well集合とrow identity SHA、10 coreと27列のschema/logical content SHAを保存する。
- 全partitionをimmutable `id` で並べ、row/well/full coverageとduplicate IDを検証する。

### Subsampleと再現性

exp111と同じくcandidate-long学習の元になるwide rowを各inner fitで最大350,000行に固定する。
seedは `SHA256("exp396|outer=<o>|inner=<i>|candidate_long")` から導出したlocal RNGへ渡す。
sample前にrow identityをstable sortし、global RNGとthread schedulingを使わない。同じrow sampleを
2目的で共有し、sample row ID SHAを保存する。

## Stage A gate

全項目をAND条件とする。

### Technical / leakage gate

- outer/inner train-valid well overlapがすべて0。
- score rowのrow/well/full coverageが1.0、duplicate IDが0。
- label join前のfeature生成・imputation・sample決定でtarget、true TVT、errorを読まない。
- 10 coreと27列が全件finiteで、列順・式・schema SHAが固定値と一致する。
- scorer model数40、median vector数40、48-feature schema数40が過不足なくmanifestにある。
- 各outer-valid scoreが、そのwellを一度も学習に含まない4 modelだけから生成される。
- projected CPU runtime `<= 30,600 sec`、peak RSS `<= 25 GB`。超過時はStage Bへ進まない。

### Scorer quality gate

outer-validの5-fold連結値について、outer-train candidate priorとの比較を固定する。

- expected-error MAEがpooledで改善し、5 outer folds中4 folds以上で改善。
- within10 loglossがpooledで改善し、5 outer folds中4 folds以上で改善。
- within10 Brierがpooledで改善し、5 outer folds中4 folds以上で改善。

threshold、candidate、model設定、feature subsetを結果後に変更して救済しない。

## Stage B: downstream TVT add-only

Stage A全gate PASSとユーザーの別承認後だけ実装・実行する。

- source surface: exp287の421特徴
- added surface: strict nested exp111 score 27
- final surface: `273 + 74 + 74 + 27 = 448`
- variant: `fold_safe_exp111_score_27_addonly`
- LightGBM configs: 3 (`0`, `1`, `2`)
- folds: 5
- 合計予定: 15 GPU boosters
- exp287 / exp264 control再学習: 0 boosters
- runtime: Nvidia Tesla T4、internet off、`gpu_use_dp=true`、`deterministic=true`、
  `force_col_wise=true`、threads 8

親exp287の10 fold-role formation cacheを再利用し、feature schema/order/content SHAを検証する。
formation、nested compact、candidate生成を本実験の結果を見て変更しない。

## Stage B promotion gate

全項目をAND条件とする。

1. exp287比pooled OOF delta RMSE `<= -0.02 ft`。
2. 5 folds中4 folds以上でexp287以下。
3. near / mid / 1000+ / hidden-like spatial / hidden-like purgedの各scopeが
   exp287比 `<= +0.02 ft`。
4. by-well delta RMSE p95がexp287比 `<= 0.00 ft`。
5. corrected exp264比worst-well delta RMSE `<= +0.25 ft`。
6. corrected exp264比 `+1/+3/+5 ft`悪化well数が `135/39/14` を超えない。

exp287はglobal RMSEを改善した一方、worst-well `+8.228410 ft`、悪化well数
`140/40/19`でtail guardを失敗した。したがってexp287比のglobal改善だけで採用せず、
clean exp264に対するtail回復も必須とする。FAIL時は結果を記録して閉じ、inference候補にしない。

## Preflight契約

Stage A実装時はbooster fit前、Stage B実装時はGPU fit前にfail-closedで確認する。

- exp287 / exp264 control SHA、row identity、outer fold、target、score rowsが一致する。
- exp111仕様参照の候補5、入力48、目的2、model paramsが固定contractと一致する。
- 旧27列、旧exp111 model、GRWR 6列が入力surfaceに存在しない。
- 27列の名前・順序・式が固定allowlistと一致し、421列との重複がない。
- Stage Aは40 CPU boosters、Stage Bは15 GPU boosters、control再学習0である。
- run approval flagsがfalseならfit/package/pushを開始しない。

## 本実験に含めないもの

- 保存済みexp111 fold0 modelの予測利用
- global 5-fold OOF score cacheの全outer foldへの共用
- full-data scorer refit、batch/full-train median
- dependent GRWR 6列
- scoreによるhard top1、probability/error加重TVT、direct blend
- score feature subset、candidate追加、objective/model/threshold/grid変更
- sample weight、hard-well rescue、formation/selector/compact変更
- exp287/exp264 control再学習
- gate緩和、同一OOFでの救済、inference、submission

## 再現性設計

- seed policy: outer/inner splitとLightGBMはfixed seed `42`。subsampleは上記stable SHA256 local seed。
- stochastic処理: CPU LightGBMとcandidate-long row subsample、Stage B承認後のGPU LightGBM。
- PF/Beam / likelihood-PF / seed bagging: PF/Beam値は保存済みtarget-free入力としてのみ使い、
  PF/Beamやseed bagを再実行しない。
- 並列処理: sample row集合をsingle-threadで先にfreezeし、model fitのthreadsと切り離す。
- deterministic flags: Stage AはCPU threadsを固定、Stage Bはexp287のT4 DP/deterministic設定を維持。
  GPU runをbitwise deterministic anchorとはみなさない。
- feature SHA: gzipはdecompressed content SHAを主証拠にし、Parquetはfile SHAと
  id+float32 logical SHAを両方保存する。
- model/prediction SHA: Stage A manifestと40 model、Stage B承認後の15 model manifestとOOFを保存する。
  inference/submissionは未承認なので現時点ではSHA生成対象外。
- bootstrap: package/push承認後にembedded config、source、schema、input artifact、
  Kaggle metadataのSHAとinternet/GPU設定を照合する。

## リスク

- リークリスク: global OOF scoreはdownstream outer-valid wellsをscorer学習に混入させ得るため使わない。
- CV/LB不一致: 3 test wellsに対し5-fold平均のscore/model構成が移る保証はない。train gateと
  inference判断を分離する。
- ランタイム/メモリ: 40 scorer fitとcandidate-long frameが大きい。wide rowを固定上限でsampleし、
  outer/inner単位でlong frameを解放し、8.5時間/25GB gateで停止する。
- 再現性: multi-thread LightGBMとGPU LightGBMはbitwise一致を保証しない。sample ID、median、
  schema、model、OOF SHAを保存し、rerunは別承認にする。
- 事後選択: 27列のsubsetやthresholdを見て選ばず、fixed add-only variant一つで判定する。

## 現在の承認境界

2026-07-25の後続指示でStage A実装を行い、さらに明示確認後に正規train notebook採用、
Kaggle private CPU package/push、0-booster preflightを実行した。version 1
（id_no `128540844`）は3,783,989 rows / 773 wells / 20 nested fold rolesの16/16 technical
checksをPASSし、booster、prediction、submissionは0だった。後続のKaggle private CPU version 2で
固定40 CPU boostersを完了し、technical 22/22、scorer-quality 6/6、runtime/memory gateを
すべてPASSした。後続の明示承認でStage Bを実装し、private T4 version 1で固定15/15 GPU boostersを
完走した。OOFは`8.134294735`でexp287比`-0.002413486 ft`だったが、固定promotion gateは
1/6項目だけPASSしたためbranchを閉鎖した。inference、submissionは未承認・未実行である。

## 次のアクション

exp287をtrain-side parent anchorに維持する。exp396のsame-OOF rescue、gate緩和、
再学習、inference、submissionへ進まない。
