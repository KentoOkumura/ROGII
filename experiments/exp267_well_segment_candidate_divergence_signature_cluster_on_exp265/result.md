# exp267_well_segment_candidate_divergence_signature_cluster_on_exp265 結果

> **部分無効化:** exp263候補だけから作る18署名、occupancy、stabilityのstructure negativeは保持する。
> exp264 Stage B scoreをjoinしたwinner/calibration評価は親のfeature availability leakageにより無効化する。

## 仮説

well内の候補bank divergence trajectoryをtarget-free 18次元署名へ固定すれば、exp265のblock-length
proxyを使わずにstableなwell clusterを作れる。cluster別candidate winnerまたはexp264 calibration差が
再現するなら、既存dual selectorへ署名とsoft membershipをadd-onlyする価値がある。

## 設定

- 親: `exp265_target_free_pairwise_candidate_divergence_soft_experts_on_exp264`
- candidate source: exp263の6 primitive / 15 pairs
- post-assignment score: 保存済みexp264 Stage B
- 検証: outer well 5-fold、18 well features、RobustScaler + clip + KMeans K=3
- seed: primary 42+fold、audit 10042+fold
- 実行量: 0 variant / 0 config / 0 trained fold / 0 booster、親/control再学習0

## 結果

| メトリック | 値 |
| --- | --- |
| 実装 | 完了 |
| targeted / repository test | 4件 / 全87件PASS |
| Stage A Kaggle guard | FAIL（structure guard FAIL） |
| CV / Public LB / Private LB | 対象外 |
| Conditional Stage B | 不採用・未実行 |
| Inference / submission | disabled |

## 再現性

- deterministic anchor: false
- seed policy: fixed explicit seed + outer-fold offset
- kernel: version 2 / id_no `127573486` / private CPU / internet off
- input / feature / preprocessor / assignment SHA: `kaggle/output/train_v2/artifacts/reproducibility_manifest.json`に保存
- model / prediction / submission SHA: Stage Aでは対象外
- rerun result: 未実行

## 解釈

Kaggle Stage Aは3,783,989 rows / 773 wells / 5 folds / 18特徴、forbidden hit 0、全773 wellsの
3区間coverage、fallback 0でtechnical guardを通過した。別seed assignment一致率はfold別
`1.0000 / 0.9935 / 1.0000 / 0.9548 / 0.9935`でstability guardを通過した。

しかしpooled occupancyはlow/middle/high=`538/41/194` wellsで、middleが基準75未満だった。
outer-valid middleもfold 1/2/3で`7/3/9` wellsと基準10未満。区間別bank rangeの
low<middle<highはfold 4だけが3区間すべて通過し、profile guardは1/5 foldsだった。
candidate winner patternも5 foldsすべて異なりmodal fold count 1でFAILした。

post-assignment score側ではhigh-low calibration bias差が5/5 foldsで負、pooled actual MAEの
worstはhigh `12.7371` ftで最大誤差wellを除いてもhighのままなのでscore separabilityはPASSした。
ただし構造guardがFAILしており、この差をK=3 semantic clusterの再現的なselector根拠には使えない。

## 次

Stage A総合guardはFAILしたため、10 CPU booster selector add-only、downstream GPU、inference、
submissionは実行せず、このK=3 branchを閉じる。再訪するならclusterを救済するparameter gridではなく、
保存済み18署名を連続量として使う0-booster monotonic risk readoutを独立候補として設計する。
