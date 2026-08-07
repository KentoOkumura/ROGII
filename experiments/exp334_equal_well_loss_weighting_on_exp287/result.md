# exp334_equal_well_loss_weighting_on_exp287 結果

## 結論

Kaggle T4 train version 2は15/15 boostersを完了した。CVは`8.09349752413077`でexp287比`-0.04321069622868201 ft`、5/5 foldsと全scope gateを通過した。一方、固定したwell-tail AND gateのうちby-well p95、worst-well、`+3/+5 ft`悪化well数が不合格だったため、exp334は**非昇格**として閉じる。追加train、inference、submissionは実行しない。

## 固定設定

- 親: `exp287_fold_safe_formation_74_addonly_on_exp264`
- clean tail control: `exp264_exp263_candidate_confidence_dual_selector`
- 変更: outer-train weight `N / (W * n_w)`だけ
- 検証: exp287と同じ5-fold group split、group=`well`、非加重RMSE
- 特徴: clean 273 + nested compact 74 + fold-safe formation 74 = 421
- 実行量: 1 variant × 3 configs × 5 folds = 15 GPU boosters、control再学習0
- validation weight: なし

## 実行結果

| 項目 | 値 |
| --- | ---: |
| Kaggle kernel | version 2 / id_no `128110184` / `COMPLETE` |
| Runtime | `21882.805369142 sec`（約6時間4分43秒） |
| OOF rows / wells | 3,783,989 / 773 |
| Models | 15 |
| exp334 CV | `8.09349752413077` |
| exp287 CV | `8.136708220359452` |
| 差分（exp334 - exp287） | `-0.04321069622868201 ft` |
| Public / Private LB | 未実行 / 未実行 |

### Fold

| Fold | exp334 | exp287 | 差分 |
| ---: | ---: | ---: | ---: |
| 0 | 7.996714884 | 8.070368347 | -0.073653463 |
| 1 | 8.175088577 | 8.255838432 | -0.080749855 |
| 2 | 7.892278533 | 7.893630011 | -0.001351477 |
| 3 | 8.091827262 | 8.106566731 | -0.014739468 |
| 4 | 8.305461844 | 8.349625947 | -0.044164103 |

5/5 foldsがexp287以下で、minimum 4/5 gateを通過した。

### Scope

| Scope | exp287比delta | Gate |
| --- | ---: | --- |
| near 0–250 | +0.003786852 | PASS |
| mid 250–1000 | +0.011048519 | PASS |
| 1000+ | -0.050434507 | PASS |
| hidden-like spatial | -0.072137859 | PASS |
| hidden-like typewell-purged | -0.068352712 | PASS |

### Well-level tail

| Gate | 観測値 | 条件 | 判定 |
| --- | ---: | ---: | --- |
| by-well delta p95 vs exp287 | +0.429584617 | ≤ 0 | FAIL |
| worst-well delta vs exp264 | +7.156485377 | ≤ +0.25 | FAIL |
| +1 ft悪化well数 | 133 | ≤ 135 | PASS |
| +3 ft悪化well数 | 40 | ≤ 39 | FAIL |
| +5 ft悪化well数 | 19 | ≤ 14 | FAIL |

exp287のworst-well `+8.228409822 ft`からは約`1.071924446 ft`改善し、`+1 ft`悪化well数も140から133へ減った。しかし、重いtailをclean control水準へ戻せず、仮説は部分的支持に留まる。

## Promotion判定

| Check | 判定 |
| --- | --- |
| pooled RMSE budget vs exp287 | PASS |
| 4/5 folds以上が非悪化 | PASS |
| 全scope RMSE budget | PASS |
| by-well p95非悪化 | FAIL |
| worst-well vs exp264 | FAIL |
| 悪化well数がexp264以下 | FAIL |
| **固定AND gate** | **FAIL** |

## 成果物監査

- 非model成果物11件とmodel 15件をmanifest SHAと照合し、全件一致した。
- OOF 3,783,989行、773 wells、5 folds、pooled/fold RMSE、by-well件数を実ファイルから再計算した。
- OOF SHA256: `7c0bab3e24d72116bf955220b7b53c66b29afed7a0c8a3f093cb97d63d033afa`
- model manifest SHA256: `8d2212b64bef1147967f68255a469965c0d60dd502726973746efaebeb816174`
- reproducibility manifest SHA256: `782d10868ac10bb54a38442bfc561be7d95e19a59d2beb8db6fd8adac7b0aacd`
- feature schema SHA256: `c1327324d6e0719eab45b9f8841033dd6cf09dd09228b044e6e8cc85f0fa8413`
- Kaggle logs SHA256: `68dc71b80709c352e24db78b881a1810311825cb31087c68cc76aed65ae3e15b`
- deterministic anchor: false。GPU bitwise rerun parityは主張しない。

## 解釈と次の扱い

well均等lossはglobal RMSE、全fold、長距離、hidden-likeを改善し、tailの一部も軽減した。しかし、severe tailを解消できなかったため、exp287のtail悪化をrow数比例のloss寄与だけでは説明できない。同一exp内のweight式gridやguard緩和は行わない。

既存バックログの0-booster `exp287_fold_safe_formation_tail_attribution_readout` は、exp334のtail不十分時だけ再開する条件を満たした。これは別途ユーザー確認後に設計・実行する候補であり、exp334からinferenceやsubmissionへは進まない。
