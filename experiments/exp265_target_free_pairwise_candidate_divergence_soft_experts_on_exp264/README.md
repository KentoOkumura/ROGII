# exp265_target_free_pairwise_candidate_divergence_soft_experts_on_exp264

> **結果無効:** regime入力そのものにtraining-onlyの`ANCC/ASTNU/ASTNL/EGFDU/EGFDL/BUDA`を含み、
> score readoutも無効なexp264 Stage Bに依存する。occupancy/stabilityを含む数値はhidden-safeな診断に使わない。

## 状態

- ルート: ensemble
- 状態: Stage 0 version 2完了・feature availability leakageにより結果無効
- Stage 0 CV: 無効
- Public LB: scope外
- Private LB: -
- Submit ID: -
- 作成日: 2026-07-17
- 親実験: `exp264_exp263_candidate_confidence_dual_selector`
- 候補source: `exp263_last_anchor_better_candidate_confidence_pair_cache`

## 仮説

global selectorが全well・全trajectory regimeを1つの写像で扱うより、正解TVTを使わずに観測できる
候補パス間の開き方、交差、傾き差、順位変化と非TVT raw contextからregimeを作り、regime別に
calibrationやbest candidate familyが異なることを先に確認した方が、安全なsoft expert設計へ進める。

## 変更点

- exp263の6 primitiveだけを読み、線形結合candidateを重複成分として使わない。
- 6本の全15 pairについて、signed/absolute gap、傾き変化、first-diff相関、zero crossing、
  sign persistence、divergence expansionを512-row blockで集約する。
- bank range、rank switch、centered path matrixの特異値構造を追加する。
- horizontal wellの`MD/X/Y/Z/ANCC/ASTNU/ASTNL/EGFDU/EGFDL/BUDA/GR`だけをraw contextに使う。
- candidate absolute TVT、last-known TVT、true TVT、TVT_input、target、error、oracleをregime特徴から除外する。
- outer foldごとにouter-train median補完、RobustScaler、KMeans K=3をfitし、outer-validへ
  hard labelとsoft membershipを出す。
- exp264 candidate-long OOFはassignment確定後にbatch集計し、regime別のcandidate RMSEと
  expected-error calibration biasだけを読む。
- Stage 0は0 booster。conditional Stage 1の30 CPU boostersは無効で、別承認とする。

## 検証方針

- Fold: exp263 canonical outer well 5-fold。
- Group: `well`。
- Block: well内evaluation row先頭から固定512行。
- Regime: K=3、outer-train-only robust scaling + KMeans、seed `42 + outer_fold`。
- Stability: 同じouter-trainを別seedで再fitし、centroid matching後のouter-valid一致率を測る。
- Leakage Check: exp264 score、candidate winner、actual errorはregime fit後のreadoutにだけ使う。
- 昇格条件:
  - 各regimeが4/5 foldsで100 wells以上かつblock share 10%以上。
  - centroid-matched assignment agreementが70%以上。
  - 少なくとも2 regimeでbest primitive familyが異なる、またはregime間calibration bias差が0.25 ft以上。

## 実行量

| Stage | variant | model config / objective | fold training | booster | control再学習 |
| --- | ---: | ---: | ---: | ---: | --- |
| Stage 0 separability audit | 0 | 0 | 0 | 0 | なし |
| Conditional Stage 1 | 3 regimes | 2 objectives | 5 | 30 CPU | なし、保存済みexp264をfallback |

Stage 1は`enabled=false`であり、Stage 0全guard通過後も別の実行承認が必要。

## 実行入口

- 学習notebook: `exp265_target_free_pairwise_candidate_divergence_soft_experts_on_exp264_train.ipynb`
- 推論notebook: `exp265_target_free_pairwise_candidate_divergence_soft_experts_on_exp264_inference.ipynb`
- Kaggle準備: `make prepare-kaggle-notebooks EXP=exp265_target_free_pairwise_candidate_divergence_soft_experts_on_exp264 EXTRA_ARGS="--strict"`
- notebook実行: Kaggle kernel runを正とする。Stage 0は`execution.run_approved=true`承認済み。

## 結果

| メトリック | 値 |
| --- | --- |
| Stage 0 guard | FAIL（occupancyのみFAIL） |
| rows / wells / blocks | 3,783,989 / 773 / 7,787 |
| block occupancy | regime 0: 2.20%、regime 1: 5.28%、regime 2: 92.53% |
| assignment stability | 1.000（PASS） |
| mean max soft probability | 0.997257（実質hard assignment） |
| candidate-family / calibration | 2 family、bias range 1.013964 ft（数値guardはPASS、解釈上は採用しない） |
| CV / LB | Stage 0では対象外 |

## 所見

### 良かった点

- 候補パス間の差を補助特徴ではなくregime定義の主成分として実装した。
- exp264の411MB級candidate-long artifactを全件DataFrame化せず、Parquet batchで集計する。
- hard routingをせずexp264 global selectorをfallbackとする当初設計だったが、そのselectorは無効化した。
- 3,783,989行、773 wells、exp264 candidate-long 22,703,934行を0 boosterで完走し、
  target-free schema、入力SHA、全fold assignment、生成物SHAを保存できた。

### リスク / 注意

- unsupervised regimeがcandidate性能差と対応する保証はないため、Stage 0 guardを通らなければbranchを閉じる。
- first-diff correlationはpathの共通trendも含む。絶対TVTそのものは使わないが、純粋なgap統計とは役割が異なる。
- K、block window、temperatureを同じOOFでgrid化しない。
- Stage 0ではinference、submission、expert学習を行わない。
- KMeansは6.6%前後の「512行に満たないwell末尾block」を`raw MD range/end/std`で分離し、
  0--2.2%の極小clusterは主に`selfgr_hmm_a070`対`exact_hmm`の巨大gap外れ値を拾った。
- terminal clusterのbest candidateは5/5 foldsで`exp226_k16`、通常clusterも4/5 foldsで
  `exp226_k16`だった。pooled labelの2-family判定は極小clusterとfold間label入替の影響を含む。

## 次

1. conditional Stage 1の30 CPU boosters、inference、submissionは実行せずbranchを閉じる。
2. exp264 global selectorは維持・fallback利用しない。再訪する場合も別実験の0-booster固定監査とし、
   raw-test availability監査、block-length proxy除外、outer-train clip、feature-family重みを事前固定する。

## 表記

用語は`backlog/KAGGLE_DIRECTION.md`の表記方針と`docs/glossary.md`に合わせ、実験名や設定名を除いて日本語優先で記録する。
