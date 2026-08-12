# 設計

## 1. アプローチ

exp504のpairwise rank出力をhard winner 1個へ潰さず、H512 blockごとに45列へ圧縮して
各blockのrowへbroadcastする。exp413の各downstream outer foldでは、outer-train側をinner
cross-fitしたrank特徴、outer-valid側をouter-train 4 foldsだけで学習したrank特徴にする。
その45列をexp413 final370へ追加し、同じTVT LightGBM 3 configsだけを再学習する。

```text
exp504 pairwise surface
  -> 45-column rank compact (strict outer/inner nested)
  -> exp413 final370 + rank compact45
  -> final415 TVT LightGBM
```

hard-selected TVTは特徴にも最終予測にも使わない。exp504のscientific FAILを再分類せず、
rank surfaceが後段モデルで有効かという別仮説だけを検証する。

## 2. 実験範囲

- 対象実験: `exp507_exp504_nested_rank_compact_addonly_on_exp413`
- Route: `ensemble`
- root parent / matched control: `exp413_scale5_likpf_full_replacement_on_exp335`
- rank source: `exp504_h512_regret_weighted_block_rank_selector`
- candidate contract: `exp293_physics_only_candidate_bank_headroom_contract`
- selector feature contract: corrected `exp264_exp263_candidate_confidence_dual_selector`
- 変更する変数: exp504 rank compact 45列のadd-only追加だけ
- 固定する変数: candidate、H512、rank loss/weight/model/guard、outer fold、exp413 370列、
  TVT target/metric/config/early stopping/postprocess/scope

## 3. 固定入力

### exp504 rank source

- Kaggle kernel: `kentookumura/exp504-h512-regret-block-rank-selector-train`
- version / id: `1 / 129488458`
- rows / wells / blocks / candidates: `3,783,989 / 773 / 7,787 / 12`
- candidate content SHA256:
  `29477141685662bae7417e788ec5dbe914c2220b7f4cc45ab01befa4e5e3b474`
- row feature schema SHA256:
  `aaef4ffdd90667893b099b76a52f1957b22197aea9cee5e5b57bc81048ddd3a4`
- block feature schema / content SHA256:
  `523ef0b85cf9873abe463a7ec7f99b703eb0fde6062bcc85e7f5f40508f060d4` /
  `f333d097d8bdd369b2b6786328dee050d6bb5ba4114d810e26e60be976fd56c8`
- model manifest logical SHA256:
  `60696a0574de0c62f8c413c2344a664f40a40634f202ec3a02a754bd2ef3de25`
- OOF prediction content SHA256:
  `1dd09844b70536ec7eae26d6656efb70a00bdc3488a57aca188fb6dfc3b2504f`
- outer-valid source files:
  `*_block_selection.parquet`、`*_pair_probability_oof.parquet`、
  `*_oof_predictions.parquet`

上記3 fileのfile / logical content SHAは、実装時Stage N preflightでKaggle version 1から取得して
設計済みmanifestへ追記し、不一致なら学習前に停止する。OOF内のtruth readout列は読み込まず、
join key、fold、block、candidate value、rank出力のallowlistだけを使う。

2026-08-03の実装preflightで固定した主な実ファイルSHA:

| file | file SHA256 | logical / allowlist SHA256 |
| --- | --- | --- |
| `block_metadata.parquet` | `2417ef9661f61cd3892e02853e92a81408372bbbff1ec6a96de8691630caa181` | `dc50b0d65b347675a1485466379637443bbd2a0db255f87640ce0a02f76cc735` |
| `block_selection.parquet` | `7a53818a1d96fa2601eb2dc4043633ea9a65d22baa47551df207715f9bd38dea` | `0cd97d79b2925e2ad9ed0b7fc5ec70f6fbb74d8156a056839130435f8b7b4f8b` |
| `pair_probability_oof.parquet` | `48b1466f278c63f7ffed069e7994348067ff708214cc00d491194a50b3aaf78a` | `fb1697339f41db0de9c0f67c67edb92c74381bd611924314d25ca891f1678272` |
| `oof_predictions.parquet` | `6dfdccfa0baf0a21a4e4fb9fb8cd026063c595ff5857f83a885c381261019181` | allowlist 6列 `30870f5c137ebe77eaf0b7683c1f9c5aee3ca8a8af07ac9d56668e7c57581a3a` |

candidate bankとblock feature 3配列、schema、pair contract、model manifest、target-free
freezeのfile SHAも`config.yaml`へ固定した。preflightはOOFから`id / well / well_row_idx /
outer_fold / md_since / h512_group`だけを読み、selected / truth列を読まない。

### candidate順とanchor

1. `exp226_k16`
2. `selfgr_hmm_a070`
3. `likpf_mean`
4. `exact_hmm`
5. `pf_ancc`
6. `beam_mean`
7. `exp226_k16__selfgr_hmm_a070`
8. `exp226_k16__exact_hmm`
9. `exp226_k16__likpf_mean`
10. `selfgr_hmm_a070__likpf_mean`
11. `likpf_mean__exact_hmm`
12. `exp226_w500_50_50`（anchor）

### exp413 downstream control

- source kernel: `kentookumura/exp413-scale5-likpf-downstream-train` version 2
- saved CV: `7.884802794404715`
- OOF SHA256: `9bd2d17778b3b27d771b12cbff72def8b87e6cdf14062e1c0ba192434cef4a9d`
- model manifest SHA256:
  `4b4f988154468ba6697cdd57c0a0c6bf7cc631e7b2bbe1f15fa8f51fdeb7c3df`
- feature formula: `clean273 + compact74 + signed23 = final370`
- fixed config: 3 LightGBM configs × outer 5 folds

exp413 controlは保存OOFだけで比較し、15 GPU boostersを再学習しない。

## 4. 45列compact schema

すべてfloat32、以下の順序に固定する。Borda / anchor / summary / provisionalの42列を
block内全rowへbroadcastし、weighted TVT mean/std 2列と相対位置1列の計3列はrowごとに変える。
初回設計の`block-constant 44 / row-varying 1`は4.6のrow単位数式と矛盾していたため、
数式を正として`42 / 3`へ機械的に訂正した。feature名・総数45・科学変数は変えていない。

### 4.1 Borda score: 12列

反対称化済み勝率`p(j beats k)`から、exp504と同じ
`b_j = sum_{k != j} p(j beats k) / 11`を使う。列順はcandidate固定順。

`rank_borda__<candidate_id>` × 12。

anchor scoreは`rank_borda__exp226_w500_50_50`そのものであり、同値の別列は作らない。

### 4.2 candidate対anchor勝率: 11列

anchor以外のcandidate固定順で`rank_p_vs_anchor__<candidate_id>`を置く。canonical pairの
左右にかかわらず、値は必ず`p(candidate beats anchor)`へ向きを揃える。全66 pairの残り55列は
downstream schemaへ入れない。

### 4.3 Borda分布要約: 5列

1. `rank_borda_top1_score`: 最大Borda
2. `rank_borda_top2_score`: deterministic順位2位のBorda
3. `rank_borda_margin`: top1 - top2
4. `rank_borda_score_std`: 12 scoreのpopulation std (`ddof=0`)
5. `rank_borda_entropy`: `-sum_j(w_j log w_j) / log(12)`

`w_j = b_j / sum_k(b_k)`、`0 log 0 = 0`とする。反対称化が正しければ
`sum b = 6`であり、tolerance `1e-6`を外れたblockはfail closedする。

### 4.4 anchor rank: 1列

`rank_anchor_rank`を1始まりの`1..12`で置く。score降順、差`<=1e-12`のtieはanchor優先、
残りはcandidate固定順とする。anchor scoreは4.1に含まれるため重複させない。

### 4.5 provisionalとfallback: 13列

- `rank_provisional_is__<candidate_id>` × 12: guard前Borda winnerのone-hot
- `rank_anchor_guard_fallback`: provisional非anchorかつ
  `p(provisional beats anchor) <= 0.5`なら1、それ以外0

provisionalのordinal index、文字列ID、guard後selected IDは入れない。guard後selectedは
provisional one-hotとfallbackから導けるが、selected TVTは入れない。

### 4.6 Borda-weighted TVT moment: 2列

同じblock・rowの12 candidate TVT値`v_j`を使い、次をrow単位で計算する。

```text
rank_borda_tvt_mean = sum_j(w_j * v_j)
rank_borda_tvt_std  = sqrt(sum_j(w_j * (v_j - mean)^2))
```

candidate TVTはtarget-free固定bankの値でありtruthではない。softmax、temperature、unbiased補正を
使わない。weighted meanを直接最終予測として採用せず、TVTモデルの2特徴としてだけ渡す。

### 4.7 H512 block内相対位置: 1列

```text
rank_h512_relative_position =
    (well_row_idx - block_start_well_row_idx) / max(row_count - 1, 1)
```

1-row blockは0、通常blockは先頭0・末尾1とする。well ID、absolute block ID、outer foldは
特徴へ入れない。

### 4.8 count

`12 + 11 + 5 + 1 + 13 + 2 + 1 = 45`。最終TVT schemaは
`clean273 + compact74 + signed23 + rank_compact45 = final415`。

## 5. nested split

exp413のStage Cは既にnestedである。ただしexp504の保存OOFは、各rowを「そのrowのouter foldを
除く4 folds」で学習したstandard 5-fold OOFであり、exp413 final modelのouter-train用featureを
構成する内側cross-fitにはなっていない。

downstream outer foldを`F`、そのouter-train内のheld inner foldを`G`とする。

- downstream outer-valid `F`: 保存exp504 outer model `M[-F]`のOOF rank出力を再利用する。
- downstream outer-trainのpartition `G`: `F`と`G`を除く3 foldsだけでrank model
  `M[-F,-G]`を学習し、held `G`へ予測する。

これを`F=0..4`、`G != F`で行うため新規inner rank modelは20本となる。各outer foldには
4 inner-train partitions + 1 outer-valid partitionがあり、合計25 partitions、
`3,783,989 × 5 = 18,919,945` row-roleとなる。

rank label / regret weightは各`M[-F,-G]`の3 training foldsのtruthだけから作り、weight平均1への
normalizationもその3 folds内だけで行う。held outer / held inner truthはfit前に読まない。

## 6. phase separation

### Stage N: nested rank compact（将来、別承認）

1. exp504 input contract、candidate、block、88列surfaceをSHA検証する。
2. target-free surfaceとblock mappingを固定し、forbidden columnsを監査する。
3. 20 inner modelsを3-fold train / 1 inner-fold predictで学習する。
4. inner予測では66 pairを一時的に計算するが、45列へ集約後にraw 66をdownstream artifactへ保存しない。
5. outer-validは保存exp504 version 1のBorda / pair outputから同じ45列を再構成する。
6. 25 compact partitionsをkey/fold/role付きで保存し、schema/content/model SHAを固定する。

### Stage D: downstream add-only（Stage N PASS後、別承認）

1. exp413のclean273、compact74、signed23の保存partitionをSHA検証する。
2. 同じ`well / row_idx / downstream_outer_fold / role`でrank compact45をstrict joinする。
3. final415を作り、重複列、NaN/Inf、missing/extra rowを0件と確認する。
4. treatment 1 × config 3 × outer 5の15 GPU boostersを学習する。
5. OOFをfreezeしてからtruthを評価し、保存exp413 OOFとmatched比較する。

Stage N technical FAILならStage Dへ進まない。Stage D scientific FAILならsame-OOF rescueをせず閉じる。

## 7. 評価と判定

Primary readout:

- exp413 / final415 pooled OOF RMSEとgain
- fold 0--4 RMSE / delta
- md_since 0--250、250--1000、1000+ RMSE / delta
- hidden-like spatial / typewell-purged RMSE / delta
- by-well delta median / p90 / p95 / p99 / worst、+1/+3/+5 ft悪化well数
- rank compact45のconfig/fold別gain importance、used feature count

Mechanism readout:

- 45列のfinite率、mean/std/min/max
- Borda sum / entropy / margin / anchor rank / fallback分布
- provisional candidate one-hot usage
- Borda-weighted TVT mean/stdとexp413 residualのfold別相関

promotionはrequirementsのTVT RMSE / fold / scope / technical all-ANDだけで決める。feature importanceや
rank accuracyだけで昇格させない。by-well tailは必須報告だが、exp413の既存評価契約に合わせて
promotionの数値gateにはしない。

## 8. 将来の実行量契約

| Stage | variant | config / objective | folds | 新規booster | runtime |
| --- | ---: | ---: | ---: | ---: | --- |
| N nested rank compact | 1 | rank config 1 | outer 5 × inner 4 | 20 | Kaggle CPU |
| D TVT add-only | 1 | LightGBM 3 | outer 5 | 15 | Kaggle GPU |
| 合計 | 1 | - | - | 35 | CPU + GPU |

control再学習、exp504 outer model再学習、candidate/PF/HMM/Beam再生成はすべて0。
Stage NとStage Dは個別承認とし、同一pushで自動連鎖させない。

## 9. 再現性設計

- seed: 42。exp504 rank configとexp413 GPU configを変更しない。
- stochastic処理: LightGBM学習のみ。候補・block・outer-valid rankは保存artifactを使う。
- CPU rank: deterministic / force_col_wise、固定threads、stable key / candidate / pair順。
- GPU TVT: exp413のdeterministic、force_col_wise、gpu_use_dp、固定seed/threads契約を継承し、
  bitwise deterministicとは断言しない。
- global RNGをthread内で使わない。fold/partition処理順を固定する。
- gzipはraw SHAとdecompressed content SHAを分け、後者を主証拠にする。
- Stage N: input、block、rank schema/content、45列schema、25 partition、20 model、manifest SHA。
- Stage D: final415 schema/content、15 model、OOF、fold/scope/by-well、manifest SHA。
- deterministic anchor: 独立rerunでfeature/model/prediction SHAが一致するまでfalse。
- Kaggle package: metadataとbootstrap ZIP内config、run flag、input sourceをpush前に照合する。
- inference/submissionへ進む場合だけcurrent-test feature/prediction/submission SHAを追加する。

## 10. リスクと停止条件

- leakage: exp504 standard OOFをouter-train全行へそのまま使うことが最大リスク。25 partition契約を必須にする。
- distribution shift: inner modelは3 folds、outer-valid modelは4 foldsで学習量が異なる。exp413/exp264と同じnested構造として受容し、fold readoutで監査する。
- tail: exp504 hard selectorはhidden-likeとby-well tailで悪化した。hard choiceを使わず、scope/tailを必ず報告する。
- collinearity: 45列には派生量がある。事前subsetは行わず、LightGBM重要度をmechanism readoutに限定する。
- runtime/memory: H512 pair featureは最大1,986列。outer-validは保存artifactを再利用し、innerは1 modelずつfitして一時memmapを削除する。
- artifact availability: exp504の全pair/block outputがKaggle sourceで取得不能、またはSHAを固定できない場合は実装前preflightで停止する。

禁止事項:

- raw 66 pairのdownstream追加、pair subset/grid、anchor以外の11列選択変更
- H128/H256/whole-well/overlap、candidate追加・削除・式変更
- rank loss/weight/model/threshold/guard/tie ruleの変更
- provisional ordinal ID、selected ID、hard-selected TVT、truth/error/oracle/well ID特徴
- Borda softmax、temperature、blend weight、weighted TVTの直接予測採用
- exp413 compact74/signed23のreplacement、final370列の削除
- control再学習、same-OOF feature subset、well/row gate、post-hoc calibration
- Stage N未通過でのStage D、Stage D未通過でのinference/submission

FAIL時は`FAIL_CLOSE_WITHOUT_PAIR_FEATURE_SUBSET_TEMPERATURE_OR_GATE_RESCUE`で閉じる。
