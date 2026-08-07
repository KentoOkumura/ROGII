# exp402_fold_safe_grwr_5_addonly_on_exp287 結果

## 状態

Kaggle private CPUのtrain source、current-test、outer-fold 0–4の
upstream 7 runはすべてPASSした。aggregate version 2も
`18 / 18` checks、status `zero_booster_preflight_passed`で完了し、
Stage 0 technical gateはPASSした。Stage 1 GPU train version 4は実行中。
CV、推論、提出は未完了。

## 仮説

exp287のfold-safe dense-formation 3候補を含む計8候補のspreadと、
既存のtarget-freeなGR/DWT/FFT成分との固定interaction 5列は、
個別の候補値・GR成分を別々に持つ421特徴では不足する局所的不確実性を補える。

## 設定

- 親: `exp287_fold_safe_formation_74_addonly_on_exp264`
- clean tail control: `exp264_exp263_candidate_confidence_dual_selector`
- 変更: fold-safeに再計算するGRWR 5列だけ
- 最終特徴数: 426
- 検証: exp287と同じ5-fold GroupKFold、group=`well`
- metric: score rowsの非加重RMSE
- seed: 42
- 条件付き実行量: 1 variant × 3 configs × 5 folds = 15 GPU boosters
- control再学習: 0

## 結果

| メトリック | 値 |
| --- | --- |
| CV | 未実行 |
| Public LB | 未提出 |
| Private LB | 未提出 |
| Stage 0 upstream | 7 / 7 PASS |
| Stage 0 aggregate | version 2 COMPLETE、18 / 18 PASS |
| outer-role partition | 10 / 10 PASS |
| current-test | 14,151 rows / 3 wells |
| model / booster / prediction / submission | 0 / 0 / 0 / 0 |
| Stage 1 train | version 4 RUNNING、15 T4 boosters予定、control 0 |

## 再現性

- deterministic anchor:
  `8b2befb44bc22b6e62675edda48c90c0593932b815c00fab0915e817c24e6635`
- GRWR-5 / formation feature generation: 新規RNGなし、float32固定式
- current-test PF: exp072 source SHAとper-well stable seedを固定
- input / feature content SHA: upstream 7 runとaggregate manifestで照合済み
- model / prediction / submission SHA: 非該当
- Kaggle aggregate kernel: version 2 / id_no `128831850`
- preflight runtime / peak RSS: 8.194秒 / 0.241 GiB
- partition manifest SHA:
  `704d8a9163f5a82c9b28f3866e2de6d3a7dfac78a4236b5352436431674f365b`
- preflight manifest SHA:
  `c8af15ad8502b172031eaa862878ba07f2a94b6eba5913259c3a4ba0e5142de8`
- reproducibility manifest SHA:
  `5456cfd9b0d2df3cac5848cb234cf382f9ebd1515439742fcff6afa3f0560fda`
- config SHA:
  `82bcce7c7d6e0694ffc67a2898068213e68d1b52cb560f89f63f8788f701bce0`
- compact implementation source SHA:
  `7098ebe2063faeaee2d0d9b65d910648777da7726f682679ce1f17a8548c4ac4`

## 実装検証

- 別名compact self-contained train / fail-closed inference候補を作成
- exp218 synthetic formula parity、fold role leakage guard、float32固定式を検証
- pycompile、Ruff、Jupytext round-trip、strict `validate-exp`: PASS
- 専用test: `13 passed`
- 正規train notebook: Stage 1候補を採用、SHA
  `310e3fd356d8f444761d571a78533c9a83ed2c87304798c281ca147e7874ef55`
- 正規inference notebook: placeholderのまま

## 解釈

旧GRWR 6列を一括復旧せず、formation依存5列だけをexp287からの独立した
add-only仮説として切り出した。exp396 entropy依存の1列は対象外であり、
exp396の閉鎖判断を変更しない。

monolithic version 1のruntime未完了後、科学仕様を変えず7 upstream shardへ
分割したことで全入力生成は完了した。最初のaggregate失敗は生成物ではなく、
fold 4だけ実slugが`-v2`になった際のresolver contract不足である。
version 2はdisk上のconfig/source SHAを維持し、wrapper内のin-memory pathだけを
補った。全foldのmanifest/file SHA、role coverage、formation境界、実行量契約が
一致したため、修正は科学仕様を変えていない。

## 次

Stage 1 version 2はaggregate inputのmount pathを固定absolute pathだけで
解決したため、10.6秒、0 boosterで失敗した。version 3はSHA-qualified mount探索と
物理T4 guardを通過したが、clean-273再構築に必要なexp145 input不足で227.3秒、
0 boosterのまま失敗した。version 4は`exp145-train`を11番目のinputに追加し、
必要3ファイルを前処理前にfail-fast検証する。同じ1 variant × 3 configs × 5 folds =
15 T4 boosters、control再学習0で実行中。完了後に保存済みexp287/exp264 OOFとの
固定promotion gateを確認する。inferenceとsubmissionは未承認なので実行しない。
