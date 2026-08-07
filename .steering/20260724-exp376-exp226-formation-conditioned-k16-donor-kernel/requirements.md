# 要件

## 依頼

`exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction` の
K=16 donor slope 補間を、周辺 train well にだけ存在する
`ANCC`、`ASTNU`、`ASTNL`、`EGFDU`、`EGFDL`、`BUDA` で条件付けする。

train well の正解 `TVT` から作る K=16 slope は exp226 と同じまま使い、
対象 well の地層列を直接読むのではなく、outer-train wells だけから推定した
地層相対座標で既存 XY donor kernel の重みだけを緩やかに変更する。

この段階では設計を確定し、backlog、steering、実験ディレクトリを作成する。
実装、正規 notebook の作り替え、Kaggle package/push/run、推論、提出は行わない。

## 2026-07-24 追加依頼

ユーザーの「exp376を実装してください」を実装承認として受け、
compact self-contained Jupytext候補、target-free Stage 0、Stage 1 direct、
Stage 2 fixed12 add-one noveltyまでを実装する。

初期設計どおり、既存の正規`*_train.ipynb` scaffoldは上書きしない。
正規notebook採用、Kaggle package/push/run、current-test生成、推論、提出は
今回の実装承認に含めず、別の明示承認を必要とする。

## 2026-07-24 実行依頼

ユーザーの「実行してください」を、compact self-contained trainの正規notebook採用と
Kaggle CPU run 1回の承認として受ける。1 variant / 0 model config /
5 reporting folds / 0 trained fold / 0 booster、親control再実行0を維持する。
current-test生成、推論、提出は引き続き別承認を必要とする。

## 2026-07-24 v1実行結果

Kaggle CPU v1は5 foldsの予測を完了後、truth前freezeでlist-valued
reference manifest cellをpandas logical hashへ直接渡したため
`TypeError: unhashable type: 'list'`となった。Stage 0/1/2、CV、truth scoringは
未評価であり、科学的なPASS/FAILとは扱わない。

container-valued object cellだけをcanonical JSON化する局所修正と再現testまでは
実装範囲に含める。修正版Kaggle CPU v2は新しい実行として別の明示承認を必要とする。

## 2026-07-24 v2実行依頼

ユーザーの再度の「実行してください」を修正版Kaggle CPU v2の明示承認として受ける。
同じcanonical kernelへversion 2としてpushし、1 variant / 0 model config /
5 reporting folds / 0 trained fold / 0 booster、親control再実行0を維持する。
current-test生成、推論、提出は承認範囲に含めない。

## 2026-07-24 v2実行結果

同じcanonical kernel version 2は`COMPLETE`。Technicalとtarget-free Stage 0は
PASSしたが、direct RMSEはexp226比`+0.016147593 ft`、改善1/5 folds、
by-well p95`+0.376679336 ft`、worst`+1.891559930 ft`でFAILした。
fixed12 add-oneもH512 / whole-well改善`0.019403532 / 0.015542019 ft`で
閾値`0.05 ft`未達となりFAILした。

固定decisionに従い、救済grid、current-test生成、selector組み込み、推論、提出、
version 3を行わずbranchを閉じる。

## 制約

- Route: `pf_beam`
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- 親 exp226 の K=16 分割、正解TVT由来 raw/smoothed slope、XY近傍50、bandwidth 500 ft、
  ridge、adaptive kappa 12項、near-strike gate、ANCC local theta、GR correction、
  U-projectionを固定する。
- 変更は同じXY近傍50に対する地層相対座標のsoft weight 1種類だけとする。
- `exp226` の保存済み OOF（CV `9.427109596582213`、
  decompressed SHA256 `709eb726cc30da523f017ed0dbd0371967b88a91ddcf25578eb9356f28e4c609`）
  を control とし、再生成しない。
- outer-valid fold の正解 `TVT` と6地層列は、地層面fit、donor signature、
  donor field、adaptive kappa、weight設定のいずれにも使わない。
- outer-train donor 自身の地層signatureも、そのwellを除いた参照から推定し、
  validation/test queryと同じ availability 条件に揃える。
- current test では全train wellsの正解TVTと6地層列を参照に使えるが、
  test well に存在しない地層列を入力として要求しない。
- 地層weightの係数、clip、近傍数、surface imputer設定を同一OOFで探索しない。
- direct prediction blend、HMM/PF再decode、selector、downstream ML、GPU学習を含めない。

## 受け入れ基準

- steering 3文書、実験scaffold、`config.yaml`、`SESSION_NOTES.md`、`README.md`、
  `result.md`、`metrics.json` に同じ設計契約が記録されている。
- `KAGGLE_DIRECTION.md` の未着手backlogに、既存P1を追い越さない
  「中・P2・CPU・design-only」として追加されている。
- `experiment_summary.md` に `exp376` が `pf_beam` route、
  `design_frozen_not_implemented` として記録されている。
- 予定実行量が 1 variant / 0 model config / 5 folds / 0 booster /
  parent control再実行0 と明記されている。
- 地層signature、robust標準化、soft weight式、fallback、support guard、
  direct評価、H512 add-one novelty評価が事前固定されている。
- 実装前の状態では notebook scaffold を実行可能な正規実装として扱わず、
  Kaggle push/run関連フラグが無効である。
- deterministic anchor として扱う場合は、feature content SHA、model SHA、prediction SHA、submission SHA、Kaggle kernel version が記録されている。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
