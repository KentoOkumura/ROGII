# 設計

## 仮説

同じ`native_overlap_1` Type Well群では、個々のwellで不安定なaffine係数そのものより、bias、MAD residual scale、fit RMSEに再現可能な群共通性があり、outer-train peerだけからheld-out suffixへ転送できる。

## アプローチ

各 outer fold で outer-train well だけを使い、Type Well GRをTVT軸へ重複median・線形補間（外挿なし）して horizontal GR と対応づける。wellごとにHuber IRLS（delta 1.345、最大50反復）を推定し、support `k=200` でidentityへ縮約した `a,b`、GR=50でのbias、残差MAD scale、fit RMSEを得る。group priorはwell等重みmedianとし、対象wellを除いた同一群統計をheld-out wellへ割り当てる。outer-valid suffix truthは全real/control priorのcontent SHAを凍結した後だけ結合し、identity Type Well GRに対するhorizontal GR再構成RMSE、prefix fit、suffix oracle統計との転送誤差を測る。

TVT候補やdecoderを生成しない制約上、suffix RMSEの単位はhorizontal GR API unitである。旧scaffoldの`*_ft` gate名はこの実装で`*_gr_*`へ修正し、閾値の数値は事前登録どおり維持する。

## 実験範囲

- 対象: `exp311_typewell_group_prefix_suffix_gr_calibration_readout`
- Route: `pf_beam`
- 親: `exp211_affine_calibrated_gr_observation_pfbeam`
- 変更する変数: group-level affine/noise priorを利用するか否か。
- 固定する変数: raw rows、fold、群定義、補間、Huber estimator、negative control、gate。
- 計算量: 1 diagnostic variant × 5 folds、model/booster/decoder 0。
- Negative control: outer-train group labelのSHA256 deterministic rotationと、matched GR pairのSHA256 deterministic circular shift。
- Stress: same-group held-out、leave-one-Type-Well-group-out identity fallback、exp115 spatial/typewell-purged roleの3面。

## 再現性設計

- fold/variant/wellのimmutable keyからSHA256順序を作り、global RNGは使わない。
- typewell content SHA、fold manifest、pair table schema/content SHA、group table SHA、score table SHAを保存する。
- gzipはdecompressed content SHAを主証拠にする。
- Kaggle実行時はCPU、internet disabled、kernel versionとbootstrap内configを記録する。
- prediction/model/submissionは生成しないためそのSHAは対象外。

## リスクと停止条件

- group共通性は slope より noise/reliability に強い可能性があるため、hard affine correctionを禁止する。
- group LOO R²、suffix gain、4/5 folds、negative-control差、worst-well guardのいずれかを満たさなければ後続を止める。
- support不足はglobal priorではなくidentity/no-correctionへ落とし、都合のよいgroup再定義は同一実験で行わない。

## 次のアクション

compact notebookをcanonicalへ採用し、Kaggle private CPU version 1を完了した。fit-RMSE R²とworst-well guardがFAILしたため停止条件を適用し、exp312〜320は開始しない。同一OOFでgroup定義、shrinkage、閾値を救済調整しない。
