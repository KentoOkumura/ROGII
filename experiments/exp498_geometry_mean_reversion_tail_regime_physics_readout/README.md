# exp498_geometry_mean_reversion_tail_regime_physics_readout

## 状態

- ルート: `pf_beam`
- 状態: Kaggle version 2完了、technical PASS / physics-regime FAIL、terminal close
- CV: なし（diagnosticでありcandidate評価ではない）
- Public LB: なし
- Private LB: なし
- Submit ID: なし
- 作成日: 2026-08-01
- 親実験: `exp490_geometry_centered_mean_reverting_offset_hmm`

## 仮説

exp490のgeometry平均回帰はpersistent誤差を改善したが、GR拘束が弱く、geometry面と
GR-corrected exp226面が競合し、suffix開始直後から非ゼロoffsetの証拠があるwellでは、
正しい長期offsetまでgeometryへ戻してtailを悪化させる。

## 変更点

予測モデルは変更しない。exp490 merge v1の保存full OOFとSHA固定raw visible prefixから
物理量をwell集約し、truth-lateで単一regime
`weak_gr_geometry_conflict`のtail悪化集中だけを監査する。

- suffix horizon、K16 segment span
- known-prefix GR sigma / information ratio
- geometry disagreement proxy
- early offset evidence
- HMM state uncertainty（secondaryのみ）

## 検証方針

- Phase A: truth / error / foldを読まず、固定bucketとprimary regimeをfreezeする。
- Phase B: 保存fold / by-well / episode outcomeをjoinする。
- primary: regime対complementのharm rate、delta RMSE、catastrophic tail capture。
- consistency: 4 / 5 folds。
- gate: steeringに固定した6項目all-AND。
- leakage check: freeze前のtruth / error / outcome readを0にする。

## 実行量

実績はreadout 1、773 well aggregation、5 fold readoutだけである。
新規HMM、prediction、model、booster、PF、Beam、GPUはすべて0。

## 実行入口

正規train notebookはcompact self-contained実装で、入力SHA、target-free feature freeze、
truth-late readout、固定all-AND、生成物保存をセル上で追える。inference notebookは範囲外の
placeholderを維持する。Kaggle train readoutは完了し、inference、submissionは無効である。

## 結果

Kaggle private CPU version 2（id_no `129328553`）で3,783,989 rows / 773 wellsを
76.685秒、peak RSS 0.361 GiBで完了した。固定入力SHA、truth-late、feature/bucket、
0 prediction/HMM/model/PF/Beam/GPUを含むtechnical checksは全PASSした。

一方、primary regimeは0 wellsだった。weak observationは359 wellsだが、geometry
disagreement `>=10 ft`は0 wells（最大`5.337991 ft`）、early abs offset `>=5 ft`は
1 wellだけで同時成立は0。coverage、harm rate、mean delta、fold consistency、
catastrophic captureの6 checksは全FAILし、decisionは
`terminate_mean_reversion_tail_regime_cause_tracking`である。

## 所見

事前仮説のgeometry conflict閾値に到達するwellがなく、仮説は支持されなかった。
結果を見た閾値緩和やsecondary bucketからの救済はせず、exp490のterminal closeを維持する。

## リスク / 注意

- `abs(exp226_pred-tvt_geop)`はgeometry真値の不確実性ではなく不一致proxyである。
- bucketやprimary閾値をoutcomeを見て変更しない。
- coverage不足はbucket統合で救済せずFAILとする。
- diagnostic結果を直接selector、inference、submissionへ使わない。

## 次

本branchは完了。復元力を弱める後続式は作らず、mean-reversion tail regime原因追跡を終了する。

## 表記

用語は`backlog/KAGGLE_DIRECTION.md`の表記方針と`docs/glossary.md`に合わせる。
