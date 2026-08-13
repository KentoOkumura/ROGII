# exp490_geometry_centered_mean_reverting_offset_hmm

## 状態

- ルート: `pf_beam`
- 状態: Stage 1 fail-closedを維持。hidden-dynamic inference v2提出完了
- CV: 8.48015525957654
- Public LB: 9.680（version 2、ref `55180208`）
- Private LB: なし
- Submit ID: `55180208`（version 1失敗refは`55163886`）
- 作成日: 2026-07-30
- 科学的親: `exp357_exp226_huber_emission_independent_audit`

## 仮説

exp357がGR尤度で誤ったresidual offsetを選んだ後、そのoffsetとrateを長く
維持することがtail誤差の主因である。exp226 geometryからの偏差に
K16区間1つ分のhalf-lifeを持つ平均回帰を加えれば、一時的な誤選択を
geometryへ戻しつつ、継続するGR証拠は尤度で維持できる。

## 変更点

exp357から変えるのは遷移中心だけである。

- `rho_t = 2 ^ (-dMD_t / K16区間MD span)`
- `rate中心 = 0.998 × rho_t × 前行rate`
- `offset中心 = rho_t × 前行offset + 当行rate × dMD_t`
- `予測TVT = exp226 tvt_geop + offset`

Huber `delta=1.345`、GR emission、sigma、state grid、process noise、initial prior、
posterior meanは固定する。hard resetやGR confidence gateは使わない。

## 検証方針

- Stage 0: 既存fixed32で1 candidate × 32 wellsの機構確認。CVではない。
- Stage 1: 2026-07-31のユーザー明示overrideにより、1 candidate × 773 wellsの5-fold OOF。
- Group: well
- Primary metric: unknown suffix pooled RMSE
- Safety: fold、1000+、hidden-like 2面、by-well p95 / worst、persistent episode
- Leakage check: predictionとcontractのSHA freezeまでtruth / error / role / episodeを読まない。
- Control: exp357 / exp281 / exp226の保存予測だけを使い、再実行しない。

Stage 1の最終条件は、exp357から`>=0.05 ft`改善し、かつexp226 final
`9.427109596582222`を`>=0.02 ft`上回ることを含む。

## 実行入口

Stage 0の正規sourceはJupytext percent形式の
`exp490_geometry_centered_mean_reverting_offset_hmm_compact_selfcontained_train.py`
で、ユーザーの実行承認後にcompact sourceから正規train notebookを生成した。
Kaggle kernelは
`kentookumura/exp490-geometry-mean-revert-offset-hmm-train` version 1
（id_no `129180511`）。private CPU、internet offで実行した。

full OOFは元のStage 0 notebookを上書きせず、stable SHA256 well modulo 4の
`train_variant0`--`train_variant3`でcandidateを生成し、SHA固定後に
`train_aggregate`でtruth-late strict mergeする。科学条件はStage 0と同一で、
合計773 HMM well-runs、保存control再実行0、GPU 0である。

4 shardとstrict mergeはいずれもKaggle private CPU version 1で完了した。
merge kernelは`kentookumura/exp490-mean-revert-full-merge`、id_no `129321382`。

明示LB監査overrideによるinferenceは
`kentookumura/exp490-geometry-mean-revert-offset-hmm-inference` version 1
（id_no `129323029`）で完了した。現行testは3 wells / 14,151 rowsで、exp226
geometryを再生成して同一HMMを3回decodeした。competition submitは別承認である。
その後の明示承認でversion 1を提出したが、hiddenデータ再実行で未処理例外となった。
2026-08-02に同じ正規inference notebookをversion 2候補へ修正した。runtimeの
sample / horizontal / typewell全件を重い処理前に読み、sample IDと全raw unknown行を
一致確認してから、実際のwell数だけexp226 geometryとexp490 HMMを実行する。
同じcanonical kernel version 2（id_no `129323029`）をprivate CPU / internet offで
実行しCOMPLETE。14,151 rows / 3 wellsのtechnical gate 14 / 14とsubmit-checkをPASSした。

## 結果

| メトリック | 値 |
| --- | --- |
| full OOF RMSE | **8.480155 ft** |
| exp357親RMSE | 9.737195 ft |
| exp357からの改善 | **1.257040 ft** |
| exp226 finalからの改善 | **0.946954 ft** |
| 改善fold | 4 / 5 |
| persistent episode SSE reduction | **41.4100%** |
| persistent episode count delta | **-59** |
| 改善well / 悪化well | 449 / 324 |
| by-well delta RMSE p95 | +7.257814 ft（FAIL） |
| worst-well delta RMSE | +49.602560 ft（FAIL） |
| Stage 1 gate | 12 / 14 PASS、fail-closed |
| Stage 0 gate | 18 / 20 PASS、fail-closed（fixed32） |
| inference v2 technical gate | 14 / 14 PASS |
| submission file check | FAIL 0 / WARN 0 |
| Public LB | 9.680（version 2、COMPLETE） |
| Private LB | なし |

## 所見

full OOFでもpooled RMSE、MD 1000+、hidden-like 2面、persistent SSE、episode数、
recoveryは改善したため、geometryへ戻す復元力そのものは有効である。一方、固定強度を
全wellへ適用すると一部wellの正しい長期offsetを消し、tailが大幅悪化した。
version 2のLBは9.680で、exp226 direct 9.837より0.157改善した一方、direct exact HMM
9.063より0.617悪い。Stage 1の安全性判定は変更せず、物理routeのLB anchorにも昇格しない。
復元力の適用条件を物理量で決める方向へ使う。

提出ファイル自体はsubmit-check PASSだった。hidden再実行の失敗原因は、推論sourceが
公開sampleのSHA、14,151 rows、3 wellsを固定assertしていたことにある。version 2では
これらをaudit-only参照値へ変更し、runtime sampleから行数・well数を導出する。
train / OOF、scientific contract SHA、物理モデルは変更していない。
公開testのsubmissionはversion 1とbyte-identicalで、SHAも`3970e9ad...be6e5`のまま。

## リスク / 注意

- 平均回帰が正しい長期offsetまで消す可能性があるため、matched-controlとtail gateを必須にする。
- fixed32のroleは過去truth由来なので、candidate生成後のreadoutにのみ使う。
- half-life、Huber、noise、gridを同一sampleで調整しない。
- Stage 0の固定all-AND gateは緩和していない。Stage 1は明示overrideとして分離し、
  inferenceと1回のcompetition submitは明示overrideで実施した。再提出は別承認なしで進まない。

## 次

exp490はfail-closeで完了。保存済みfull OOFを使う0-HMM readoutで改善wellと悪化wellを
比較し、segment span、GR情報量、geometry不確実性、初期offset、suffix horizonから
平均回帰の物理的な適用強度を定義する次実験へ進む。初回submitはhidden再実行ERROR、
修正版version 2はPublic LB 9.680まで完了した。固定strengthの救済調整や追加提出は行わない。

## 表記

用語は `backlog/KAGGLE_DIRECTION.md` の表記方針と `docs/glossary.md` に合わせ、実験名や設定名を除いて日本語優先で記録する。
