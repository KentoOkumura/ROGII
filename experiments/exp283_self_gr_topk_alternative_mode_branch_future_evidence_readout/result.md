# exp283_self_gr_topk_alternative_mode_branch_future_evidence_readout 結果

## 状態

Kaggle private CPU version 2を完了した。technical guardは全PASSしたがscientific guardはFAILし、
negative diagnosticとして不採用確定。CV/LB、推論、提出は対象外で実行していない。

## 固定仮説

self-GR過去matchをtop-3 alternative mode proposalに限定し、baseを保持したまま、proposalと行方向で
分離した未来256行typewell evidenceでbranchを検証すれば、exp282のhard donor-transferとは異なる形で
安全なmode回復候補にできるかを監査した。

## 実行

- kernel: `kentookumura/exp283-self-gr-topk-future-evidence-readout-train`
- id_no / version: `127849798` / 2
- runtime: 1,331.408秒（約22分11秒）
- runtime設定: private CPU、internet off、1 audit variant、0 config / 0 trained fold / 0 booster
- 処理量: 3,783,989 rows / 773 wells / 4,397 events / 13,191 proposals / 103,624 evidence rows

version 1はexp226 OOF foldとexp263 cache partitionの数値label一致を誤って要求し、118秒で評価前に停止した。
version 2は両partitionのwell内一意性と5-fold coverageを個別に検証し、exp226 foldをreadoutの正とする
technical fixだけを加え、科学条件を変更せず同じkernelで完走した。

## Guard結果

| Guard | 実測 | 判定 |
| --- | ---: | --- |
| technical checks | coverage / identity / freeze 全PASS | PASS |
| top-3 within10 lift vs shuffled | `0.755288 - 0.722083 = +0.033204` | PASS |
| positive proposal lift folds | 5/5 | PASS |
| branch-choice AUC | pooled `0.622168`; fold min `0.605266` | PASS |
| selected H256 RMSE gain | `8.221613 - 14.606586 = -6.384973 ft` | FAIL |
| nonregressing selected folds | 0/5 | FAIL |
| base unique-best false switch | `55.5647%`（上限5%） | FAIL |

fold別selected gainはfold 0--4で`-8.096628 / -6.502113 / -5.571780 / -5.966414 /
-5.893711 ft`。hidden-like spatial / typewell-purgedも`-7.174194 / -7.125766 ft`で悪化した。
768 event wells中、gain正108、0が23、負637。worst well `af7a59ce`は`-48.601538 ft`だった。

oracle-best RMSEは`6.547182`でbaseよりheadroomがある。一方、実際のevidence選択RMSE `14.606586`は
shuffled選択`14.369928`よりも悪い。pairwise AUCが0.60を超えても、候補間の累積尤度最大値をそのまま
branch選択へ使うとbaseを大量に誤って捨てるため、安全なutilityにはなっていない。

## 再現性・生成物監査

- truth attachment before freeze: 0
- event / proposal / evidence content SHA: `e4e5c159...bbed` / `2d1d38ac...c31d` /
  `61e261ad...455`
- summary SHA: `8de2db0b5d73f31fc66171893f3355db267376ff39392050ca880ac2bf82fe99`
- metric CSV 8件と5 gzip生成物のraw/decompressed SHAを`/tmp/exp283-v2-output`の実ファイルで照合済み

## 判断

proposal生成にはshuffled比の弱い再現signalがあるが、固定future-evidence verifierはbranch commitに
使えない。総合decisionは`close_without_rescue_grid_or_decoder_connection`とする。K/window/horizon/
veto/margin/threshold救済、decoder接続、inference、submissionへ進まない。

exp284は別の明示overrideでstandalone実行されたが、exp283からscientific promotionは付与しない。
したがって`triggered_fixed_horizon_self_gr_multibranch_hmm_recovery`の先行条件は満たさない。

## 次のアクション

exp283はここで完了とする。追加のself-GR branch選択実験は作らず、独立routeの既存backlogへ戻る。
