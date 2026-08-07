# exp253 prefix-verified bounded candidate controller 結果

## 結論

固定したprefix candidate evaluationとbounded correctionを全773 wellsへ適用した結果、技術契約は通過したが、
overall RMSEは`7.936701 -> 8.205455`（`+0.268755`）へ悪化したため不採用とする。
worst-wellを拒否条件から外したユーザー指定の判定でも、overall、1000+、hidden-like 2面、fold stabilityが不通過だった。
`adoption_supported=false`、`inference_allowed=false`とし、inference / submissionへ進めない。

## 仮説と設定

- 親: `exp238_nested_hmm_exp226_selector_rank_slot_addonly_on_exp218`
- candidate親: `exp072_exp063_full_replay_feature_cache`
- Route: `ensemble`
- 検証: 50/65/75% masked-prefix pseudo-holdout後のofficial-tail readout
- candidate: exp072既存9 familyのみ
- correction: alpha最大0.40、move最大30 ft
- 学習コスト: 1 scientific variant / 4 CPU shards / 0 config / 0 fold / 0 booster / parent再学習なし

## Stage 1結果

4 shardは773 wells / 2,319 requestsをerror 0で完走し、row-level aggregateは3,783,989 rowsを結合した。
scored well 100%、全well 3 cuts、各request 9 candidates、nonfinite 0、alpha/move cap内で、technical checksは9/9通過した。

| surface | base RMSE | controller RMSE | delta |
| --- | ---: | ---: | ---: |
| overall | 7.936701 | 8.205455 | +0.268755 |
| 000_050 | 0.821725 | 0.818979 | -0.002746 |
| 050_100 | 1.158488 | 1.152843 | -0.005645 |
| 100_250 | 1.890906 | 1.878531 | -0.012375 |
| 250_500 | 3.016693 | 2.992514 | -0.024179 |
| 500_1000 | 4.528797 | 4.532969 | +0.004172 |
| 1000_plus | 8.703461 | 9.011444 | +0.307983 |
| hidden-like spatial | 8.622839 | 8.905712 | +0.282873 |
| hidden-like typewell-purged | 8.599611 | 8.867154 | +0.267543 |

fold 0-4はすべて悪化し、deltaは`+0.233741 / +0.195464 / +0.498533 / +0.074539 / +0.303746`だった。
361 wellsへ補正が適用され、150改善 / 211悪化、412 wellsはbase維持だった。worstは`fcfcc902`の+10.310641。

## Stage 0との差

Stage 0のsorted 32 wellsではoverallが`-0.198601`、1000+が`-0.226789`、foldは4/5改善していた。
しかし全wellでは補正対象が9から361 wellsへ増え、near 250 ftまでは小改善を保った一方、500 ft以遠で符号が反転した。
したがってStage 0の小標本改善は全wellのlong-tailへ転移せず、公開notebook由来のprefix scoreはこの候補集合・baseに対する採用根拠にならない。

## 採用ガード

| guard | 必須 | 結果 |
| --- | --- | --- |
| overall改善 | はい | FAIL |
| 000-050非悪化 | はい | PASS |
| 1000+非悪化 | はい | FAIL |
| hidden-like spatial非悪化 | はい | FAIL |
| hidden-like typewell-purged非悪化 | はい | FAIL |
| 3/5 folds改善 | はい | FAIL（0/5） |
| worst-well +0.25以内 | いいえ・monitor-only | FAIL（+10.310641） |

## 再現性

- aggregate kernel: `kentookumura/exp253-prefix-bounded-controller-aggregate` version 1 / id_no `127430343`
- aggregate runtime: 73.396秒、CPU、internet off
- deterministic anchor: いいえ（stochastic PF replayの独立rerun SHA未取得）
- base OOF decompressed SHA: `0e7390ac3b3a432b1d432e432cb374cbf38da393a9b95f8f0d6c22732030010c`
- controller OOF decompressed SHA: `a66296559152eed8b3b9a753c0965ad5db2f693a28576d51b05861487dd03b22`
- candidate cache decompressed SHA: `99a3c70a19b2f22a8c76c5947b14692e7b8207ea45f38b8b1c5f327d320e1350`

## 後続

この固定手法はbranchを終了する。性能悪化後のparameter grid、gate/alpha/clip緩和、inference、submissionは行わない。
新規backlogは追加せず、再訪する場合はprefix pseudo-holdout scoreがlong-tail official-tailへ転移しない原因を独立仮説として扱う。
