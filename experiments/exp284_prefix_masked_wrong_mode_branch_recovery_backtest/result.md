# exp284_prefix_masked_wrong_mode_branch_recovery_backtest 結果

## 状態

Kaggle private CPU version 2を完了した。technical guardは全PASS、scientific / safety guardはFAILし、
`close_without_parameter_rescue`としてbranchを閉じた。推論・提出は行わない。

## 固定仮説

known prefix末尾640行をmaskしてGR-supported wrong modeを注入し、safe baseを常時保持したうえで
self-GR top-3と未来256行evidenceを比較すれば、oracle failure triggerなしにwrong-mode recoveryを改善し、
no-injection時のfalse switchも抑えられる。

## 設定

- parent contract: `exp283_self_gr_topk_alternative_mode_branch_future_evidence_readout`
- route: `pf_beam`
- mask / observation / primary horizon: 640 / 128 / 256 rows
- branches: safe base、wrong active、real / shuffled self-GR top-3
- policies: wrong-only、safe+wrong、full、shuffled、no-injectionの5固定readout
- folds / active variant / booster: 5 / 1 / 0
- HMM / PF regeneration、parent/control再学習: 0 / 0、なし
- truth attachment: 全target-free tableのpersist・content SHA固定後のみ

## 実行

- kernel: `kentookumura/exp284-masked-wrong-mode-recovery-backtest-train`
- successful version / id_no: `2 / 127852894`
- runtime: `11,717.244秒`（約3時間15分17秒）
- eligible / ineligible: `766 / 7 wells`
- scored folds: `0, 1, 2, 3, 4`
- version 1: raw horizontalに存在しない`id`列を要求して評価前に停止
- version 2 fix: 監査専用IDを`<well>:<row_idx>`から決定的に生成。科学契約は変更なし

## 結果

全technical guardはPASSした。branch / evidence finite coverageはともに1.0、固定branch identity、mask
identity、5-fold coverage、minimum shift、eligible wellsを満たし、post-cut truth access before freezeは0だった。

pairwise selectorはpooled AUC `0.675153`だったが、fold AUCは
`0.690068 / 0.833659 / 0.754505 / 0.509459 / 0.555936`で2 foldsが0.60未満だった。
pooled choice accuracyも`0.590078 < 0.60`でFAILした。fold choice accuracyは
`0.610390 / 0.653595 / 0.584416 / 0.581699 / 0.519737`で全fold 0.50超だったが、pooled guardを
補えない。

H256 RMSEは次のとおり。

| policy | RMSE |
| --- | ---: |
| wrong active only | 37.557085 |
| safe base + wrong | 23.633930 |
| safe base + wrong + real self-GR top-3 | 26.072230 |
| safe base + wrong + shuffled self-GR top-3 | 25.520057 |
| no-injection safe base + real self-GR top-3 | 20.314398 |

full branchはwrong-onlyより`+11.484854 ft`改善し、5/5 folds改善した。一方、より安全な
`safe base + wrong` pairより`-2.438300 ft`悪く、改善foldは0/5だった。H512のwrong-only比gainも
`11.454901 ft`でH256 gainをわずかに下回り、persistence guardをFAILした。

real self-GR fullはshuffled controlより`+0.552173 ft`悪く、nonregressing foldは3/5に留まった。
no-injectionでbaseがunique bestのwellにおけるfalse switch率は`30.1724%`で、上限5%を大幅に超えた。
したがって総合guardはFAILである。

## 生成物と再現性

Kaggle summaryが記録したmetrics / manifest 8件を対象指定で取得し、SHAを全件照合した。ローカル確認先は
`/tmp/kaggle-output/exp284_prefix_masked_wrong_mode_branch_recovery_backtest/train_v2_metrics`。3,137,536行の
branch pathなど大容量target-free table本体は取得せず、Kaggle summaryのraw / content / decompressed SHAを
記録した。

- summary SHA: `3d9863ef...9e3e`
- overall / fold / pairwise metrics SHA: `d836b977...304` / `cbe7b823...9009` / `b025c18e...b623`
- by-well / mask / input manifest SHA: `8ebd8542...1a01` / `74026afa...fbf` / `71b039ac...61e4`
- target-free rows: branch paths 3,137,536、evidence 18,384、injection 9,958、policy 11,490、
  proposals 4,596
- executed config SHA: `0308717d...96b9`
- packaged notebook SHA: `13bfeca7...ae88`

## 解釈

safe baseを候補bankへ戻すことで、意図的なwrong-active-only状態から回復できること自体は確認できた。
しかしself-GR top-3の追加価値は否定された。safe+wrong pairへ追加すると全foldで悪化し、shuffled donorにも
負け、no-injection false switchも高い。exp283で見えたproposal-levelの弱いsignalは、安全なbranch選択や
incremental recoveryへ変換できない。

これはstandalone controlled backtestのnegative resultであり、exp283 PASSを意味しない。exp283も既に
scientific FAILでclosedのため、`triggered_fixed_horizon_self_gr_multibranch_hmm_recovery`、decoder接続、
current-test生成、inference、submissionへ進めない。

## 次

固定契約どおりK/window/horizon/veto/margin/thresholdのparameter rescueは行わずbranchを閉じる。exp285も
long-horizon prefix offset predictabilityを否定済みで、exp284/285から新しい救済backlogは追加しない。
