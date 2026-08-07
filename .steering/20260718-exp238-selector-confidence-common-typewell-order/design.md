# 設計

## アプローチ

exp065 の `common_typewell_cluster_assignments.csv` を新しい read-only input として解決し、
`method=native_overlap`、`threshold=0.999` の773行を抽出する。対応表を
`cluster_id`、`well_id` の stable sort にかけ、clusterを1始まりの `typewell_order`、
全wellを1始まりの `plot_order` として採番する。

PNG名は `typewell_{typewell_order:04d}_{well_id}.png` とする。この名前ならKaggle output UIと
zipの辞書順が共通 typewell順になり、同一typewell内はwell ID順になる。manifestは描画順のまま
保存し、typewell metadataと実ファイル名を持たせる。zipはfilesystem globの再sortではなく
manifestの `plot_filename` 順に格納する。

## 実験範囲

- 対象実験: `exp238_nested_hmm_exp226_selector_rank_slot_addonly_on_exp218`
- Route: `ml_model`
- 親実験: exp238 selector-confidence plot v2（指定 `scriptVersionId=335655690`）/ exp065 common typewell discovery
- 変更する変数: PNGの列挙順・ファイル名、manifest/summaryのtypewell metadata、exp065 kernel source
- 固定する変数: 3,783,989 OOF rows、773 wells、outer-valid selector surface、11 candidates、全線色、plot内容、全RMSE、診断専用判定

## 再現性設計

- seed policy: 新規乱数なし。対応表とstable sortだけで順序を決める。
- stochastic 処理の有無: なし。
- PF/Beam / likelihood-PF / seed bagging の有無: 保存済み候補を従来どおり表示するだけで、再生成しない。
- 並列処理と乱数の関係: 並列処理も乱数も追加しない。
- CPU/GPU runtime と deterministic flags: CPU、internet disabled、model fit 0を維持する。
- train cache / test feature regeneration の SHA 記録方針: exp065対応表のfile SHAとmethod/threshold/row/well/group contractをsummaryへ保存する。既存input SHA contractは維持する。
- model manifest / prediction / submission SHA 記録方針: model・prediction・submissionは新規生成しない。manifest/plots zip/summary SHAは実行時に従来どおり保存する。
- Kaggle package bootstrap 確認方針: self-contained notebookのcanonical/package cell source一致と、metadataにexp065 kernel sourceが1件だけ追加されていることを確認する。

## リスク

- リークリスク: typewell順はtrain typewell GRだけから作られたtarget-free対応表であり、OOF scoreやtrue TVTによる順序付けを禁止する。
- CV/LB 不一致リスク: 可視化順だけの変更なのでCV/LBと採用判断は変えない。summaryにも診断専用を残す。
- ランタイム/メモリリスク: 11,595行の対応表追加だけで影響は軽微。PNG 773件の描画本体は従来と同じ。
- 再現性リスク: 対応表のmethod/threshold曖昧化や欠損fallbackで順が変わり得るため、列・重複・coverage・54 groupsをfail-fastし、file SHAを保存する。
