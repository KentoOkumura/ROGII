# exp204_heatmap_mdn_topk_as_selector_candidates_on_exp158 結果

closed / rejected。heatmap から path を生成して selector 候補にするアイデア自体を閉じる。exp202/207/208/210/212/215 で oracle headroom は確認できたが、生成 path 単体は弱く、selector の通常候補として扱う根拠が不足した。

## 判断

旧実装は exp202 row-interpolated heatmap paths を候補化する設計だったが、full-grid trajectory ではなかった。exp212 は full-grid contract を満たしたものの fallback-heavy / endpoint hold tail が残った。exp215 は fallback 0.0 の MTP full-tail artifact を作れたが、learned MTP top5 only は RMSE 32.333142886、weighted path は RMSE 59.272141581 と弱かった。

exp203 feature-only も exp184 best を更新しなかったため、heatmap path は selectable candidate、direct replacement、softmax weighted TVT、PF weight replacement、postprocess、inference port、submit のいずれにも使わない。exp202/203/207/208/210/212/215 の artifacts は diagnostic history として残す。

## 次の扱い

- `heatmap_mdn_topk_as_selector_candidates_on_exp158` backlog は active backlog から外す。
- exp204 は再実行しない。
- heatmap 由来 path 生成を再開する場合は、この exp204 の小修正ではなく、新しい根拠と別仮説として扱う。
