# exp204_heatmap_mdn_topk_as_selector_candidates_on_exp158

## 状態

closed / rejected。heatmap から path を生成して selector 候補にする route は閉じた。exp202/207/208/210/212/215 で oracle headroom は見えたが、生成 path 自体が弱く、通常候補として採用する根拠が不足した。

## 仮説

当初仮説は、exp202 heatmap MDN topK を既存 PF/Beam/dense 候補に足すと oracle headroom が増えるため、selector が target-free に有効候補を選べるかを確認するものだった。実際には exp203 feature-only は exp184 を更新せず、exp212 は fallback-heavy、exp215 は fallback 0.0 でも learned-only / weighted path が弱かったため、この仮説は採用しない。

## 目的

履歴として、旧 exp204 実装と失敗理由を残す。今後の実験入力としては使わない。

## 方針

- 親: `exp203_heatmap_mdn_candidates_into_selector_features`
- 比較基準: `exp184_cnn_sdf_mtp_heatmap_path_features_on_exp158`
- route: `pf_beam`
- historical selectable candidates: 18
- 追加 candidate prefix: `hmdn_top`
- 既存 `hmpf_` / `hmdn_` confidence feature は維持
- parent/control retraining: なし
- 再実行: なし

## 評価

旧実装の Kaggle v3 はユーザー判断で中断したため、CV として採用しない。exp202/203/207/208/210/212/215 の結果から、heatmap path 生成 route は diagnostic history に留める。

## 検証方針

追加の Kaggle train / inference / submit は行わない。heatmap MDN/MTP paths は selectable candidate、direct replacement、softmax weighted TVT、PF weight replacement、postprocess blend、inference port、submit に使わない。

## 所見

旧実装は exp202 top10 を `ranker.candidates` に追加し、初期 cache には存在しない `hmdn_top*_tvt` を後段生成候補として扱うようにした。ただし full-grid coverage 前提を満たさず、後続 artifact でも生成 path の弱さが解消しなかったため、exp204 は closed/rejected とする。
