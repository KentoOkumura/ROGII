# 要件

## 依頼

`supervised_gr_window_matcher_from_known_tvt_prefix` backlog を実装する。既知 `TVT_input` 区間で GR matching scorer を教師あり学習し、real GR が shuffled/no-GR control を上回るかを 1 fold / row cap smoke で確認できる状態にする。

## 制約

- Route: pf_beam
- 再現性: `docs/06_reproducibility.md` に従い、stable seed、SHA、Kaggle bootstrap の扱いを config と SESSION_NOTES に記録する。
- Kaggle Notebook 実行を正とし、ローカル notebook 実行はしない。
- known `TVT_input` prefix row だけを label/window center に使う。
- `TVT_input` NaN 評価区間や tail true TVT を特徴生成、正例、threshold 選択に使わない。
- direct TVT regression、hard path replacement、softmax weighted average、candidate midpoint、PF weight 直接置換、inference port、submission は対象外。
- Kaggle train push 前の GPU/booster コストを明記する。

## 受け入れ基準

- `exp178_supervised_gr_window_matcher_from_known_tvt_prefix` が作成され、config/README/SESSION_NOTES/result/metrics が TODO なしで記載されている。
- Jupytext percent 形式の train script から正規 train notebook を生成できる。
- train notebook で pair dataset、real/shuffled/no-GR controls、expected-error regressor、pair/rank/by-well metrics、SHA 付き artifact 保存を追える。
- `make validate-exp EXP=exp178_supervised_gr_window_matcher_from_known_tvt_prefix` が通る。
- deterministic anchor ではないこと、submission を作らないことが記録されている。
- gzip 生成物は raw SHA と decompressed content SHA を summary JSON に記録する。
