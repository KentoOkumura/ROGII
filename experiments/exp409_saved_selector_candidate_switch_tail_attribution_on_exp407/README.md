# exp409_saved_selector_candidate_switch_tail_attribution_on_exp407

## 状態

- ルート: `ml_model`
- 状態: private CPU v1完了、technical PASS、tail consistency gate FAILで閉鎖
- CV / Public LB / Private LB: 対象外 / - / -
- 作成日: 2026-07-26
- 親実験: `exp407_fold_safe_inverse_rmse_weighted_dual_selector_on_exp264`
- 比較対象: `exp264_exp263_candidate_confidence_dual_selector` corrected Stage B v5

## 仮説

exp407のinverse-RMSE weightingが特定候補の局所signalを弱めたなら、
親exp264からexp407への同じdirected candidate switchが、1000+とhidden-like 2面の
各scopeで4/5 foldsの最大positive excess-SSE原因となり、exp407で固定済みの
worst well `52f1e77a`でも最大原因になる。

## 変更点

- 保存済みの親/exp407 candidate-score OOFだけを読む。
- hard selectorを同じ11候補domainとcandidate-order tie-breakで再現する。
- truthを読む前にtransition、distance、hidden-like roleをfreezeしてSHAを保存する。
- freeze後だけ`actual_abs_error`をjoinし、`exp407 SSE - parent SSE`を帰属する。
- model、booster、candidate生成、prediction、inference、submissionはすべて0。

## 実装・実行

- 正規train Notebook:
  `exp409_saved_selector_candidate_switch_tail_attribution_on_exp407_train.ipynb`
- Jupytext正本:
  `exp409_saved_selector_candidate_switch_tail_attribution_on_exp407_compact_selfcontained_train.py`
- Kaggle kernel:
  `kentookumura/exp409-selector-switch-tail-attribution-train` version 1、
  id_no `128678587`
- runtime: private CPU、internet off。gate出力まで約179.259秒
- parent private Dataset:
  `kentookumura/exp409-exp264-stage-b-v5-oof-input`

## 検証方針

- 5 outer folds、well group、TVT_input欠損行を固定する。
- Phase 1ではtruth列を拒否し、selection / transition / tail scopeをfreezeする。
- Phase 2だけで`actual_abs_error`を読み、additive excess SSEを集計する。
- 同じrank-1 transitionが1000+とhidden-like 2面で各4/5 folds、
  固定worst wellでもrank-1の場合だけ仮説を支持する。

## 結果

- technical checksと入力SHAはPASS。truth-free phaseの禁止truth readは0。
- 3,783,989行中1,289,588行（34.0801%）でcandidateが変化した。
- overall RMSEはparent `8.587004`、exp407 `8.668141`
  （`+0.081137 ft`）。
- 1000+ `+0.091232`、hidden-like spatial `+0.103759`、
  typewell-purged `+0.079052 ft`。
- 固定worst wellでは
  `exp226_k16__selfgr_hmm_a070 -> likpf_mean__exact_hmm`が
  positive excess SSEの約85.99%を占めた。
- ただし同じtransitionが全3 tail scopeで各4/5 foldsのrank-1になる条件は
  0件だった。

gateは`diffuse_or_nonreproducible_candidate_switch_cause`でFAIL。
単一candidate switch原因仮説は支持されず、exp407はscientific FAILのまま維持する。

## 所見

固定worst wellでは原因が強く集中したが、そのtransitionはfoldとtail scopeを
またいで再現しなかった。したがって局所的な失敗例をglobal selector変更の根拠に
一般化できず、悪化は複数のcandidate switchへ拡散したと解釈する。

## 次

exp409原因分解branchを閉じ、selectorのsame-OOF rescueは行わない。
新しいselector案は追加せず、既存P2
`predictive_rate_innovation_reset_preflight`の優先度を維持する。
