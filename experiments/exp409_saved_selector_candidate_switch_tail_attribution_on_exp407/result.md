# exp409_saved_selector_candidate_switch_tail_attribution_on_exp407 結果

## 状態

Kaggle private CPU version 1完了。technical PASS、事前登録した
tail consistency gateはFAIL。exp407はscientific FAILのまま閉鎖する。
inference、submissionは未実施。

## 目的

exp407 hard-primaryのtail悪化を、corrected exp264 Stage B v5からexp407への
row-level selected candidate遷移へ保存OOFだけで加法的に帰属する。

## 実装契約

- model / booster / prediction / PF / HMM / Beam: 0
- Phase 1: truth-free selection、transition / scope freeze、SHA保存
- Phase 2: freeze後だけ`actual_abs_error`を読み、SSE差を集計
- 固定scope: 1000+、hidden-like 2面、worst well `52f1e77a`
- gate: 同一rank-1 positive excess-SSE transitionが各tail scope 4/5 foldsかつ
  worst wellでもrank-1
- exp407救済、再分類、threshold/weight/candidate gridは禁止

## 結果

### 実行

- kernel:
  `kentookumura/exp409-selector-switch-tail-attribution-train` version 1
- id_no: `128678587`
- runtime: private CPU、internet off。gate出力まで約179.259秒
- 親OOF input:
  `kentookumura/exp409-exp264-stage-b-v5-oof-input`
- model / booster / prediction / inference / submission: `0 / 0 / 0 / 0 / 0`

### Technical integrity

- 3,783,989行を処理し、期待行数と一致した。
- 親OOF、exp407 OOF、hidden-like assignmentの実読込SHAは事前固定値と一致した。
- truth-free phaseの禁止truth readは0件。`selection_freeze.parquet`のSHAを
  確定してから`actual_abs_error`を読む2-phase ledgerを確認した。
- 1,289,588行（34.0801%）でselected candidateが変化し、121 transitionを
  固定した。
- ダウンロードしたgate、manifest、集計CSV 9件のSHAは
  `reproducibility_manifest.json`と一致した。大きなrow-level Parquetは
  ダウンロードせず、Kaggle出力SHAだけを記録した。

### Attribution

| scope | corrected exp264 RMSE | exp407 RMSE | delta |
| --- | ---: | ---: | ---: |
| overall | 8.587004 | 8.668141 | +0.081137 |
| distance 1000+ | 9.432345 | 9.523577 | +0.091232 |
| hidden-like spatial | 9.516693 | 9.620452 | +0.103759 |
| hidden-like typewell-purged | 9.415269 | 9.494321 | +0.079052 |

overall最大の正の寄与は
`exp226_k16__selfgr_hmm_a070 -> likpf_mean__exact_hmm`で、
delta SSEは`+2,058,442.514`だった。固定worst well `52f1e77a`でも同遷移が
rank-1で、positive excess SSE shareは約85.99%だった。

ただし1000+とhidden-like 2面のfold別rank-1は分散した。同一遷移が各scopeで
4/5 foldsを満たす候補は0件で、上記worst-well遷移も1000+では1/5、
hidden-like 2面では0/5だった。

### Gate

- `passed: false`
- `cause_transition: null`
- decision:
  `diffuse_or_nonreproducible_candidate_switch_cause`
- exp407 status change:
  `none_exp407_remains_scientific_fail_closed`

したがってexp407のtail悪化を、foldとtail scopeをまたいで再現する単一の
parent-to-exp407 candidate switchへ帰属する仮説は支持されなかった。
worst well単体の強い集中は局所現象であり、selector全体の救済根拠にはしない。

## 次

exp409を完了済みとしてバックログから削除し、この原因分解branchを閉じる。
weight、threshold、clip、exponent、candidate domainのsame-OOF救済は行わない。
新しいselector救済案は追加せず、次の優先候補は独立familyの既存P2
`predictive_rate_innovation_reset_preflight`を維持する。
