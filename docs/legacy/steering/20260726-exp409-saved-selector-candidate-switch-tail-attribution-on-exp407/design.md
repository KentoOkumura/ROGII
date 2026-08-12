# 設計

## アプローチ

1. 親exp264 corrected Stage B v5とexp407 Stage B v1の
   `candidate_score_oof.parquet`をfile SHAで照合する。
2. `actual_abs_error`を一切読まないPhase 1で、両surfaceについて同じ11候補domainの
   `pred_abs_error` argminをcandidate order安定tie-breakで再現する。
3. base key、candidate順、candidate値、fold、schema/contract SHAのparityを確認し、
   親→exp407 transition、distance bucket、hidden-like roleを
   `selection_freeze.parquet`へ保存してSHAを固定する。
4. freeze完了後のPhase 2だけで両OOFの`actual_abs_error`を読み、同一候補の
   error parityを検証して、選択された候補のSSE差を行単位で計算する。
5. transition、fold、distance、hidden-like、well別にrows、SSE、RMSE、
   positive excess SSE shareを集計する。
6. 1000+とhidden-like 2面で各foldの最大positive excess-SSE遷移を決め、
   同じ遷移が各scope 4/5 foldsかつ固定worst wellでもrank-1の場合だけ
   `candidate_switch_tail_cause_supported`とする。それ以外は
   `diffuse_or_nonreproducible_candidate_switch_cause`とする。

## 実験範囲

- 対象実験:
  `exp409_saved_selector_candidate_switch_tail_attribution_on_exp407`
- Route: `ml_model`
- 親実験:
  `exp407_fold_safe_inverse_rmse_weighted_dual_selector_on_exp264`
- 比較対象:
  `exp264_exp263_candidate_confidence_dual_selector` corrected Stage B v5
- 変更する変数: なし。保存surface間の選択遷移を診断するだけ。
- 固定する変数: OOF SHA、12候補順、11候補domain、candidate values、
  fold、distance bucket、hidden-like assignment、worst well。

## データフロー

```text
parent/exp407 candidate-score OOF
          |
          | pred_abs_error + target-free columns only
          v
truth-free selection + transition freeze -- SHA固定
          |
          | freeze完了後だけ actual_abs_error
          v
row-level additive excess SSE
          |
          +--> transition / fold / distance
          +--> hidden-like 2面
          +--> well / fixed worst well
          v
ordinal 4-of-5 concentration readout
```

SSE差は加法的なので、集計scopeの悪化をtransition別に過不足なく分解できる。
RMSE差だけをtransitionへ足し合わせることはしない。

## 再現性設計

- seed policy: RNGなし、保存artifactだけ。
- stochastic 処理の有無: なし。
- PF/Beam / likelihood-PF / seed bagging の有無: なし。
- 並列処理と乱数の関係: single process、乱数なし。
- CPU/GPU runtime: CPUのみ、GPU 0。
- input SHA: 両candidate OOFとhidden-like assignmentを実行前に照合。
- freeze SHA: truth列を読まずに書き出した直後に記録。
- truth ledger: Phase 1のtruth read 0、Phase 2の`actual_abs_error`だけを記録。
- model / prediction / submission SHA: 対象外。model、prediction、submissionを生成しない。
- Kaggle bootstrap: 実行承認後にcanonical packageを作る場合だけ、
  configとinput sourceをbootstrapから再照合する。
- deterministic anchor: implementation時点ではfalse。Kaggle version未作成、
  parent private input未作成のため。

## リスク

- リークリスク: truthを見てtransition軸やscopeを選ぶとpost-hoc rescueになる。
  Phase 1/2を物理的に分け、freeze manifestでtruth read 0を監査する。
- 帰属リスク: RMSE差は非加法。主帰属量をSSE差とし、RMSEは説明用に限定する。
- candidate alignment: 両Parquetのrow/candidate順がずれる可能性がある。
  batchごとにkey、candidate order、value、foldを全件照合してfail closedする。
- hidden-like: assignmentは診断専用で学習・選択に使わない。
- ランタイム/メモリ: 入力合計約1.33 GB。20,000 base-row batchでstreamし、
  全candidate-longを同時にpandasへ載せない。
- Kaggle入力: exp407 OOFはkernel sourceに存在するが、親corrected Stage B v5
  OOFは現在のkernel最新版にない。private dataset等の作成は別承認まで行わない。
- 再現性: parent input sourceとKaggle kernel versionが未固定の間はanchorと呼ばない。
