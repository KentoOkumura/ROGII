# exp440_ambiguity_gated_predictive_prior_hmm

## 状態

- Route: `pf_beam`
- 最終状態: `stage1_full_oof_failed_closed`
- CV: candidate `12.992063` / parent exp209 `11.938287` RMSE
- Public / Private LB: なし
- inference / submission: 無効
- 親: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`

## 仮説

raw GR observed行で、通常emission適用後のcausal filtered TVT marginalが
exp236固定契約で二峰と判定されたときだけ、emission lambdaを`1.0 -> 0.0`にし、
親transition後のpredictive分布を保持するexact HMM候補である。GMM、soft gate、
threshold探索、点推定TVTのhard freezeは使わない。

Stage 0 fixed32はtechnical 13/15、mechanism 2/8でFAILした。その結果を維持した
まま、明示依頼により念のためfull 773 wellsを4 CPU shardsで確認した。

## 検証方針

- 773 wellsをsuffix rowsのdeterministic LPTで4 shardsへ一意に分割する。
- candidate 1本だけを各wellで1回実行し、保存済み親predictionと比較する。
- 全prediction / schedule / diagnosticをstrict merge・SHA freezeした後だけ
  truth、fold、hidden-like roleを結合する。
- 全体gain、5 folds、ambiguous-row SSE、6 safety scopes、by-well p95 /
  worst-wellを事前固定gateで判定する。

## Full OOF

- 773 wells / 3,783,989 rows
- candidate HMM well-runs 773、保存parent control rerun 0
- technical gate: 全PASS
- candidate RMSE: `12.992063 ft`
- parent exp209 RMSE: `11.938287 ft`
- gain: `-1.053776 ft`
- positive folds: `1/5`
- ambiguous-row SSE reduction: `-21.3117%`
- by-well delta p95 / worst: `+11.631749 / +45.003490 ft`
- raw observed / missing、高欠損、MD 1000+、hidden-like 2 scopes:
  すべて悪化

## 実行入口

- 正規train:
  `exp440_ambiguity_gated_predictive_prior_hmm_train.ipynb`
- shard wrappers:
  `*_train_variant0.ipynb`から`*_train_variant3.ipynb`
- strict merge:
  `exp440_ambiguity_gated_predictive_prior_hmm_train_aggregate.ipynb`
- inference guard:
  `exp440_ambiguity_gated_predictive_prior_hmm_compact_selfcontained_inference.ipynb`

Kaggle merge version 1:
`kentookumura/exp440-ambiguity-gated-predictive-prior-hmm-merge`。

## 所見

predictive prior holdはfull OOFで明確に悪化した。事前契約どおり
`close_without_blend_selector_continuous_gate_or_same_oof_rescue`とし、
rerun、inference、submissionへ進まない。
