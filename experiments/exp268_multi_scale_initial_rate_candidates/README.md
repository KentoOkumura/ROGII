# exp268 multi-scale initial-rate candidates

## 状態

- ルート: `pf_beam`
- 状態: Kaggle CPU shard 0/1・aggregate完了
- rows / wells: 3,783,989 / 773
- tail30 RMSE: 11.938287
- initial-rate-5 H256 oracle RMSE: 11.836137（headroom 0.102151 ft）
- Public / Private LB: 対象外
- 親実験: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`

## 仮説

exact HMMの初期rateをprefix末尾30行の単一推定に固定せず、known prefix末尾
`32/64/128/256`行のrobust slopeを独立candidateとして残すと、HMM grammarとGR emissionを
変えずにblock / whole-wellで候補headroomを得られる。

## 変更点

- 保存済みexp209 `tail_n=30` HMMをcontrolとして再利用し、再生成しない。
- `median((delta TVT_input + delta Z) / delta MD)`のwindowだけを`32/64/128/256`へ変える。
- 全4候補をtarget-free SHA256 well shard 2本で生成し、正規train notebookで統合する。
- candidate平均、blend、oracle deploy、selector、weight学習、inference、submissionは実装しない。

## 実行契約

- 4 HMM variants / 2 well shards / 3,092 HMM well-runs
- LightGBM config / fold / booster: 0 / 0 / 0
- GPU / control再生成 / inference / submission: なし / なし / なし / なし
- aggregate kernel id: `127887734`

## 検証方針

- 全773 wells / 3,783,989 rowsのcoverage、id整合、shard/aggregate SHAをhard guardする。
- row、H128/H256/H512 block、whole-well oracleはheadroom診断に限定する。
- true TVTは候補生成後のreadoutにだけ使い、oracle predictionを保存しない。

## 実行入口

- shard 0: `exp268_multi_scale_initial_rate_candidates_train_variant0.ipynb`
- shard 1: `exp268_multi_scale_initial_rate_candidates_train_variant1.ipynb`
- aggregate: `exp268_multi_scale_initial_rate_candidates_train.ipynb`
- inference: disabled guardのみ

## 所見

best direct rate candidate `w128`はtail30から0.042706 ft改善し、initial-rate-5 bankのH256 oracle
headroomは0.102151 ft、whole-well headroomは0.097314 ftだった。一方、423/773 wellsはrate spread 0で、
候補間path duplicate率も高い。子実験exp292ではtarget-free識別性がFAIL-closeとなった。

## 次

候補bankをML feature、selector、raw-test inference、submissionへ昇格しない。同一truth上のwindowや
選択規則の救済探索も行わない。
