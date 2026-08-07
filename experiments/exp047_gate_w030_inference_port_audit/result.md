# exp047_gate_w030_inference_port_audit 結果

## 仮説

`exp047_public_pf_beam_gate_only_audit` で固定 `exp026_to_pf_gate_w0p30` は original-fold、well-hash、stratified-group の全 surrogate split で最良だった。learned gate や meta residual ではなく、この固定 gate だけを見えない test well 用推論に移すと、PF/Beam の転移失敗リスクを抑えつつ改善候補を検証できる可能性がある。

## 設定

- 親: `exp047_public_pf_beam_gate_only_audit`
- 実装親: `exp045_public_pf_meta_strict_parity_audit`
- 推論候補: `exp026_anchor + 0.30 * (public_pf_pred - exp026_anchor)`
- 適用先: hidden / unseen test wells only
- visible public sample branch: 既存 physical branch を維持
- anchor: exp026-style pseudo-tail distance bucket shrink

## 結果

Kaggle inference version 2 が完了した。version 1 は `audit.distance_buckets` 欠落により exp026 anchor 学習で失敗したが、config を修正して version 2 で完了した。

| メトリック | 値 |
| --- | --- |
| kernel | `kentookumura/exp047-gate-w030-infer` v2 |
| rows | 14,151 |
| visible sample wells | 3 / 3 |
| hidden_rows / hidden_wells | 0 / 0 |
| changed_rows / changed_wells | 0 / 0 |
| diff RMSE | 0.000000 |
| prediction range | 11587.038593 - 12240.016066 |
| exp026 anchor fit | 773 wells / 242,843 rows / 788 source rows |
| PF settings | 16 seeds / 250 particles |
| submit-check | PASS |
| submission SHA256 | `2b86386f19279e79e7184096f353ccf2b97785de67b268caa56aa5f85405a815` |
| code submit | ref `53509425`, UI submit |
| Public LB | 11.056 |

## 解釈

Public sample は全 3 wells が train に存在する visible wells で、従来の physical branch を使った。そのため hidden branch の fixed gate は sample output 上では発火せず、submission SHA は exp027 系と同一になった。これは想定どおりだが、output sanity だけでは `exp026_to_pf_gate_w0p30` の Public LB 転移を確認できない。

Kaggle 実行経路、exp026 anchor fit、PF 設定、artifact 保存、submit-check は確認できた。そのうえで UI から code submit した結果、Public LB は 11.056 だった。exp027 の 8.781 から +2.275 悪化し、exp031 の fixed `pf090_hold010` 8.956 よりも +2.100 悪い。

固定 `w=0.30` は exp031 の `pf090_hold010` より保守的だったが、hidden branch の Public LB 転移には失敗した。exp047 の train-side surrogate 改善は、見えない test well 用の実 Public LB には転移しなかった。

## 次

exp027 anchor 8.781 を維持する。public PF gate inference port は停止する。次は PF/Beam hidden branch ではなく、ML route の XGBoost、LightGBM micro tune、seed bagging のような別候補へ戻る。
