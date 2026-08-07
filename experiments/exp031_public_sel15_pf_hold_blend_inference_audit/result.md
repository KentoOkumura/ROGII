# exp031_public_sel15_pf_hold_blend_inference_audit 結果

## 仮説

`exp030` の fixed `pf090_hold010` は、公開 sel15 selector の大外しをわずかに hold 側へ縮めることで、`exp027` replay より安定する可能性がある。

## 設定

- 親: `exp027_public_replay_needless090_sel15_spread3`
- supporting audit: `exp030_public_sel15_pf_candidate_selector`
- 検証: Kaggle inference output diff、submit-check、必要なら Public LB
- メトリック: RMSE
- blend: 見えない test well のみ `0.90 * selector + 0.10 * last_known_TVT_input`

## 結果

| メトリック | 値 |
| --- | --- |
| CV | - |
| Kaggle kernel | `kentookumura/exp031-public-sel15-pf-hold-blend-inference-audit` v1 completed |
| Kaggle output SHA256 | `2b86386f19279e79e7184096f353ccf2b97785de67b268caa56aa5f85405a815` |
| Submit-check | PASS |
| Rows | 14,151 |
| exp027 diff RMSE | 0.000000 |
| Changed rows | 0 |
| Submit ref | `53443300` |
| Public LB | 8.956 |
| Private LB | - |

## 解釈

Kaggle inference version 1 は完了し、submit-check は PASS。公開 sample output は exp027 と完全一致した。理由は public test の 3 wells がすべて train にも存在し、今回変更していない physical-model branch が使われたため。したがって public sample の `submission.csv` は exp027 と同じ SHA256 になり、新規 file submission の価値はない。

code competition の hidden test では 見えない test wells が `train_wids` に入らないため、今回変更した `pf090_hold010_hidden` branch が使われた。提出 ref `53443300` の Public LB は 8.956 で、exp027 基準 8.781 から +0.175 悪化した。

したがって fixed `pf090_hold010` 見えない test well 用処理は採用しない。`exp030` の train well の途中以降を隠した疑似 test OOF-like 改善は hidden Public LB にそのまま移らなかった。

## 次

exp027 を public sel15 基準として維持し、固定 blend ではなく residual 補正 / meta-stack の fold-safe validation に戻る。
