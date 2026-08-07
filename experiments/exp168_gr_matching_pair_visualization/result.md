# exp168_gr_matching_pair_visualization 結果

## 仮説

GR matching の採用 pair を図示すると、score 集計だけでは分からない波形一致、decoy、prior 依存の失敗を確認できる。

## 設定

- 親: exp167_fft_denoised_gr_matching_audit
- 検証: train-side visualization diagnostic
- メトリック: scored pair metadata と可視化 PNG / HTML
- シード: 42

## 結果

| メトリック | 値 |
| --- | --- |
| CV | - |
| Public LB | - |
| Private LB | - |
| Kaggle kernel | `kentookumura/exp168-gr-matching-pair-visualization-train` v5 COMPLETE |
| rows scored | 4096 |
| wells scored | 16 |
| figures | 48 global/local diagnostics + 32 legacy four-panel + 32 simple overlay |
| OOF join | exp098 lgb1 OOF error matched 512 selected/scored IDs, after scanning 15,135,956 rows |
| wrong-depth buckets | `lt6ft`, `ge6_lt10ft`, `ge10_lt15ft`, `ge15ft` |
| output | `kaggle/output/train_v5/artifacts/` |

## 実装検証

- `py_compile`: 通過
- `ruff --select F821,F722,F823`: 通過
- Jupytext 変換 / `--test`: 通過
- `validate-exp`: 通過
- Kaggle package: `kentookumura/exp168-gr-matching-pair-visualization-train` を生成済み
- Kaggle v1: kernelspec metadata 不足で起動前 ERROR
- Kaggle v2: kernelspec `python3` を追加して COMPLETE
- Kaggle v3: query vs matched overlay と exp098 lgb1 OOF error good/bad HTML を追加して COMPLETE
- Kaggle v5: top-k local minima、true-near minimum、shift-cost curve、全体 GR context、wrong-depth bucket 別 OOF 集計を追加して COMPLETE

## 再現性

- deterministic anchor: いいえ。可視化診断であり submission anchor ではない。
- seed policy: no RNG deterministic sampling
- kernel version: `kentookumura/exp168-gr-matching-pair-visualization-train` v5
- feature content SHA: scored pairs decompressed SHA `0e86bbf8b3433acf18a72bbe11950626e2425c8ab863cf1784218c17da4af69a`
- model SHA / manifest SHA: 対象外
- prediction SHA: 対象外
- submission SHA: 対象外
- rerun result: 未実行

## 解釈

v5 の可視化生成は成功した。主ビューは `wrong_depth_index.html` で、OOF abs error を wrong-depth bucket
別に集計し、その下に bad / ambiguous / good 例の global/local 図を並べる。各図では水平井全体 GR と query
window、typewell 全体 GR と best / alternative / true-near / true center、局所 waveform overlay、shift-cost
curve、top local minima 表を同時に確認できる。

代表的には、`028d7b28 hidden_tail raw row=4945` は best が true から 330 ft 外れており、true-near minimum
の cost が best より 13.20 高い。GR 形状として wrong depth を強く選んでいる例である。一方、
`015fe0d2 hidden_tail raw row=3039` は best と second の cost が同値で、true-near との差も 0.69 と小さい。
これは複数の似た波形が競る ambiguous wrong-depth 例である。`015fe0d2 hidden_tail raw row=1654` は best と
true-near が一致する good match 例である。

OOF bucket summary は 16 wells / 4096 scored rows / 512 OOF matched IDs の診断集計であり、wrong-depth が
そのまま OOF 悪化を単調に説明するとは読まない。目的は、GR shift-scan が「強く間違う」のか「近い複数解で
迷っている」のかを見分け、PF/Beam の hard commit や posterior 化の候補を検討するための readout とする。

## 次

`wrong_depth_index.html` を最初に見て、red best と cyan true-near の cost 差、gray alternatives の密度、
OOF abs error を同時に確認する。true-near delta cost が小さい wrong-depth 例は posterior / top-k candidate
化、delta cost が大きい例は GR 単独尤度以外の prior / likelihood 補正候補として扱う。
