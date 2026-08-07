# exp222_row_step_delta_target_ablation_on_exp148 結果

## 仮説

exp148 feature surface のまま教師を row-to-row step delta に変えると、well 内の形状一貫性が改善し、long-tail の TVT RMSE が改善する可能性がある。ただし step delta は誤差が累積するため、cumulative drift と worst-well regression が主な失敗モードになる。

## 設定

- 親: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- 検証: 5-fold GroupKFold by `well`
- メトリック: predicted step delta を well ごとに累積復元した後の TVT RMSE
- シード: 42
- Runtime: Kaggle CPU
- 学習対象: `step_delta_target_lgb0` 1 variant x CPU 1 mode x `lgb0` x 5 folds = 5 boosters
- 親 control 再学習: なし

## 結果

| メトリック | 値 |
| --- | --- |
| CV | 15.301575123885728 |
| Public LB | なし |
| Private LB | なし |

Kaggle train v1 は `kentookumura/exp222-stepdelta-lgb0` で実行したが、入力確認後に `DeadKernelError: Kernel died` で失敗した。ログ上の最後の正常出力は exp145 learned likelihood cache `3,783,989` rows / `773` wells / `51` columns の読み込みで、LightGBM 学習開始前に落ちている。原因は preview 目的で learned likelihood cache を全量ロードし、その後本処理でも再ロードしていたことによる Kaggle CPU RAM 圧迫と判断する。

v2 修正では、入力確認セルを header + `nrows=8` の軽量 preview に変更し、helper 側も learned feature join のキー順一致時に大きな `merge` を避ける fast path と一時 DataFrame の明示破棄を入れた。しかし v2 も full feature assembly 中に、LightGBM 学習開始前の `DeadKernelError: Kernel died` で失敗した。v2 の最後の正常出力は `learned feature preview rows: 8 columns: 51`。

v3 では full-frame `copy()`、巨大な一括 finite check、anchor merge、target sort の full feature copy を削減し、主要段階に `stage` / peak RSS log を追加した。同じ kernel に version 3 として push し、完了した。URL: https://www.kaggle.com/code/kentookumura/exp222-stepdelta-lgb0

fold RMSE は 14.576648 / 15.961225 / 14.807865 / 16.358125 / 14.716812。pooled `lgb0` / `lgb_mean` は同一で、復元 TVT RMSE 15.301575、step-delta target RMSE 0.036166。exp148 lgb0 8.599786 から +6.701789、exp148 lgb_mean 8.501281 から +6.800294 の大幅悪化。

distance bucket では `000_050` は 0.601446 と改善したが、`1000_plus` は 16.933071 で exp148 lgb_mean bucket 9.325405 から +7.607666 悪化した。worst-well は `1b1eba53` 67.727455、`896d15b9` 58.371578、`81bf5923` 51.912468。cumulative drift の final error は、それぞれ -69.507812、-84.615234、+78.115234 と大きく、well-wise cumsum が誤差を蓄積している。

## 再現性

- deterministic anchor: いいえ。train-side ablation であり submission anchor ではない。
- seed policy: LightGBM config seed と GroupKFold seed 42。
- kernel version: v1 failed; v2 failed; v3 complete。
- feature content SHA: exp072 source `14faee3a24de587b7190e7febffd33cc1256e418c7505f2d1e6f5f7c9b3c2f18`。
- model manifest SHA: `a1bb1bbf20c35ffa23ecb8a4811d6314cd2b6e49c8ff57857f4b8181e4974d25`。
- prediction SHA: `5c807ad31c3ae3604a6c59b2bd130405b62c704b8a41ad6c364e80c7ba1e281e`。
- submission SHA: submission を生成しない。
- rerun result: 未実施。

## 解釈

row-to-row step delta target は train-side で明確に不採用。raw step-delta RMSE は小さいが、TVT へ累積復元すると long-tail と worst-well が大きく壊れる。これは、このコンペの tail 長では「小さい局所誤差」が well-wise cumsum で許容できない global drift になることを示している。lgb1/lgb2 展開、inference port、submit は行わない。

## 次

次は recursive delta prediction を直接モデル化するのではなく、累積 drift を予測・抑制する診断や posthoc guard の材料として扱う。
