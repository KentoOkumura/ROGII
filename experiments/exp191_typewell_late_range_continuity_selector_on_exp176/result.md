# exp191_typewell_late_range_continuity_selector_on_exp176 結果

## 状態

Kaggle train v1 完了。train-side continuity audit としては支持。ただし direct inference / submit はしない。

## 仮説

exp176 の typewell late-range prior 入り row-wise selector は global OOF で強いが、path switch が高いため direct submit には不安定。exp158-style Viterbi continuity selector を同じ score surface に適用すれば、exp176 の RMSE 改善を保ちながら path switch と短い segment を抑えられる可能性がある。

## 実装

- exp176 feature schema / model manifest / saved boosters を読む。
- raw train typewell context から exp176 と同じ `tlp_` row feature を復元する。
- candidate-long score 復元では exp176 v3 と同じく row-level `tlp_` を除外し、`candidate_tlp_` feature を追加する。
- exp158 と同じ Viterbi grid 180 variants を評価する。
- `selected_tvt` の direct replacement、blend、postprocess、submit はしない。

## CV

- Best: `viterbi_sw400_bias000_jw050_jf025_d075_std999999_md0000_seg012`
- RMSE: 10.598006880
- MAE: 6.402336928
- within10: 0.793110657
- oracle label accuracy: 0.265050718
- `likpf_mean_single`: 11.594897672
- exp176 row-wise `lgb_candidate_error_ranker`: 10.641296371
- exp158 best Viterbi: 10.789163253
- delta vs `likpf_mean_single`: -0.996890792
- delta vs exp176 row-wise: -0.043289491
- delta vs exp158 best Viterbi: -0.191156373

## Continuity

- best Viterbi path switches: 3,620 / 0.956662 per 1000 rows
- exp176 row-wise path switches: 261,391 / 69.078161 per 1000 rows
- max path switch per 1000 rows by well: 4.045605
- exp176 row-wise max path switch per 1000 rows by well: 330.841857
- exp176 row-wise 比で 417 wells 改善、356 wells 悪化
- max regression: +1.450447 RMSE (`6d1d74e1`)
- max improvement: -10.746658 RMSE (`57f05c51`)

## 所見

global RMSE と path continuity は exp176 row-wise より良い。特に path switch は 261,391 から 3,620 まで下がり、row-wise selector の不安定さはほぼ解消した。一方、distance bucket は `1000_plus` が -0.051216 改善する反面、`000_050`、`050_100`、`250_500` は小幅に悪化する。

結論として、exp191 は train-side continuity audit として完了/支持。direct selected TVT submission は行わず、後続で使うなら exp148 系の selector confidence / segment stability feature surface に限定する。
