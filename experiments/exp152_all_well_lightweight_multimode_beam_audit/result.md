# exp152_all_well_lightweight_multimode_beam_audit 結果

## 状態

Kaggle train v1 完了。

- Kernel: `kentookumura/exp152-allwell-light-mmbeam-audit-train`
- URL: https://www.kaggle.com/code/kentookumura/exp152-allwell-light-mmbeam-audit-train
- Output: `experiments/exp152_all_well_lightweight_multimode_beam_audit/kaggle/output/train_v1`
- Scope: all train wells、tail 500 rows / well
- 評価 rows / wells: 386,407 rows / 773 wells
- Runtime: 780.239 sec
- Route: `pf_beam`

## 判定基準

主比較は `exp072_beam_mean`。全体 RMSE / MAE の改善に加え、well-level の improved/worsened 件数、最大悪化 well、tail bucket delta を確認する。

`exp072_likpf_mean` / `exp072_pf_z` を超えることはこの軽量監査の必須条件にしない。ただし ML replacement や full-row candidate cache へ進めるには、Beam 比改善が一部 well に偏らず、最大悪化が許容範囲に収まる必要がある。

## 実行結果

| candidate | RMSE | MAE | bias | within10 |
| --- | ---: | ---: | ---: | ---: |
| `exp072_likpf_mean` | 16.115835 | 10.421046 | -1.111434 | 0.646595 |
| `exp072_beam_mean` | 19.685742 | 14.052127 | -1.270644 | 0.478829 |
| `multimode_pf_zacc_s010_a020_noise050_best_lik_seed` | 20.110701 | 14.501435 | -1.662677 | 0.454707 |
| `exp072_pf_z` | 24.439192 | 16.117721 | -1.092848 | 0.488236 |

`exp143` の 6 well scoped audit では multimode PF-Z が `exp072_beam_mean` より良かったが、全 773 well の tail slice では主比較の Beam に届かなかった。

- multimode vs `exp072_beam_mean`: RMSE +0.424958、MAE +0.449309、within10 -0.024122
- multimode vs `exp072_likpf_mean`: RMSE +3.994865
- multimode vs `exp072_pf_z`: RMSE -4.328491
- well-level: improved 364 wells、worsened 409 wells
- mean / median well RMSE delta vs Beam: +0.392917 / +0.225596
- 最大悪化: `f140c2fa`、RMSE delta +15.334797
- 最大改善: `a959858c`、RMSE delta -29.117956

## Bucket / quality readout

`md_since` bucket では 0-50、50-100、100-250、500-1000 ft で Beam より良い一方、250-500 と 1000+ ft で悪化した。特に全体 rows が tail slice でも longtail 側に偏るため、global RMSE は Beam 比で悪化した。

multimode quality は平均 effective sample fraction 0.700324、平均 resample count 54.493208、平均 collapse rate 0.128466、平均 mode count 1.056388、mode_count<=1 rate 0.944352。軽量化しても多くの well で single mode に潰れており、exp143 の positive 結果は scoped wells への偏りが強かったと判断する。

strict PF-Z parity はこの実験の採用 guard ではないが、`max_abs_diff=121.738281`、`rmse_diff=20.554779` で fail。軽量 strict probe は parity anchor ではなく diagnostic としてのみ扱う。

## 結論

`all_well_lightweight_multimode_beam_audit` は完了、採用しない。multimode candidate は `exp072_pf_z` よりは良いが、主比較の `exp072_beam_mean` と採用 guard の `exp072_likpf_mean` に負け、worsened wells が improved wells を上回った。

この候補から full-row minimal candidate cache、ML replacement、inference port、submission へは進めない。PF/Beam mode diversity は引き続き direct replacement ではなく、exp092 系 ML confidence feature / segment-level verifier / learned likelihood / normalized shape feature の診断材料に限定する。
