# 設計

## アプローチ

exp221 の `run_hmm2(..., return_post=True)` を変更せずに呼び、各 well の marginal
posterior `(evaluation row, TVT grid)` を一時的に解析する。local maximum を fixed
prominence / fixed separation で検出し、top1 / top2 peak とその間の valley を定義する。
二峰性は、(1) 2 peak 以上、(2) second peak mass、(3) peak separation、(4) valley
depth のすべてを満たす target-free fixed rule とする。

dominant mode は top1 / top2 間の最深 valley で領域を分けたうえで、mass の大きい
側を採用する。隣接rowの dominant mode は fixed TVT jump allowance 内で同一trackに
対応付け、segment length と switch を数える。二峰かつ posterior mean が二peakの間で
valley density 以下となる行を `mean_in_valley` とする。

true TVT は HMM input、peak選択、二峰判定、mode tracking、代表plot選択には渡さず、最終的な
decoder metrics、error lift、oracle top2 coverage、および train-side diagnostic plot
overlay にのみ使う。

## 実験範囲

- 対象実験: `exp236_exact_hmm_posterior_bimodality_audit`
- Route: `ensemble`
- 親実験: `exp221_lgb_oof_gaussian_emission_hmm_on_exp148`
- 参照: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`、
  `exp133_gr_bimodal_match_ambiguity_detector`、必要なら完了後の `exp234`。
- 変更する変数: posterior の読み出しと診断集計だけ。
- 固定する変数: exp221 HMM / LGB emissionの全値、saved exp148 `lgb_mean` OOF、
  score row、distance bucket、hidden-like split。

固定する二峰判定値:

- grid step: parent HMM の `0.35` を読む（再設定しない）。
- local-peak prominence: `0.02` posterior mass。
- top2 peak mass: `0.10` 以上。
- top2/top1 mass ratio: `0.25` 以上。
- peak separation: `6.0 ft` 以上。
- valley depth: `0.30` 以上（`1 - valley_density / min(peak_density)`）。
- same mode track allowance: `6.0 ft`。
- representative plot: mean-in-valley row数、bimodal row数、well IDの固定順で最大 `12` wells。

これらは target / error / Public LB を見ずに一度だけ固定する。変更が必要なら、
同じ exp の config variant として明示して追加監査する。

## 再現性設計

- seed policy: HMM / audit に新規乱数なし。CPU `outer_workers=1`、
  `numba_num_threads=1` を固定する。
- stochastic 処理の有無: なし。upstream exp148 OOFは保存済み生成物としてSHAを記録する。
- PF/Beam / likelihood-PF / seed bagging の有無: exact HMMだけを再生し、PF/Beamを
  再実行しない。
- 並列処理と乱数の関係: parallel RNGなし。well処理順はソートし、plot対象の順位は
  fixed sort keyで決める。
- CPU/GPU runtime と deterministic flags: CPU-only、GPU / internet disabled、booster 0。
- train cache / test feature regeneration の SHA 記録方針: exp148 OOF gzipは
  decompressed content SHA、row / segment summaryとmetricsはfile SHAを記録する。
- model manifest / prediction / submission SHA 記録方針: 新規model / inference /
  submissionはない。HMM decode summaryと代表plotのSHAを記録する。
- Kaggle package bootstrap 確認方針: `prepare-kaggle-notebooks --strict`後にbootstrap
  内configとCPU metadataを確認する。

## リスク

- リークリスク: true TVT/error/oracleがpeak解析に混入すること。入力関数とmetric関数を
  分離し、row diagnostic作成後にのみtargetをmergeする。
- CV/LB 不一致リスク: exp221のCV改善がLBに転移していない。diagnostic結果をそのまま
  inference / submit 方針に使わない。
- ランタイム/メモリリスク: full posteriorを全wellで保存しない。wellごとの後に破棄し、
  plotも固定上限を持たせる。
- 再現性リスク: upstream OOF sourceやKaggle packageの差異。ID被覆、row order、
  decompressed SHA、HMM config snapshotを保存する。
