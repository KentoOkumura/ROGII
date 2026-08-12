# 設計

## 結論

exp412 と同じ unchanged exp209 first-pass HMM から persistent beta-filter rate-gap schedule を truth-free で作り、最初の inactive→active row だけを event にする。beta の符号は捨て、PF transition proposal に base / `-datum` / `+datum` を `0.80 / 0.10 / 0.10` で一度だけ混ぜる。importance ratio `p0/q` により元の exp404-compatible PF target を保つ。

これは exp412 の directional treatment の救済ではない。exp412 は backward-cause SSEを6.96%悪化させ terminal FAIL であり、その判定を変えない。exp412 から使うのは事前固定済みの target-free event時刻だけである。

## 系譜

- PF 親: `exp404_scale5_likpf_selector`
- particle support原因証拠: `exp410_likpf_particle_resampling_basin_audit`
- trigger定義とnegative result: `exp412_beta_filter_rate_disagreement_two_pass_reset`
- 対称HMM案の比較: `exp425_symmetric_rate_gap_hmm_branching`（未実行の別機構）
- route: `pf_beam`
- scientific factor: event時に一度だけ使う defensive position proposal

## trigger

first pass は exp209 exact HMM を変更せず、suffix rowごとの filtered rate mean/std、smoothed rate mean、filtered position std を保存する。

```text
z_beta_t =
    (mu_smoothed_rate_t - mu_filtered_rate_t) /
    max(sigma_filtered_rate_t, 0.005)
```

直近 inclusive 16 rowsで

- `abs(z_beta) >= 2.0` が8 row以上
- qualifying rowの sign の75%以上が同じ

のとき active とする。tie は inactive。ここまでは exp412 と同一である。

本実験では sign を保存台帳には残しても treatmentへ渡さない。各 well の最初の `inactive -> active` rowだけを `event_row` とし、PF proposalを変更する最初の transition は `event_row` へ入る transition とする。eventがなければ treatment PF は parent と bitwise identicalでなければならない。

future beta は future GR を使うが future TVT は使わない。code competition上のoffline suffix readoutとしてのみ利用し、cause/truth/error は trigger/prediction freeze 後に late joinする。

## defensive proposal

event rowで、各 ancestor particleの元 transition densityを `p0(x'|x)` とする。position centerだけを `±d_w` ずらし、同じ covarianceとrate dynamicsを持つ密度を `p-(x'|x)`、`p+(x'|x)` とする。

```text
d_w = max(filtered_hmm_position_std_at_event, 0.35 ft)
q(x'|x) = 0.80*p0(x'|x) + 0.10*p-(x'|x) + 0.10*p+(x'|x)
```

proposal componentを `C ∈ {base, minus, plus}` から固定確率で選び、元 transition innovationとcommon random numberを使ってposition sampleを生成する。rate sample、momentum、clamp、GR emission、ESS resamplingは parent のまま。

emission前の particle log weight に次を加える。

```text
log_importance = log p0(x'|x) - log q(x'|x)
```

`q >= 0.8 p0` なので、有限な点では `p0/q <= 1.25`。importance ratioのclip、renormalized target、branch別温度は導入しない。mixture densityは log-sum-exp で全三成分を評価し、sampled componentだけの密度で割らない。

### 数値実装

`datum >= 0.35 ft`に対してparent position noiseは`0.005 ft`であり、shifted
component上の`p0/q`をfloat64の通常比として実体化すると0へunderflowし得る。
そのためweight更新は`log p0 - log q`をfiniteなまま直接log-weightへ加え、
log-sum-expで正規化する。ratioのfloor/clipは行わない。positivityはfinite
log-ratio、上限は`log(p0/q) <= log(1.25)`で監査する。

event後は通常の parent PFへ戻る。branch labelは particle ancestry監査にだけ残し、branch固有state、持続的datum offset、追加noiseを残さない。

## RNG契約

- base propagation RNG: baseline/treatmentで共通
- component draw RNG: immutable `(experiment, well_id, seed_index, event_row, particle_index)` 由来の独立 stream
- resampling/roughening RNG: parentと同じkey。event後はweight/ancestry差で結果は分岐してよいが、乱数消費順は固定
- global RNG、shard番号、再開順をseed keyに含めない

component drawをbase propagation streamへ混ぜて baseline/treatmentのcommon-random比較を壊すことは禁止する。

## prediction readout

- particles: 500
- seeds: 128
- scale: x1.0
- PF filtering likelihood: parent Gaussian
- seed aggregation: exp404/exp417 Gaussian full-suffix evidence、centered softmax、temperature 5.0

Huber、affine/AR(1)、self-GRによる再採点は行わない。

## Stage 0: fixed32 mechanism preflight

### sample

exp412 の固定 manifestをそのまま再利用する。

- backward-cause 8
- forward-cause 8
- matched nonpersistent control 16
- total 32、重複0、5 folds
- manifest SHA: `1edb1e1481af84af4e8178fb6e0743fa40315eab0b7441eeff9232b571f93c30`

cause membershipは sample選択とlate readoutだけに使い、trigger/PFへ渡さない。

### 実行量

- unchanged HMM trigger generation: 32 well-runs
- baseline PF: 32 well-runs
- treatment PF: 32 well-runs
- total PF: 64 well-runs
- seed-well trajectories: 8,192
- particle starts: 4,096,000
- PF variants: baseline 1 + treatment 1
- LightGBM config / trained fold / booster: 0 / 0 / 0
- Beam / GPU: 0 / 0

親PF controlを32 wellsで再実行するため、実装後も Kaggle Stage 0 push は別の明示承認が必要。

### technical AND gate

1. fixed32 manifest、exp209 first-pass parity、trigger schedule SHA が固定値と一致する。
2. suffix truth/error/cause の trigger、proposal、prediction freeze前readが0。
3. eventは各 well 0または1回で、最初の false→true rowと一致する。
4. no-event wellのbaseline/treatment particle/prediction SHAがbitwise一致する。
5. `q`、`p0/q`、weight、predictionは全てfiniteで、`0 < p0/q <= 1.25 + 1e-12`。
6. synthetic quadrature/Monte Carlo contractで importance-corrected proposal が `p0` の既知momentを許容誤差内で再現する。
7. base/minus/plus mass contractが `0.80/0.10/0.10`、minus/plus shiftが厳密に対称。
8. baseline/treatmentのbase innovationがeventまで一致し、component RNGがbase RNGを消費しない。
9. baseline predictionが同一設定の保存 parent と max abs `<=1e-5 ft`。
10. Stage 0の実測から fullを trigger-cache + 4 PF shardへ分けたruntime/RSSが実行制限内と見積もれる。

### mechanism AND gate

truthとexp410 support ledgerは全artifact freeze後にだけ joinする。評価窓は `event_row` から `min(event_row+511, suffix_end)`。

- triggered wells: `>=8/32`
- majority-seed truth-outside-particle-support row fraction: baseline比 `>=0.05` absolute reduction
- triggered evaluation-window SSE: baseline比 `>=10%` reduction
- evaluation-window SSEが非劣化する reporting fold: `>=4/5`
- matched nonpersistent control pooled RMSE delta: `<=+0.02 ft`
- matched nonpersistent control worst per-well RMSE delta: `<=+0.25 ft`

どれか一つでもFAILなら proposal mass、datum floor、event回数、trigger threshold/windowを同じfixed32で救済せず branchを閉じる。Stage 0は mechanism sampleであり prediction CVではない。

## full OOF

Stage 0の全gate PASS、exp410 baseline support artifact parity、別のユーザー承認後だけ実行資格を得る。

### 実行量

- HMM trigger cache: 773 well-runs
- treatment PF: 773 well-runs
- treatment seed-well trajectories: 98,944
- treatment particle starts: 49,472,000
- CPU PF shards: 4
- 保存済み親 PF control の独立 full rerun: 0
- LightGBM config / fold / booster: 0 / 0 / 0
- Beam / GPU: 0 / 0

HMM trigger cacheとPF shardsは別artifactに分ける。single-kernel 8.5時間内へ無理に詰め込まない。各実行の承認とruntime projectionを push前に SESSION_NOTESへ再記録する。

### promotion AND gate

- saved exp404 T=5比 overall RMSE gain `>=0.10`
- 改善または非劣化 reporting fold `>=4/5`
- deep/shallow、missingness、roughness scopeは全て非劣化
- paired per-well squared-error delta p95 `<=0`
- worst paired per-well RMSE delta `<=0.25 ft`
- frozen support-audit対象で truth-outside-support row fraction `>=0.05` absolute reduction
- triggered 512-row window SSE `>=10%` reduction
- no-event well pooled predictionは parentと numerical parity

全PASSでも inference/submissionは別承認とする。

## 再現性

- HMM triggerは RNGなし、well/row順固定
- PFは immutable well × seed index、componentは well × seed × event × particle のstable SHA-256
- trigger、event ledger、trajectory、branch ancestry、importance ratio、prediction、metricsのcontent SHAを保存
- gzipは decompressed content SHAを主証拠とする
- baseline/treatmentのcommon random ledgerを保存
- Kaggle kernel id/version、package、CPU/Numba、input SHA、runtime/RSSを保存
- stochastic PFのcross-rerun parityまで deterministic anchor としない

## 判断済みの分岐

- beta方向にだけ枝を出す: exp412 の失敗を再利用するため不採用
- ±datumをtarget transitionとして扱う: modelを変え、defensive proposalでなくなるため不採用
- importance correctionを省略/clip: targetを変えるため不採用
- 毎active rowで再注入: intervention量とancestryが不明確になるため不採用
- global roughening/process noise増加: exp410でgain集中とcatastrophic riskがあるため不採用
- datumやmixture massのgrid: same-sample救済になるため不採用

## 実装開始条件

2026-07-28のユーザー指示`exp432を実装してください`により、Stage 0のcompact
self-contained実装とcontract test開始条件は満たした。親control 32 PF well-runsを
含む正規Notebook採用、Kaggle package/push/runは、実行量を再提示してさらに
明示承認を得る。

## Stage 0実行開始条件

2026-07-29のユーザー指示`実行してください`により、親control 32 PF well-runsを
含む正規train Notebook採用とfixed32 Kaggle package/push/runを承認済みとする。
full、inference、submissionは引き続き別承認とする。

## Stage 0実行結果

2026-07-29にKaggle private CPU version 1（id_no `128974856`）を完了した。
technical contractはruntime projection以外PASSしたが、mechanism gateはsupport外率
`-0.004278`、nonworse folds `3/5`、worst control `+0.583073 ft`でFAILした。
SSE reduction `0.120871`だけではAND gateを満たさず、full eligibleはfalse。

runtime projection実装は4 PF shardsを反映せず`91,746.964秒`を記録した。
設計どおり単純4分割した保守的参考値は`22,936.741秒`だが、mechanism FAILが
独立に確定しているため再実行やgate救済は行わない。
