# 設計

## 仮説

exp490の絶対予測ではなく、exp357からexp490へのmean-reversion correctionだけが
現在のfinal anchorに対して相補的である。単一の小さなfold-safe係数でも改善が安定しないなら、
exp490の大きな親比改善を最終アンサンブルへ移植できるという仮説を棄却する。

## アプローチ

保存済みOOFを同一`well / row_idx`へ整列し、exp490がexp357へ加えた補正だけを抽出する。

```text
d_i = p_exp490_i - p_exp357_i
r_anchor_i = p_anchor_i - y_i
lambda_train = clip(-sum(r_anchor_i * d_i) / sum(d_i ** 2), 0.00, 0.10)
p_primary_i = p_anchor_i + lambda_train * d_i
```

各held meta foldについて、`lambda_train`は他4 outer foldsだけからclosed formで求める。
held foldのtruth/errorは係数fitへ入れない。intercept、row weight、well weightは使わず、
competition metricと同じsuffix-row unweighted SSEを最小化する。

通常の予測blendは次式で同じmeta-fold手順を実行するが、report-only controlとする。

```text
c_i = p_exp490_i - p_anchor_i
w_train = clip(-sum(r_anchor_i * c_i) / sum(c_i ** 2), 0.00, 0.10)
p_convex_control_i = p_anchor_i + w_train * c_i
```

controlの勝敗でprimaryを救済せず、controlを推論候補にしない。primaryは
`exp490 - exp357`のincremental valueだけを検証する。

## anchor resolution

exp506実装前にexp497 Stage Eの記録を読み、次の一意な規則でanchorをfreezeする。

1. exp497が自身の事前登録all-AND gateをPASSし、selected predictionがexp413以外なら、exp497の保存selected OOFをanchorにする。
2. exp497がFAIL、technical stop、またはselected predictionがexp413 fallbackなら、exp413 Stage D保存OOFをanchorにする。
3. anchor名、OOF SHA、CV、fold manifest SHAをtruth join前にmanifestへ保存する。
4. exp506の結果を見た後のanchor変更、両anchorの良い方選択、3-way stackを禁止する。

2026-08-04、exp497 Stage E version 1は`completed_gate_failed_closed`で終端した。
candidate CV`7.87448814999802`はexp413比`0.010314644 ft`改善したが事前gateに未達し、
selected predictionは`exp413_oof`となった。したがってexp506 anchorはexp413 Stage D保存OOF、
prediction列`scale5_x1p0_full_replacement__lgb_mean__pred_tvt`、CV`7.884802794404715`、
file SHA`9bd2d177...cef4a9d`へ固定する。

## 実験範囲

- 対象実験: `exp506_exp490_mean_reversion_correction_blend_on_exp413`
- Route: `ensemble`
- root parent: `exp413_scale5_likpf_full_replacement_on_exp335`
- upstream dependency: `exp497_strict_public_core_fold_safe_ensemble_on_exp413`
- correction source: `exp490_geometry_centered_mean_reverting_offset_hmm`
- correction parent: `exp357_exp226_huber_emission_independent_audit`
- negative references: exp494、exp499、exp501、exp502、exp500、exp503、exp505
- 変更する変数: 最終anchorへ加える単一のmean-reversion correction係数だけ
- 固定する変数: source predictions、outer folds、row role、scope、metric、weight bound、式、gate

## 入力契約

### exp413 root anchor

- source kernel: `kentookumura/exp413-scale5-likpf-downstream-train` version 2
- OOF: `stage_d_oof_predictions.parquet`
- expected CV: `7.884802794404715`
- expected OOF SHA256: `9bd2d17778b3b27d771b12cbff72def8b87e6cdf14062e1c0ba192434cef4a9d`
- fold manifest SHA256: `fa41084c5fcb4adffb88d44211b4cc5d2d2f46b5bd4d65828b6af941184b2a6d`
- fold metrics SHA256: `82e70b6674f218f2892d6e5f70e327dfcbbdaf0fa5e431c4e07231009e9e2d8f`
- scope metrics SHA256: `c89add97cd4cae628b79774615a717e4cfbffe7b65a4a68c58b2c2e2737948ed`
- hidden-like metrics SHA256: `eafa3546e4ea5c0d180d380f7fe2c39b5cac970ea4c8097b68b077017da1f1b8`
- by-well SHA256: `e82c6908ed2caa9b3e5c1664bc66a3226b3bc6d9284f4863bd4fa941ae32d080`

### exp497 conditional anchor

- Stage E terminal resultが実装前prerequisite。
- PASS時だけselected OOF file、prediction column、OOF SHA、CV、5 meta weightsを追加固定する。
- unresolved、partial folds、Stage M componentだけをanchorにすることは禁止する。

### exp490 correction source

- source kernel: `kentookumura/exp490-mean-revert-full-merge` version 1
- file: `exp490_geometry_centered_mean_reverting_offset_hmm_stage1_full_oof_predictions.csv.gz`
- rows / wells: `3,783,989 / 773`
- raw gzip SHA256: `99030b33d493cc5f195f7d1a867f0d812a539143da9e1f59277e53779261b72c`
- decompressed SHA256: `e020e82e748a7836085657c4058070ff7853ed285639f2c2555cab721f9e9a07`
- prediction column: `geometry_mean_reverting_hmm`
- parent column: `exp357_parent_prediction`
- exact pre-freeze allowlist: `well`、`row_idx`、`suffix_offset`、`md_since`、上記2 prediction列
- exp490 source foldはsplitやfeatureとして使わず、exp413 fold manifestへ再partitionする。

### assignment

- hidden-like assignment SHA256: `5f9ac9fac6bb3725a7c613f09856a85bdf73b8206fd2edf1b79e8eaa9bca6597`
- join key: `well / row_idx`
- `suffix_offset`、row count、well count、fold、score-row roleを完全一致させる。
- duplicate、missing、extra key、NaN、Infを0とする。

## phase separation / leakage guard

1. anchor OOFとexp490 allowlist列だけを読み、key/fold/row-roleを検証する。
2. `d = exp490 - exp357`、anchor、component SHAをtruthなしでfreezeする。
3. freeze後だけsuffix truthと固定scope assignmentを接続する。
4. 各meta foldで他4 foldsだけをweight fitへ渡す。
5. held predictionを全件freezeし、5 folds結合後にprimary metricsを計算する。
6. primary decision freeze後だけreport-only convex controlと相関・covariance診断を表示する。

controlを先に見てprimary式、weight bound、gateを変えることは禁止する。

## 評価と判定

Primary:

- selected-anchor OOF RMSE
- correction meta-fold OOF RMSEとgain
- fold 0--4 RMSE / delta
- MD 0--250、250--1000、1000+ RMSE / delta
- hidden-like spatial / typewell-purged RMSE / delta
- by-well delta RMSE median / p90 / p95 / p99 / worst
- +0.25 / +1 / +3 / +5 ft悪化well数
- meta-fold lambda 5個、range、bound hit

Diagnostics:

- `r_anchor`と`r_exp490`のpooled/fold residual Pearson相関
- `r_anchor`とcorrection `d`のdot product / cosine / covariance
- correction normとanchor residual norm
- report-only convex blendのweightとcross-fitted RMSE
- exp490 standalone、exp357 parent、selected anchorの参照RMSE

Primary all-AND gateはrequirements.mdの条件をそのまま使う。gate判定後のweight、式、
component、scope、tail上限変更を許可しない。

## 将来の実装・実行量契約

Stage A OOF audit:

| 項目 | 実行量 |
| --- | ---: |
| scientific primary variant | 1 |
| report-only control | 1 |
| outer / meta folds | 5 / 5 |
| learned model / booster | 0 / 0 |
| HMM / PF / Beam run | 0 / 0 / 0 |
| exp413 / exp497 / exp357 / exp490 retraining | 0 |
| GPU | 0 |

Stage A実装とKaggle CPU runは別承認とする。PASS後のhidden-dynamic inference統合も
別承認とし、exp413/exp497 anchor inferenceとexp490 version 2 HMM inferenceを同一
Notebook内で再生成する。public submission CSV同士の静的平均ではhidden testへ対応しない。

## 再現性設計

- seed policy: no RNG。closed-form scalar fit、固定fold、stable key orderのみ。
- stochastic処理: Stage Aにはなし。保存sourceが持つ過去のGPU/PF/HMM再現性は各source manifestで固定する。
- PF/Beam / likelihood-PF / seed bagging: 再実行0。保存prediction SHAだけを使用する。
- 並列処理と乱数: RNGなし。reduce順序を固定し、float64 accumulationを使う。
- runtime: Stage AはKaggle private CPU、internet offを想定する。
- input gzip: raw gzip SHAとdecompressed content SHAを分け、後者を主証拠とする。
- record: anchor resolution manifest、input SHA、key/fold manifest SHA、correction SHA、primary OOF SHA、weight SHA、metrics/gate SHA。
- deterministic anchor: Stage A OOFが再実行一致するまではfalse。
- inferenceへ進む場合: current-test prediction content SHA、submission SHA、kernel version、hidden-dynamic inventoryを追加記録する。
- Kaggle package: metadataとbootstrap ZIP内config/input contractの一致をpush前に検証する。

## リスク

- リークリスク: same-OOF global weight、held fold truth、exp490 outcome列、exp497結果後のanchor選び直し。
- CV/LB不一致: exp490はCV 8.480155に対してPublic LB 9.680、exp494はCV改善後にLB悪化した。平均CVだけで提出しない。
- tailリスク: exp490はby-well p95 `+7.257814 ft`、worst `+49.602560 ft`。小weightでもtail gateを必須にする。
- component重複: exp413は既に複数physics面を持つため、correction covarianceがfold間で不安定なら即閉じる。
- upstream依存: exp497 Stage Eが未完了。partial結果を使わない。
- inferenceリスク: code competitionでは両pipelineをhidden test上で動的生成する必要がある。Stage A PASS前に実装しない。
- 再現性リスク: 保存source SHA不一致、gzip metadata差、join order差。decompressed SHAとlogical key SHAでfail closedする。

## 終端結果

Kaggle private CPU version 2（id_no `129631767`）で3,783,989 rows / 773 wellsを完了した。
technical契約は全PASSしたが、primaryはanchor比`+0.017265667715181 ft`悪化、lambdaは
`[0, 0.041578388, 0, 0.004513714, 0]`、nonworse`3/5 folds`、固定scope`0/5`、
worst-well`+1.816049513 ft`となった。事前all-AND gateに従って仮説を棄却し、
weight / component / scope / gate rescue、inference、submissionを行わない。
