# 設計

## アプローチ

exp101 の LightGBM booster を再学習せずにロードし、exp099 v2 wide cache から exp101 と同じ OOF fold / feature schema / imputation で score surface を復元する。

復元する score:

- `lgb_candidate_error_ranker`: 候補別 predicted absolute error、top1 / top2 margin。
- `lgb_candidate_binary`: 候補別 oracle probability、top1 / top2 margin。
- `lgb_multiclass`: 補助として class probability margin。

評価 variant は `likpf_mean` default を固定し、`lgb_candidate_error_ranker` が `pf_ancc` または `beam_mean` を選んだ行のうち、error margin、binary margin、predicted error、candidate disagreement、`pf_ancc_std`、switch-rate cap を満たす行だけ切り替える。grid は小さくし、OOF への過適合を抑える。

## 実験範囲

- 対象実験: `exp102_confidence_gated_likpf_fallback_on_exp101`
- Route: `pf_beam`
- 親実験: `exp101_pf_candidate_ranker_or_nway_classifier`
- 変更する変数: high-confidence gate の閾値、switch-rate cap、切替候補。
- 固定する変数: exp099 v2 feature cache、exp101 candidate set、exp101 GroupKFold split、exp101 booster、default `likpf_mean`。

## 再現性設計

- seed policy: exp101 と同じ `validation.seed=42`。candidate-long train median 復元のため、exp101 と同じ local RNG seed で sampled train rows を再現する。
- stochastic 処理の有無: 追加学習はなく、乱数は exp101 の sampled long-frame imputation median を再現するためだけに使う。
- PF/Beam / likelihood-PF / seed bagging の有無: 生成済み exp099 v2 cache を入力として読む。PF/Beam/likelihood-PF を再生成しない。
- 並列処理と乱数の関係: 追加並列 RNG は使わない。
- CPU/GPU runtime と deterministic flags: LightGBM booster inference のみ。CPU / GPU 学習はしない。
- train cache / test feature regeneration の SHA 記録方針: exp099 cache raw SHA / decompressed SHA、feature schema SHA を summary に記録する。
- model manifest / prediction / submission SHA 記録方針: exp101 manifest SHA、各 booster SHA、gated OOF prediction gzip raw / decompressed SHA を記録する。submission は作らない。
- Kaggle package bootstrap 確認方針: push 前に `prepare-kaggle-notebooks --notebook train --run-on-push --strict` で notebook package を作り、必要なら bootstrap 内 config を確認する。

## リスク

- リークリスク: label は評価にのみ使い、gate 条件には exp101 OOF score / target-free features だけを使う。fold は GroupKFold by well を再現する。
- CV/LB 不一致リスク: posthoc OOF gate は閾値探索で過適合しやすい。改善しても inference port せず、診断材料として扱う。
- ランタイム/メモリリスク: 3.78M rows x 5 candidates の score surface を復元するため CPU/メモリ負荷がある。候補 grid は小さくする。
- 再現性リスク: exp101 / exp099 の Kaggle output mount 名が変わる可能性があるため、local path と `/kaggle/input/**` glob の両方で artifact を探す。
