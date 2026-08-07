# 要件

## 依頼

`trajectory_local_typewell_self_gr_switch_audit` を実装する。KAGGLE_DIRECTION の backlog にある、1 trajectory 内で typewell match と horizontal self-match を局所切替する監査を exp128 として切り出す。

## 制約

- Route: `ensemble`
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- 親は exp099 PF/Beam / likelihood-PF candidate cache とし、exp091 / exp093 の self-GR 失敗を前提に保守的な監査にする。
- self-GR source は同じ horizontal well の visible prefix だけを使い、evaluation zone の true TVT は scoring 専用にする。
- typewell cost は候補 trajectory TVT と typewell GR から計算し、target TVT を switch 判断に使わない。
- この実験では inference port / submission は作らない。

## 受け入れ基準

- `experiments/exp128_trajectory_local_typewell_self_gr_switch_audit/` に config、train/inference notebook、補助 script、記録ファイルがある。
- train notebook は設定確認、入力確認、監査実行、生成物確認をセルで追える。
- 補助 script は candidate metrics、bucket metrics、by-well metrics、signal metrics、window diagnostics、OOF gzip、feature schema、summary JSON を書く。
- `config.yaml` に route、親実験、local switch threshold、出力生成物、再現性方針がある。
- 未実行状態は `implemented_not_run` として README / SESSION_NOTES / result / metrics に記録する。
- deterministic anchor として扱わない。採用候補になった場合だけ、別途 raw-test parity audit で feature content SHA、prediction SHA、Kaggle kernel version を記録する。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
