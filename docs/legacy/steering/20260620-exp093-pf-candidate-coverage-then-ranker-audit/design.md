# 設計

## アプローチ

exp072 の deterministic full replay train cache を固定入力にし、既存 PF/Beam/likelihood-PF 候補と exp091 由来の self-GR 候補を同じ long-format candidate table に展開する。真値は `last_known_tvt + target` として scoring にだけ使い、候補別 RMSE/within threshold、candidate set 別 oracle topK coverage、target-free rank score coverage、bucket 別 weak coverage を保存する。

ranker はこの実験では学習しない。`baseline_primary` と `baseline_plus_self_gr` の oracle coverage / headroom を比較し、次を summary JSON に recommendation として出す。

- coverage と oracle gain が十分: `pf_candidate_ranker_or_nway_classifier` へ進む。
- coverage はあるが target-free ranking が弱い: likelihood scorer / learned GR similarity の改善を検討する。
- coverage が低い bucket が多い: ranker ではなく候補生成側の失敗地図へ戻る。

## 実験範囲

- 対象実験: `exp093_pf_candidate_coverage_then_ranker_audit`
- Route: `pf_beam`
- 親実験: `exp091_self_gr_likelihood_pf_beam_probe`
- 変更する変数: candidate set 比較、bucket 別集計、ranker readiness 判定。
- 固定する変数: exp072 feature cache、PF/Beam/likelihood-PF 候補生成、self-GR candidate generation config、LightGBM/ML prediction、submission policy。

## 再現性設計

- seed policy: exp093 では新規乱数を使わない。upstream PF/Beam は exp072 cache の SHA を記録して固定入力として扱う。
- stochastic 処理の有無: exp093 内にはなし。exp072 upstream cache 由来の PF/Beam stochasticity は再生成せず、入力生成物として扱う。
- PF/Beam / likelihood-PF / seed bagging の有無: 新規実行なし。候補列を読み込んで audit するだけ。
- 並列処理と乱数の関係: self-GR 候補生成は逐次 well loop。global RNG は使わない。
- CPU/GPU runtime と deterministic flags: CPU train notebook audit。GPU 不要。
- train cache / test feature regeneration の SHA 記録方針: source cache の raw SHA と gzip decompressed content SHA、feature schema SHA を summary JSON に記録する。
- model manifest / prediction / submission SHA 記録方針: model、prediction、submission は作らない。deterministic submission anchor として扱わない。
- Kaggle package bootstrap 確認方針: `prepare-kaggle-notebooks --notebook train --strict` で bootstrap 内 support files を再生成してから push する。

## リスク

- リークリスク: true TVT を oracle scoring に使うため、出力は診断限定。candidate generation と target-free rank score には true TVT を入れない。
- CV/LB 不一致リスク: train-side pseudo-tail audit であり、LB 提出判断には直結させない。
- ランタイム/メモリリスク: full candidate long table は 3.8M rows x candidate count になり重い。Kaggle train notebook で実行し、local smoke は `max_rows` 指定時だけにする。
- 再現性リスク: upstream cache が Kaggle input に存在しないと実行できない。summary に source path と SHA を残す。
