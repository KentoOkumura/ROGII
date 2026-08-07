# 設計

## アプローチ

exp148 `lgb_mean` の保存済み OOF prediction を base とし、exp072 の `tvt_dense` / `tvt_densew` / `tvt_dense50` 候補を target-free な well/segment-level verifier で低頻度に採用できるかを診断する。exp135 の gate audit を土台にするが、単純 high-drift / high-disagreement gate の焼き直しではなく、near guard と candidate path continuity guard を追加する。

## 実験範囲

- 対象実験: `exp154_segment_level_dense_candidate_verifier_on_exp148`
- Route: `ml_model`
- 親実験: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- 比較基準: exp148 `lgb_mean`、exp092 `lgb1`、exp073 `lgb_mean`
- 変更する変数: segment verifier 条件、candidate、alpha、clip、minimum segment length、near guard、candidate path continuity guard
- 固定する変数: exp148 OOF、exp073 OOF、exp072 PF/Beam/dense cache、score rows

## 再現性設計

- seed policy: 新規乱数なし。保存済み OOF / cache を読むだけ。
- stochastic 処理の有無: exp154 内にはなし。upstream exp148 / exp073 / exp072 の stochastic component は source meta として記録する。
- PF/Beam / likelihood-PF / seed bagging の有無: 再生成せず、exp072 cache の保存済み列を使用する。
- 並列処理と乱数の関係: 乱数なし、single process audit。
- CPU/GPU runtime と deterministic flags: Kaggle CPU audit。GPU 不要。
- train cache / test feature regeneration の SHA 記録方針: input file SHA と gzip decompressed SHA を summary に記録する。
- model manifest / prediction / submission SHA 記録方針: 新規 model / submission は作らない。posthoc prediction sample と summary は audit output として保存する。
- Kaggle package bootstrap 確認方針: `prepare-kaggle-notebooks --strict` と `validate-exp` を通してから push する。

## リスク

- リークリスク: verifier 条件に true TVT / true error / oracle best を混ぜると漏洩になる。コード上は scoring 後の readout に限定する。
- CV/LB 不一致リスク: posthoc OOF 改善が hidden test に転移しない可能性がある。global RMSE だけで採用せず、near、worst-well、raw-test parity を必須 guard にする。
- ランタイム/メモリリスク: 3.78M rows を複数 prediction 付きで読むため Kaggle CPU memory に注意する。LightGBM 学習はしない。
- 再現性リスク: exp148 output がローカル未取得の場合、Kaggle input mount 依存になる。実行時の source path / SHA を summary に残す。
