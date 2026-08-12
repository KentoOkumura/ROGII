# 設計

## アプローチ

exp209 exact HMMの`prefix_stats(..., tail_n=30)`をcontrolとする。新規candidateではcalibrationと
GR sigmaをwellごとに一度だけ計算し、known prefix末尾`32/64/128/256`行の
`median((delta TVT_input + delta Z) / delta MD)`だけを差し替えて、同じHMM kernelを4回decodeする。
prefixがwindowより短い場合は利用可能なknown rowsだけを使い、valid stepが3未満ならrate 0へ
fail-safeする。候補名とwindowは実行前に固定する。

全773 wellsで4 HMMを1 notebookに載せるとexp209実測から12時間超の懸念があるため、wellを
`sha256("exp268::well_shard::<well>") % 2`で2 shardに分ける。`train_variant0/1`は各wellの4候補を
同時生成し、正規`train` notebookは両shard、exp209 HMM control、exp072 referenceを統合する。
shardは科学variantではなく実行分割であり、candidate contractは両shardで同一とする。

oracleはexp243と同じ定義を使う。rowは各row最小絶対誤差、blockはwell内row順の固定
128/256/512行ごとにblock RMSE最小candidate、whole-wellはwell全体RMSE最小candidateを選ぶ。
oracle predictionは診断表の作成中だけ保持し、candidate cache、inference、submissionには書かない。

## 実験範囲

- 対象実験: `exp268_multi_scale_initial_rate_candidates`
- Route: `pf_beam`
- 親実験: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- 参照: `exp072_exp063_full_replay_feature_cache`、`exp242_two_regime_rate_noise_pf`、
  `exp243_pf_seed_medoids`、`exp115_hidden_like_spatial_holdout_from_ppt`
- 変更する変数: prefix初期rateのwindow `32/64/128/256`のみ。
- 固定する変数: HMM grid、41 rates、rate span、position/rate transition、momentum、GR emission、
  sigma mode、calibration、start position/rate prior width、band、score rows。
- 比較candidate: exp209 `tail_n=30` HMM、4 multi-scale HMM。exp072 `likpf_mean`は外部reference。
- 禁止: candidate平均/blend、oracle deploy、selector/weight学習、追加window/rate estimator、
  dynamic regime/process noise、raw-test inference、submission。

## 再現性設計

- seed policy: HMMはno RNG。well shardだけをstable SHA256で決める。
- stochastic 処理の有無: なし。Numba parallel floating arithmeticの微小差はあり得る。
- PF/Beam / likelihood-PF / seed bagging の有無: exact HMM 4候補。likelihood-PFは保存済み
  exp072 referenceを読むだけで再生成しない。
- 並列処理と乱数の関係: joblib thread 2 × Numba thread 2。乱数を使わないためscheduleで乱数系列は変わらない。
- CPU/GPU runtime と deterministic flags: Kaggle CPU、GPU false、internet false。shard 2本。
- train cache / test feature regeneration の SHA 記録方針: exp072/exp209入力のdecompressed SHAをhard guardし、
  shard gzipのraw/decompressed SHA、schema SHA、aggregate candidate array content SHAを記録する。testは生成しない。
- model manifest / prediction / submission SHA 記録方針: model/submissionは対象外。train-side candidate
  array content SHAだけをprediction証拠として記録する。
- Kaggle package bootstrap 確認方針: `train_variant0/1`はkernel sourceなし、正規`train`はexp072、exp209、
  exp115、2 shard kernel sourceを持つ。prepare後にmetadataとbootstrap configのSHAを照合する。

## リスク

- リークリスク: raw train true TVTはcandidate path固定後のcache targetとaggregate診断にのみ使う。
  rate、shard、HMM emission/transition、candidate名、windowはknown prefixだけで決まる。
- CV/LB 不一致リスク: official evaluation tail形状のtrain-side candidate-headroom auditであり、
  raw test適用性やPublic LBを直接主張しない。
- ランタイム/メモリリスク: 4 variants × 773 wellsはexp209 1 variantの約4倍CPU。2 shardへ分け、
  各shardは約6.3時間見込み。well単位で4候補を縮約して保存し、posterior tensorは保持しない。
- 再現性リスク: no RNGだがNumba parallel reductionとgzip metadataでbyte差が起こり得るため、
  deterministic anchorとせずdecompressed content SHAとmetric toleranceを主証拠にする。
