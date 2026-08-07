# 設計

## アプローチ

exp072 feature cache には exp073 anchor が使う PF/Beam/likelihood-PF 系候補と target-free confidence scalar が含まれている。この cache を TVT 空間に戻し、各 well の pseudo-hidden tail を `prefix_backtest_fraction <= 0.35` の calibration phase と、それ以降の holdout phase に分ける。

calibration phase の行だけで `abs(primary_candidate - true_tvt)` を説明する ridge confidence model を fit し、well-hash fold 外の held-out well 全体へ `expected_tvt_error` を出す。主評価は holdout phase で、confidence bin と distance bucket が高 TVT error を識別できるかを見る。

## 実験範囲

- 対象実験: `exp087_prefix_backtest_tvt_confidence`
- Route: `pf_beam`
- 親実験: `exp083_pf_beam_true_tvt_2d_well_eda`
- 入力 cache: `exp072_exp063_full_replay_feature_cache`
- Anchor: `exp073_gpu_reproducibility_guard_for_exp063_full_replay`
- 変更する変数: confidence feature からの expected TVT error 推定と bucket 診断
- 固定する変数: PF/Beam/likelihood-PF 候補生成、ML anchor、raw test inference、submission policy

## 再現性設計

- seed policy: 乱数は使わない。well-hash fold は `blake2b(well_id) % n_folds` で固定する。
- stochastic 処理の有無: なし。PF/Beam は既存 cache を読むだけで、新規生成しない。
- PF/Beam / likelihood-PF / seed bagging の有無: 既存生成物を入力として読む。再生成はしない。
- 並列処理と乱数の関係: 並列乱数なし。
- CPU/GPU runtime と deterministic flags: CPU only。
- train cache / test feature regeneration の SHA 記録方針: 入力 cache の raw SHA と gzip decompressed SHA を summary JSON / metrics.json に記録する。
- model manifest / prediction / submission SHA 記録方針: learned submission model ではないため model SHA と submission SHA は記録しない。row-level confidence prediction は CSV gzip として保存する。
- Kaggle package bootstrap 確認方針: `prepare_kaggle_notebooks --strict` で metadata と bootstrap の config を生成する。

## リスク

- リークリスク: calibration phase の true TVT を使うため、これは train-side 診断であり inference policy ではない。well-hash fold 外評価と holdout phase 評価で過信を抑える。
- CV/LB 不一致リスク: submission を出さないため LB 判定はしない。次実験へ渡すのは confidence feature 候補だけにする。
- ランタイム/メモリリスク: exp072 cache は大きい。row-level gzip 出力は必要列だけに絞る。
- 再現性リスク: source cache が Kaggle input mount に依存するため、input SHA を記録する。
- 近似リスク: この実装は短い synthetic cutoff で PF/Beam を再実行せず、既存 exp072 cache の行内 backtest として校正する。結果は direct candidate replacement ではなく confidence diagnostic として扱う。
