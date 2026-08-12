# 設計

## アプローチ

exp099 の hand-crafted multi-observation likelihood は oracle headroom を増やしたが、top1 scorer としては崩壊した。exp101 の supervised row-wise selector も `likpf_mean` 単体を超えなかった。したがって今回は candidate index を直接選ばず、各 candidate TVT が観測 GR / trajectory context と整合する確率を calibrated likelihood として学習する。

実装は candidate-long 形式にする。1 row x 5 candidates を作り、候補ごとの `multiobs_score` / `multiobs_mae` / `multiobs_ncc`、row 内 rank / gap、候補間 disagreement、prefix / distance context を特徴量にする。label は `within_10ft` binary と `abs_error` で、LightGBM classifier / L1 regressor を 1 fold smoke で学習する。

## 実験範囲

- 対象実験: `exp111_learned_pf_observation_likelihood_probe`
- Route: `pf_beam`
- 親実験: `exp099_pf_multi_observation_likelihood_probe`
- 変更する変数: target-free likelihood features を candidate-long に再構成し、learned within10 probability / expected abs error を出す。
- 固定する変数: exp099 v2 cache、候補集合、train pseudo-tail rows、GroupKFold by well、提出なし。

## 再現性設計

- seed policy: GroupKFold と LightGBM `random_state` を `validation.seed` から固定。candidate-long train row subsample は fold ごとの local `np.random.default_rng(seed + 101 * fold)`。
- stochastic 処理の有無: upstream PF/Beam cache と LightGBM histogram training と train row subsample が stochastic。
- PF/Beam / likelihood-PF / seed bagging の有無: この実験内では PF/Beam を再生成しない。exp099 v2 cache を固定入力にする。
- 並列処理と乱数の関係: feature generation は並列 RNG なし。LightGBM は `n_jobs=-1` のため bitwise deterministic anchor とは扱わない。
- CPU/GPU runtime と deterministic flags: Kaggle CPU、GPU false、internet false。
- train cache / test feature regeneration の SHA 記録方針: exp099 input raw SHA / decompressed SHA / schema SHA を summary に記録する。
- model manifest / prediction / submission SHA 記録方針: model manifest、OOF likelihood gzip raw/decompressed SHA、probability content SHA を記録する。submission は作らない。
- Kaggle package bootstrap 確認方針: `prepare-kaggle-notebooks --strict` と `validate-exp` を通してから push する。

## リスク

- リークリスク: true TVT を label に使うため train-side OOF 診断限定。valid well の label は fold 学習に入れない。
- CV/LB 不一致リスク: 提出候補ではなく likelihood smoke のため LB は見ない。改善した場合も raw-test parity と PF-weight ablation が必要。
- ランタイム/メモリリスク: full candidate-long は重いので `run_folds=1`、`max_train_rows_per_fold=350000` に制限する。
- 再現性リスク: upstream cache は stochastic PF/Beam 由来であり、この実験は deterministic submission anchor ではない。
