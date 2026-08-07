# 要件

## 依頼

`exact_replacement_prune_on_exp148` を `exp198_exact_replacement_prune_on_exp148` として実装する。
親は ML route submitted anchor の `exp148_learned_likelihood_fulltrain_addonly_on_exp092` とし、後追いで追加されたが既存列と完全一致・符号反転・定数になっている 17 列だけを active model feature から削る。

## 制約

- Route: `ml_model`
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- 新規特徴量は足さない。
- drop-only の初回評価では候補を 17 列から広げない。
- exp148 control / parent は再学習しない。保存済み exp148 CV `lgb_mean=8.50128118189582` / Public LB `7.960` を historical baseline とする。
- Kaggle train push 前に active variant 1、LightGBM config 3、fold 5、合計 booster 15、control 再学習なしを `SESSION_NOTES.md` に記録する。
- direct replacement / blend / postprocess / submit は、この train-side drop-only 実装の範囲外にする。

削除対象の 17 列:

```text
sc_trust
ll_candidate_tvt_likpf_mean_minus_likpf_mean_tvt
ll_candidate_tvt_beam_mean_minus_last_known_tvt
ll_candidate_tvt_beam_mean_minus_likpf_mean_tvt
ll_candidate_tvt_hyb_minus_last_known_tvt
ll_candidate_tvt_likpf_mean_minus_last_known_tvt
ll_candidate_tvt_pf_ancc_minus_last_known_tvt
ll_candidate_tvt_pf_ancc_minus_likpf_mean_tvt
ll_candidate_tvt_sc_ens_minus_last_known_tvt
tda0
dense_bias
uproj_beam_mean_resid
uproj_beam_med_resid
uproj_diff_pf_ancc_minus_pf_z
uproj_likpf_mean_resid
uproj_pf_ancc_resid
uproj_pf_z_resid
```

## 受け入れ基準

- `experiments/exp198_exact_replacement_prune_on_exp148/` に config、train/inference notebook、記録ファイルがあり、route / lineage / active variant / drop list が明記されている。
- train 実装は exp148 の feature assembly を保ち、active feature list から 17 列だけを除外する。
- feature count は exp148 の 294 から 17 減った 277 を期待値として確認できる。
- train notebook は Kaggle 実行用の Jupytext percent `.py` から生成され、構文チェック、`jupytext --to ipynb --test`、`ruff --select F821`、`validate-exp` 相当が通る。
- `SESSION_NOTES.md` に GPU 学習コストガード、親 control 再学習なし、比較基準、Kaggle push 前チェック結果が記録されている。
- deterministic anchor として扱う場合は、feature content SHA、model SHA、prediction SHA、submission SHA、Kaggle kernel version が記録されている。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
