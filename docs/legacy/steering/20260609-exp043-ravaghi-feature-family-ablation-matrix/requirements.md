# 要件

## 依頼

`ravaghi_feature_family_ablation_matrix` を実装する。

## 制約

- Route: `ml_model`
- Kaggle Notebook train 実行を正とする。
- 入力は `exp029_public_sel15_pf_oof_feature_generation` の train well の途中以降を隠した疑似 test rows。
- direct PF/beam replacement、Ridge/meta-stack、exp026 OOF bridge columns は model feature にしない。
- `ANCC` など train-only formation columns は直接読まない。
- exact beam と NCC/GR は pseudo cutoff 後の `TVT_input` を隠した状態から再生成する。
- spatial/formation proxy は 見えない test で使える な X/Y/Z/MD と candidate disagreement に限定する。

## 受け入れ基準

- `exp043_ravaghi_feature_family_ablation_matrix` が作成され、`config.yaml` に route、lineage、feature families、variants が明記されている。
- train notebook は setup、入力確認、audit 実行、metrics/artifacts 確認をセル単位で追える。
- audit script は overall、split/fold、distance bucket、well metrics、feature importance を保存する。
- candidate ごとの feature family flag を持つ `single_lgbm_family_matrix.csv` を保存する。
- full run 前に static validation と lightweight smoke が実行できる。
