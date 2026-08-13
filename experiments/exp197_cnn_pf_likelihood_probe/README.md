# exp197_cnn_pf_likelihood_probe

## 状態

- ルート: pf_beam
- 状態: implemented_pending_kaggle_train
- CV: -
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-07-04
- 親実験: `cnn_pf_likelihood_probe` backlog

## 仮説

PF の point-GR likelihood を local CNN/SDF likelihood に置き換える前に、固定 PF/Beam 候補上の scorer として real GR が shuffled/no-GR control を上回るか確認する。candidate AUC、topK coverage、weighted RMSE、ESS、worst-well が exp099 multiobs / exp111 / likPF baseline を上回らなければ終了する。

## 変更点

- exp099 train v2 の fixed candidates を candidate-long 化。
- candidate TVT 周辺 typewell GR window と horizontal GR window から 5ch CNN/SDF image を作る。
- `real_gr` / `shuffled_gr` / `no_gr` を同一 split / sample schedule で学習。
- live PF weight replacement、raw-test generation、submit は対象外。

## 検証方針

- Fold: GroupKFold fold0 smoke
- Group: `well`
- Stratification: なし
- Leakage Check: true TVT は label / metrics のみ。candidate window center は fixed candidate TVT、observed prefix は `TVT_input` の finite prefix のみ。

## 実行入口

- 学習 notebook: `exp197_cnn_pf_likelihood_probe_train.ipynb`
- 推論 notebook: `exp197_cnn_pf_likelihood_probe_inference.ipynb`
- Kaggle 準備: `task prepare-kaggle-notebooks EXP=exp197_cnn_pf_likelihood_probe`
- notebook 実行: Kaggle kernel run を正とする。ローカル実行は `--allow-local` を付けた smoke debug のみに限定する。

## 結果

| メトリック | 値 |
| --- | --- |
| CV | - |
| Public LB | - |
| Private LB | - |

## 所見

### 良かった点

- 未実行。

### 悪かった点

- 未実行。

### リスク / 注意

- GPU training のため deterministic anchor ではない。
- real GR が shuffled/no-GR を明確に上回らなければ PF weight follow-up へ進まない。
- 改善しても raw-test parity と worst-well guard なしで submit しない。

## 次

1. `make validate-exp EXP=exp197_cnn_pf_likelihood_probe`
2. `make prepare-kaggle-notebooks EXP=exp197_cnn_pf_likelihood_probe EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp197-cnn-pf-likelihood-probe-train --title 'exp197 cnn pf likelihood probe train' --run-on-push --strict"`
3. GPU cost guard を確認してから Kaggle train push。

## 表記

用語は `backlog/KAGGLE_DIRECTION.md` の表記方針と `docs/glossary.md` に合わせ、実験名や設定名を除いて日本語優先で記録する。
