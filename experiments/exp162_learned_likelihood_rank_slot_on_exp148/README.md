# exp162_learned_likelihood_rank_slot_on_exp148

## 目的

exp148 の learned likelihood confidence add-only surface を親に、exp098 の rank-slot U-shape 表現を exp145 の learned probability / predicted-error 順位で作り直して評価する。

## 状態

実装済み、Kaggle CPU split train / hidden-safe inference / code submission まで完了。Public LB は 8.100 で exp148 の 7.960 を上回らなかったため、不採用。

## 仮説

exp098 の heuristic rank-slot は exp073 では効いたが、exp092/exp148 にそのまま足すと重複やノイズが強かった。rank の作り方を exp145 の candidate 別 learned probability と predicted absolute error に置き換えることで、exp148 がすでに持つ confidence 特徴の上に、候補間の U-shape / path disagreement をより意味のある形で渡せる可能性がある。

## 変更点

- 親: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- base surface: `exp092_u_projection_correction_disagreement_fullrun`
- learned likelihood source: `exp145_learned_likelihood_rawtest_feature_generator_parity`
- rank-slot source concept: `exp098_selector_rank_slot_features_on_exp073`
- control: 再学習しない。保存済み exp148 CV / Public LB を historical baseline として参照する。
- variant: `learned_likelihood_rank_slot_addonly`
- runtime: CPU deterministic threads8
- route: `ml_model`

Candidate TVT path の direct selector、soft average、blend、postprocess replacement は入れない。

## 検証方針

GroupKFold 5 folds を well group で実行し、active variant 1 個、LightGBM config 3 個、fold 5 個、合計 15 boosters を CPU で学習する。CPU runtime timeout を避けるため、学習 code は `train_lgb0` / `train_lgb1` / `train_lgb2` に分割し、それぞれ 5 boosters だけを担当する。比較基準は exp148 `lgb_mean` CV 8.501281182 / Public LB 7.960、exp098、exp139、exp147、exp153。

## 所見

split train の pooled RMSE は `lgb0` 8.488049241、`lgb1` 8.456600574、`lgb2` 8.443346041。CV では exp148 `lgb_mean` 8.501281182 を上回る single model が出たが、hidden-safe inference v4 の code submission は Public LB 8.100 で exp148 7.960 より悪化した。

## 実行状態

Kaggle Notebook 実行を正とした。`exp162_learned_likelihood_rank_slot_on_exp148_train_lgb0.py/ipynb`、`train_lgb1.py/ipynb`、`train_lgb2.py/ipynb` が CPU split train の正の実装。
