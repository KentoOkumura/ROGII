# 設計

## アプローチ

exp029 の row artifact を読み込み、候補予測列を作って選択監査を行う。

1. `last_anchor_tvt`、公開 selector の `pf_pred`、scale 別 PF、`beam_pred`、PF/beam/hold blend から候補を構成する。
2. 各候補について全体 RMSE、fold 別 RMSE、well-hash fold 別 RMSE、distance bucket 別 RMSE、well 単位の win/loss を集計する。
3. same-OOF best / bucket oracle / well oracle は上限診断として保存する。
4. 採用判定用には、候補選択を train folds 上で行い held-out fold で評価する `leave_one_original_fold_out_selection` と、deterministic well hash 5-fold で同じ選択を行う `well_hash_holdout_selection` を使う。
5. rule selector は confidence columns だけで分岐する。`target_tvt`、error 列、future `TVT_input` は selector feature に入れない。

## 実験範囲

- 対象実験: `exp030_public_sel15_pf_candidate_selector`
- Route: `pf_beam`
- 親実験: `exp029_public_sel15_pf_oof_feature_generation`
- 変更する変数: public sel15 PF/Beam 候補の選択・blend 監査
- 固定する変数: exp029 の feature generation、cutoff 0.65、PF/Beam primitive、公開 replay anchor

## リスク

- リークリスク: selector feature に error / target / future TVT を混ぜると過適合する。コード側で feature 列と target/error 列を分け、same-OOF oracle を採用判定から除外する。
- CV/LB 不一致リスク: exp029 は train well の途中以降を隠す cutoff 0.65 だけなので、本番 test 条件と完全には一致しない。fold 外 selection と well-hash selection の両方で改善が安定する場合だけ次へ進める。
- ランタイム/メモリリスク: row artifact は約 1.78M 行。必要列だけを読み、集計中心にする。学習器を使う場合も shallow model に限定する。
