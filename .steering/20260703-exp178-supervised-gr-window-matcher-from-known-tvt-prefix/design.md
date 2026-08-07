# 設計

## アプローチ

raw train の各 horizontal well から finite `TVT_input` prefix row を deterministic に抽出し、horizontal GR window と typewell GR window の pair を作る。positive は observed `TVT_input` 近傍、negative は +/-15/25/50/100ft decoy と、hand-crafted combo score が高い local hard decoy とする。

pair feature は real GR descriptor、shuffled typewell GR descriptor、candidate/context feature に分ける。real GR logistic、shuffled GR logistic、no-GR logistic、real GR expected-error regressor を同じ GroupKFold fold 0 で fit/evaluate し、real GR が negative controls を上回るかを見る。

## 実験範囲

- 対象実験: exp178_supervised_gr_window_matcher_from_known_tvt_prefix
- Route: pf_beam
- 親実験: supervised_gr_window_matcher_from_known_tvt_prefix backlog
- 変更する変数: GR window pair construction、real/shuffled/no-GR supervised matching score
- 固定する変数: raw train data、known prefix only、no PF/Beam rerun、no TVT prediction replacement、no submission

## 再現性設計

- seed policy: sorted well + evenly spaced row sampling。shuffled GR roll は experiment name + well id の SHA256。
- stochastic 処理の有無: sklearn logistic solver と HistGradientBoostingRegressor があるため `random_state=42` を固定する。
- PF/Beam / likelihood-PF / seed bagging の有無: なし。既存 PF/Beam cache も読まない。
- 並列処理と乱数の関係: 並列 RNG なし。`num_workers=1`。
- CPU/GPU runtime と deterministic flags: CPU only、GPU disabled。
- train cache / test feature regeneration の SHA 記録方針: pair feature gzip と validation prediction gzip の raw/decompressed SHA、feature schema SHA を summary JSON に記録する。
- model manifest / prediction / submission SHA 記録方針: model artifact は保存しない。係数 CSV と validation prediction gzip SHA を記録し、submission SHA は対象外。
- Kaggle package bootstrap 確認方針: `prepare-kaggle-notebooks --strict` 後に canonical train kernel id/title で push する。

## リスク

- リークリスク: candidate minus observed TVT を feature に入れると trivial label leak になるため入れない。tail true TVT と `TVT_input` NaN 評価区間は pair 生成から除外する。
- CV/LB 不一致リスク: pair AUC smoke は LB に直結しない。positive でも downstream add-only feature 実験が必要。
- ランタイム/メモリリスク: max_wells=160、rows_per_well=64、max_pairs=120000 に制限する。
- 再現性リスク: sklearn モデルは bitwise deterministic anchor と扱わない。採用候補ではなく diagnostic として記録する。
