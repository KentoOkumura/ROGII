# exp180_learned_gr_window_matcher_features_on_exp148 結果

## 現在の結論

Kaggle feature cache と 3 split train は完了したが、採用しない。3-model `lgb_mean` OOF は 8.5145263671875 で、親 exp148 `lgb_mean` 8.50128118189582 から +0.0132451852916803 悪化した。個別 model も `lgb0` 8.554800137862696、`lgb1` 8.581198811251788、`lgb2` 8.5779985181889 で全て exp148 より悪い。

したがって inference port、submit-check、submit は実行しない。

## 実装済み

- exp178 型の GR window pair scorer を exp148 full-train feature surface 用の add-only feature generator として実装した。
- train feature cache では `fold_safe_by_well=true` を既定にし、validation well の matcher score を他 well の observed-prefix pairs で学習した scorer から生成する。
- 出力 feature group は `learned_gr_window_matcher`。probability、expected-error、top1/top2 margin、entropy、real-vs-shuffled/no-GR gap、candidate-family indicator、`md_since` interaction を含む。
- `gr_matcher_features` notebook kind を `scripts/prepare_kaggle_notebooks.py` に追加した。
- `train_lgb0` v1 の kernel death 対策として、cache loader の `float32` 読み込み、列単位 join、不要 DataFrame の早期解放、学習不要列の drop を実装した。
- `train_lgb0` v2、`train_lgb1-r1` v1、`train_lgb2-r1` v1 の Kaggle CPU split train を完了した。
- OOF prediction を取得し、3 split の `lgb_mean` ensemble CV をローカルで計算した。

## 評価

- rows: 3,783,989
- wells: 773
- features: 355
- `lgb_mean` CV: 8.5145263671875
- exp148 `lgb_mean` CV: 8.50128118189582
- delta: +0.0132451852916803
- worst wells top3: `86454a6f` 48.29558181762695、`1b1eba53` 45.27809524536133、`fb03ae90` 45.117767333984375
- 1000+ distance bucket: individual split で 9.37612533569336 / 9.40521240234375 / 9.403876304626465
- feature importance: `grm_no_gr_prob_*` / `grm_shuffled_prob_*` は中位に入るが、expected-error 系はさらに下位で、exp148 anchor を上回るほどの寄与はなかった。

## 次アクション

`learned_gr_window_matcher_features_on_exp148` は backlog 完了/不採用として閉じる。GR window matcher は pair smoke では有効だったが、exp148 add-only feature としてはノイズが勝ったため、同じ形で exp148 へ追加拡張しない。続けるなら PF/Beam selector 側の候補信頼度に限定し、global ML feature 追加ではなく候補選択の不確実性診断として扱う。
