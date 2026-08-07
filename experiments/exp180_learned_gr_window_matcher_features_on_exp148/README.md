# exp180_learned_gr_window_matcher_features_on_exp148

## 状態

- Route: `ml_model`
- Status: `completed_train_side_rejected_no_submit`
- 親実験: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- 比較基準: exp148 `lgb_mean` CV 8.50128118189582 / Public LB 7.960
- feature cache: Kaggle v1 `kentookumura/exp180-gr-matcher-exp148-features` complete
- train: Kaggle `lgb0` v2、`lgb1-r1` v1、`lgb2-r1` v1 complete
- OOF: `lgb_mean` CV 8.5145263671875
- exp148 比較: +0.0132451852916803 悪化
- control 再学習: なし
- 提出: なし

## 仮説

exp178 の known-prefix supervised GR window matcher では、real GR が shuffled/no-GR control を明確に上回った。候補 TVT を直接置換せず、exp148 の既存候補に対する match probability、expected-error、top1/top2 margin、entropy、real-vs-shuffled/no-GR gap を add-only feature として渡せば、exp148 が PF/Beam/SC 候補の信頼度をよりよく使える可能性がある。

## 検証方針

`learned_gr_window_matcher_addonly` だけを学習する。matcher scorer は observed `TVT_input` prefix pair から作り、train cache では GroupKFold by well で validation well の scorer を別 well の prefix pairs だけから学習する。候補 TVT の hard switch、weighted TVT、direct correction、PF/Beam 再生成は行わない。

## 実装

- feature cache notebook: `exp180_learned_gr_window_matcher_features_on_exp148_gr_matcher_features.py`
- train notebook: `exp180_learned_gr_window_matcher_features_on_exp148_train.py`
- split train notebooks: `train_lgb0.py` / `train_lgb1.py` / `train_lgb2.py`
- core implementation: `learned_gr_window_matcher_features_on_exp148.py`

## 所見

Kaggle feature cache notebook v1 と 3 split train は完了。`lgb0` v1 は `gr_matcher_join_start` 直後の kernel death で失敗したため、memory 対策を入れた v2 を採用した。`lgb1` / `lgb2` は canonical slug が `Notebook not found` になったため、retry slug `-r1` で実行した。

個別 pooled OOF は `lgb0` 8.554800137862696、`lgb1` 8.581198811251788、`lgb2` 8.5779985181889。3 split の OOF prediction をローカルで align して計算した `lgb_mean` は 8.5145263671875 で、exp148 `lgb_mean` 8.50128118189582 から +0.0132451852916803 悪化した。feature importance では `grm_no_gr_prob_*` / `grm_shuffled_prob_*` が中位に入ったが、全体改善にはつながらなかった。

global OOF が baseline 未達のため、hidden-like stress、inference port、submit は実行しない。

## 注意

- `TVT_input` が NaN の評価区間 true TVT は matcher label、window center、threshold selection、normalization に使わない。
- global OOF が改善しても、near-row、`1000_plus`、worst-well、hidden-like stress、current-test parity が弱ければ submit しない。
