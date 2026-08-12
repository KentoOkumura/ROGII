# 要件

## 依頼

バックログ `hmm_exp226_candidate_selector_on_exp183` を実装する。exp183のselectorにHMM familyとexp226 K16 geometryを独立候補として追加し、候補絶対誤差rankerとcontinuity selectorでtrain-side auditする。

## 制約

- Route: `ensemble`
- 既存8候補を削除しない。新規候補はexp209 `blend_likpf_hmm_w500`、exp223 `hmm_selfgr_boost_only_a070_c100`、exp226 `v6_k16_geometry_gr_u_projection`。
- `pf_z`はselectableにしない。保存済みOOFによるoracle guardを別途満たすまで候補化しない。
- 3 headを再比較せず、過去最良のcandidate-error regressorだけを学習する。
- GroupKFold 5 folds、1 config、5 boosters、parent/control再学習なし。
- Viterbi penaltyはexp183 best ruleに固定し、grid tuningを行わない。
- raw-test inference、submission、selected path直接提出、softmax平均、true error gate、LGB出力のHMM再入力を行わない。
- `docs/06_reproducibility.md`に従い、OOF source、schema、model、predictionのSHA方針を記録する。

## 受け入れ基準

- 11候補のID/well/row/target/last-known TVT contractを実行時に検証する。
- 学習前に候補別RMSE、残差相関、unique-best rate、candidate oracle headroomを保存する。
- overall、distance bucket、hidden-like、worst-well、path switch、候補選択率を保存する。
- exp183 continuity 10.601482とexp226単体9.427110の双方を比較基準として記録する。
- train notebook、inference status notebook、config、README、SESSION_NOTES、result、metricsが一致する。
