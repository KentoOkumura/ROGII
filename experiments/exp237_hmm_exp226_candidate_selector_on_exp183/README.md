# exp237_hmm_exp226_candidate_selector_on_exp183

## 状態

- Kaggle CPU train v1 完了。global CV は支持したが、near / worst-well guard は不通過。
- Route: `ensemble`
- ユーザー承認によりraw-test inferenceを実装中。Kaggle competition submitは行わない。

## 仮説

exp183 の PF/Beam/dense selector に、GR/HMM 主体の exp209 / exp223 と z/geometry 主体の exp226 を独立した候補 path として加える。候補ごとの絶対誤差を予測する ranker と、事前固定した well-local Viterbi continuity rule により、exp183 selector と exp226 単体の双方を guard 付きで上回れるかを検証する。

## 実装範囲

- 既存8候補を保持し、以下の3候補を追加する。
  - exp209 `blend_likpf_hmm_w500`
  - exp223 `hmm_selfgr_boost_only_a070_c100`
  - exp226 `v6_k16_geometry_gr_u_projection`
- exp183 の cluster/prior confidence、dense enrichment、hidden-like readoutを維持する。
- `lgb_candidate_error_ranker` だけを GroupKFold 5 folds で学習する。
- continuity は exp183 best rule（switch=20、jump weight=1、free jump=25 ft、likPF差75 ft以内、最小segment=1）に固定する。
- 学習前に候補別RMSE、残差相関、unique-best rate、8候補oracleからのheadroomを保存する。

## 実行計画

- active variants: 1
- LightGBM configs: 1
- folds: 5
- boosters: 5
- Viterbi variants: 1
- parent/control retraining: なし
- runtime: Kaggle CPU、internet off

## 検証方針

11候補のsource contractを確認後、outer well GroupKFold OOFでcandidate-error rankerを学習する。row-wise選択と固定Viterbi選択の双方を、overall、distance bucket、exp115 hidden-like、worst-well、path switch、selection distributionで比較する。candidate oracleは診断専用で、選択規則には使わない。

## 判定基準

overall RMSEだけでなく、全distance bucket、exp115 hidden-like、worst-well、path switch、候補選択率を読む。次段 `hmm_exp226_selector_rank_slot_addonly_on_exp218` に進むには、exp183 continuity CV 10.601482 と exp226単体CV 9.427110 の双方を上回り、guard悪化がないことを要求する。

## 禁止事項

- `pf_z` の selectable化
- true TVT/errorによる推論時gate
- selected pathの直接提出、softmax TVT平均、固定blend探索
- LightGBM予測のHMM emission/transitionへの再入力
- raw-test inference、submit

## 所見

Kaggle v1 は 3,783,989 rows / 773 wells、5 boosters を 3,051.086 秒で完走した。fixed Viterbi は RMSE 8.545093 で exp183 より -2.056388、exp226 単体より -0.882016 改善した。11候補 oracle も既存8候補 oracle 4.564605 から 2.883510 へ改善した。

一方、exp183 同一Viterbiとの比較では near 000_050 bucket が +0.167573、worst-well の最大回帰は 70925e23 の +25.639010 だった。ユーザー承認によりraw-test artifactだけは生成するが、submitとhmm_exp226_selector_rank_slot_addonly_on_exp218には進まない。

raw-test portはexp073 deterministic PF/Beam cache、exp226 K16 prediction、raw test再生成のexp209 exact HMM / exp223 self-GR HMM、raw GRで再計算するexp099 multi-observation score、saved exp237 5 fold modelを使う。train v1がfold imputation medianを保存していないため、model schemaごとにraw-test median（all-missing列は0）を再計算する。exp109/114のOOF-only cluster/prior confidence列はraw-test artifactがなく、出力summaryで列名・件数を監査する。これはtrain-side safety guardを解消するものではない。
