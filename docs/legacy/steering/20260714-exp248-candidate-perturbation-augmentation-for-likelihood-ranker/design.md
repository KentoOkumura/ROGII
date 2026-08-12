# 設計

## アプローチ

exp237 の固定11候補とtarget-free row contextを読み、各outer foldで以下を行う。

1. train wellから決定的にbase rowをsampleする。
2. `original_only` は11候補のclean candidate-long rowだけを作る。
3. `perturbation_augmented` は同じclean rowを必ず保持し、stable hashで各source candidateへ最大1つのaugmented cloneを割り当てる。
4. augmented cloneは `fixed_shift`、`common_datum_shift`、`low_frequency_drift`、`candidate_dropout`、`family_dropout`、`target_free_top_dropout`、`spread_scale` のいずれか1種類だけを適用する。摂動を合成しない。
5. 候補集合viewごとにcandidate mean/std/range、pair disagreement、candidate rank、availability maskを再計算する。
6. candidate TVTを変えるviewではraw horizontal GRとknown prefix TVTからmulti-observation score/MAE/NCCを再計算する。
7. train foldのtrue TVTから `within10` と `abs_error` labelだけを作る。
8. within10 classifierとL1 expected-error regressorを学習し、clean outer-validのoriginal11候補だけをscoreする。
9. expected errorのrow-wise argminとexp237固定Viterbiを評価する。augmented candidateは選択対象にしない。

shift gridは `±2/5/10/20/40/80 ft`。低周波driftはwell内の正規化tail位置に対する、開始点0の固定cosine rampとする。
common datum shiftはview内全候補へ同じshiftを加える。spreadはcandidate medianを中心に固定scaleで拡縮する。
dropoutのtopはmulti-observation scoreまたはfold-safe predicted scoreだけで決め、oracle topを使わない。

初回の原因分離ではexp218を候補bankへ追加しない。exp218 CV 8.475794は参考比較に留める。
augmentationがclean OOFとguardを改善した場合だけ、後続でexp218 candidateまたはnested rank-slotへの移植を検討する。

## 実験範囲

- 対象実験: `exp248_candidate_perturbation_augmentation_for_likelihood_ranker`
- Route: `ensemble`
- 親実験: `exp237_hmm_exp226_candidate_selector_on_exp183`
- 参照: exp111 learned likelihood、exp157/158/183 selector、exp218 ML anchor、exp115 hidden-like。
- 変更する変数: candidate-long train viewへの正解非依存augmentationの有無。
- 固定する変数: 11候補、入力OOF、row context、outer folds、LightGBM params、validation候補、fixed Viterbi、評価bucket。
- active variants: `original_only`, `perturbation_augmented`。
- objectives: `within10_classifier`, `expected_error_regressor`。
- cost: CPU、2 variants x 2 configs x 5 folds = 20 boosters。parent/control候補生成とexp218は再学習しない。

## 再現性設計

- seed policy: `sha256(seed, fold, well, id, candidate, view_slot)`からstable seed/choiceを作る。
- stochastic 処理: samplingと摂動割当のみ。global RNGを使わず、入力順やthread schedulingに依存させない。
- PF/Beam / likelihood-PF / seed bagging: 新規実行なし。保存済みfold-safe OOF candidateを固定入力にする。
- 並列処理: augmentation生成はsorted well/id/candidate順。LightGBMは固定seedを明示し、CPU runtimeを記録する。
- CPU/GPU: Kaggle CPU、GPU false、internet false。deterministic submission anchorとは扱わない。
- 入力証拠: exp099/072/065/109/114/115/209/223/226のfile/decompressed SHA、row/well/schemaを記録する。
- augmentation証拠: view type/count、amplitude分布、source family、dropout率、manifest content SHAを保存する。
- model証拠: 20 model manifestと各SHA、feature schema SHA、OOF likelihood/error/selectionのdecompressed SHAを保存する。
- inference/submission SHA: 初回は対象外。
- Kaggle bootstrap: prepare後にembedded configのvariant/config/fold/booster数、seed、GPU/internet、kernel sourceを確認する。

## リスク

- リークリスク: targetでshift方向・dropout対象・sample採否を決めるとoracle augmentationになる。stable hashだけで決定し、targetはlabel作成後段に限定する。
- foldリスク: valid wellのerror/rankをscore featureへ流すとleakする。validationはclean original、train augmentationはtrain_idxだけに限定する。
- feature整合リスク: candidate値だけ変えてmultiobsやspreadを据え置くと人工的な不整合を学ぶ。candidate依存特徴をviewごとに再計算する。
- CV/LB不一致: exp237はoverall改善に対しnear/worst-well guardとraw-test parityが失敗した。初回はtrain-side監査のみとする。
- ランタイム/メモリ: 全shift直積は5億long row級になる。1 source candidateにつき最大1 clone、base row cap、long row cap、chunked valid predictionで抑える。
- dropout shortcut: `is_augmented`、shift符号、真のdropout理由を入力featureにせず、診断列だけに保存する。
- 再現性: Pythonの`hash()`を使わずSHA256を使用する。gzipはdecompressed SHAを主証拠にする。
