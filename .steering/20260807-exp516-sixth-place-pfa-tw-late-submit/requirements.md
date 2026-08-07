# 要件

## 依頼と手法契約

- 依頼原文: 「6位解法のPFを再現する実験を行ってください。late subであることがわかるようにしてください。」追加確認により「PF単体の再現」へ固定した。
- 期待する成果: 6位解法の公開submission Notebookに含まれる単体`pfA × twGR`を再現し、hidden testを動的に処理するKaggle inference Notebookを作る。技術gate通過後、コンペ終了後の再現監査であることを明記した固定版を1回だけlate submitする。
- 一次資料: `docs/discussions/rogii-wellbore-geology-prediction-733226.md`（Kaggle discussion 733226、6th Place - PF, PF, PF, PF, Physics and Row-Level Bagging）。
- 参照実装: `docs/notebooks/rogii-wellbore-geology-prediction/solution_6th/k256net__public20th-private6th-pf-pf-pf-pf-and-bagging/public20th-private6th-pf-pf-pf-pf-and-bagging.ipynb`、kernel `k256net/public20th-private6th-pf-pf-pf-pf-and-bagging`。
- input: current runtimeのhorizontal well `MD, Z, GR, TVT_input`、対応typewell `TVT, GR`、train wellsの`X, Y, Z, TVT, TVT_input`からfold-safeに作るGR-free物理anchor、公開済み5-fold learned-emission encoder checkpoint。
- target / objective: PF実行時の教師lossはなく、hidden suffixの状態`position = TVT + Z`とrateの逐次posteriorを近似する。GR-free anchor GRUは参照実装どおりtrain wellsから学習し、learned-emission encoderは公開kernel outputの保存済みcheckpointを再学習せず読む。
- output: 各hidden rowの`pfA × twGR` whole-interval smoothed TVT、seed間加重std、正規化log-likelihood。late submissionではsmoothed meanを`sample_submission.csv`へIDで1対1整列した`submission.csv`を出す。
- loss: PF本体は学習lossなし。GR observationは公開実装のpower likelihood、GR-free anchor likelihood、learned-emission likelihoodの積。GR-free anchor GRUのみ参照実装のmasked Huber loss（delta 8 ft）で学習する。
- decode: 600 particlesを32 seedsで実行し、各seed内はfinal particle weightを祖先へbacktraceするwhole-interval ancestral smoother、seed間は正規化run log-likelihoodでsoft weightingしたmean/std。global de-shrink、91候補fusion、TCN、GBMは適用しない。
- context unit: `whole-well`。既知prefix末端からhidden suffix全体をforward filterし、suffix末端から全履歴をbackward祖先sweepする。GR-free anchorはfield内train wellsをfold-safeに利用する。
- 実装区分: `faithful`。公開submission sourceの`v96 pfA × tw`単体機構、公開`v96_art/pf_banks_config.json`、公開checkpoint、公開anchor生成コードをそのまま契約とする。
- 省略する機構と理由: 単体PFの契約からは省略なし。91候補bank、candidate-curve NN、TCN、GBM、de-shrinkはユーザーが選ばなかった最終system側であり、この実験の対象外。
- proxyで検証できない主張: N/A。なお6位最終systemのCV 5.4577 / Public 5.626 / Private 5.984は本実験が再現する主張ではない。
- proxyの場合のユーザー承認: N/A。

## 制約

- Route: `pf_beam`
- late submission phase: `post_competition_late_submission`。競技中の順位、正式最終順位、モデル選択用submissionと混同しない。
- Notebook title、submission message、`config.yaml`、`README.md`、`SESSION_NOTES.md`、`result.md`に`LATE SUBMIT`または`post-competition reproduction audit`を明記する。
- PF variant / representation / particle / seedは`pfA / tw / 600 / 32`の1本に固定し、OOFやlate LB後に変更しない。
- 物理anchorはvalidation情報混入を避ける公開5-fold手順、testは5 folds × 3 seedsの15モデル平均を使う。
- learned-emission encoder checkpointは公開kernel outputから読み、再学習しない。見つからない場合はemissionを無断でoffにせずfail-closeする。
- raw testのwell ID、well数、row数、内容SHAをハードコードせず、runtimeの`sample_submission.csv`をschema、ID集合、行順、行数の正とする。
- 再現性は`docs/06_reproducibility.md`に従い、PF seed、chunk/device割当、source/config/checkpoint/prediction/submission SHAを記録する。
- Kaggle push前にmetadataでGPU resourceを確認し、`kaggle quota --format json`の残時間を記録する。Active Sessions数はpush前gateにしない。

## 受け入れ基準

- 手法契約の`input / target / output / loss / decode / context unit`がコードと一致する。
- 公開v96 `pfA` configの全key/valueと実験内vendor copyのSHAが一致する。
- PF状態が`TVT + Z`、GR observationがtypewell interpolation、anchorとlearned emissionが同じparticle weightへ乗ることをcontract testで確認する。
- 600 particles、32 seeds、whole-interval mode、`pfA`、`tw`以外をlate-submit pathが実行しない。
- 公開checkpoint 5本が解決され、SHA manifestを保存する。欠損時にfallbackしない。
- inferenceはruntime train/testを動的列挙し、sample IDへ1対1整列し、missing/duplicate/extra/non-finiteを0にする。
- 技術gateを通った同一fixed versionだけを1回late submitする。OOF/LB後の救済gridや2回目提出は新しいユーザー承認なしに行わない。
- deterministic anchorと呼ぶ場合は、source/config/checkpoint、test prediction、submission、kernel version、rerun一致を記録する。GPU差で未確認なら`stochastic replay candidate`と記録する。
- gzip生成物を比較する場合はdecompressed content SHAを主証拠とする。
