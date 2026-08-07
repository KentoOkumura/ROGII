# exp312 セッションノート

## 現在の状態

- 2026-07-21: 設計確定。旧条件はexp311全gate PASS待ち。
- 2026-07-21: ユーザーがexp311の平均改善を根拠に次段へ進むよう指示。exp293 deployable12を固定評価bankにする方針を確認し、exp312実装を承認。
- 2026-07-21: ユーザーがKaggle実行を承認。compact trainを正規train notebookへ採用し、private CPUで実行する。
- Route: `pf_beam`
- 実行契約: scientific 1 + controls 2 / 5 folds / 0 model / 0 booster / 0 decoder。
- canonical kernel: `kentookumura/exp312-typewell-gr-emission-rank-audit-train`（title: `exp312 typewell gr emission rank audit train`）。
- 親候補の再生成・親controlの再実行: 0。Kaggle internet/GPU: off/off。

## 仮説と変更点

exp311で平均的に転送できたType Well群GR residual情報を、scalar affineではなく条件付きStudent-t emissionとして表現すれば、exp293 deployable12のtruth-nearest候補rankをglobal residual modelより安定して改善できる。候補bank、候補値、decoderは固定し、変更するのは候補TVTに与える診断用GR尤度だけ。

## 設計契約

- fixed bins、Student-t df=5、support k=200、4段fallbackを変更しない。
- tableとcandidate scoresはouter-valid truth結合前に凍結する。
- candidate/bin/table/fallback/rank readoutのschema/content SHAを記録する。
- exact-HMM/PF/Beam、ML、inference、submissionは禁止する。
- exp311 summary/fold/group SHAをhard preflightし、親の`completed_gate_failed`と2つの失敗checkを保持する。
- exp293 deployable12はexp263 partitionから同じformula順で再構成し、manifest SHA、partition SHA、small parity、candidate content SHAを検証する。candidate生成runは0。
- baselineはglobal-unconditional Student-t。realはgroup conditional→group unconditional→global conditional→global unconditional、df=5、support k=200、scale floor 1 GR API。
- group shuffleはwell label multisetのSHA256 rotation、TVT shiftはwell内candidate-TVT行のSHA256 circular shift。いずれもtruth-free。

## 親gateの明示上書き

- exp311平均gain: `0.376220` GR API、5/5 folds改善、noise R² `0.202320`。
- 保持するFAIL: fit-RMSE R² `-0.003255`、worst-well delta `+12.914716`。
- 本実験を実装する判断だけを上書きし、exp311をpromotion PASSへ変更しない。exp312もMRR/top3、shuffle差、4/5 folds、hidden-like、fallbackの全gateを要求する。

## 実装

- compact self-contained trainはdeployable12 reconstruction、target-free raw/typewell context、fold別table、rank-order memmap freeze、late truth join、promotion、10生成物までnotebookセルへ展開した。
- compact inferenceはcandidate regeneration、decoder、inference、submissionをfail-closedにした。
- ユーザーの実行承認に基づき、compact trainを正規train notebookへ採用する。inferenceはfail-closed placeholderのまま実行しない。
- 親exp311 compact train 1,388行に対しexp312 compact trainは約2,000行。親と同等のruntime/input/SHA/fold-safe fit/control/late truth/metrics/orchestration章を持ち、deployable12 reconstructionとhierarchical table章を追加した。

## 静的検証

- Jupytext train/inference変換と`--test`: PASS。
- train/inference `py_compile`、ruff: PASS。
- exp312専用テスト9件: PASS。exp311と合わせて16件PASS。
- `make validate-exp EXP=exp312_typewell_group_conditional_gr_emission_table`: strict PASS。
- `make validate-template`: PASS。
- notebook sourceの`__file__`参照: 0。
- `review_exp_docs.py`: core evidence categories present。

## Kaggle CPU version 1

- 正規train採用後、Jupytext、構文、ruff、16 tests、strict experiment/template validationを再実行し、すべてPASSした。
- `make prepare-kaggle-notebooks EXP=exp312_typewell_group_conditional_gr_emission_table EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp312-typewell-gr-emission-rank-audit-train --title 'exp312 typewell gr emission rank audit train' --run-on-push --strict"`: PASS。
- `make push-kaggle-train EXP=exp312_typewell_group_conditional_gr_emission_table`: version 1 push成功、完了。
- push後のmetadata pull: `id_no=128090149`、private=true、GPU/internet=false、kernel source 3件を確認。
- URL: `https://www.kaggle.com/code/kentookumura/exp312-typewell-gr-emission-rank-audit-train`
- status: `completed_gate_failed`、runtime `326.622947 sec`。

## 結果

- baseline MRR/top3: `0.3361118143 / 0.3580901002`。
- conditional real MRR/top3: `0.3345193890 / 0.3556458542`。
- gain: MRR `-0.0015924252`、top3 `-0.0024442460`、改善fold `0/5`。
- real minus group-shuffle MRR: `+0.0016114684`（閾値`+0.02`をFAIL）。
- real minus candidate-TVT shift MRR: `+0.0638088042`。
- hidden-like nonregression: spatial/typewell-purgedともFAIL。
- lookup fallback rate: `0.0182310696`（上限`0.25`をPASS）。
- late truth: 全5 foldsでfreeze前outer-valid truth access `0`。
- 候補bank: 3,783,989 rows / 773 wells / 12 candidates、生成run `0`、candidate values changed=false、formula parity PASS。
- 禁止出力: candidate generator/model/booster/decoder/prediction `0`、submission=false。

## 生成物・再現性

- Kaggle生成物は予定10/10件。選択取得した9 manifest対象ファイルのraw SHAはsummaryと9/9一致。
- metrics raw SHA: `301f3ab63e68ec481acb0459c50bd10e1e33e6e3beb6efbe9b6fea5d44e893f7`。
- summary raw SHA: `5849d389288ec0a0922185b9264dec222e7928c0715f27e820260a3dbd33d477`。
- candidate content SHA: `29477141685662bae7417e788ec5dbe914c2220b7f4cc45ab01befa4e5e3b474`。
- emission-table content SHA: `59d3d4a4ff7207b991fb91cd8e830964906f584ad7b18edc103e2ef9f0616b68`。
- rank-metrics content SHA: `b8e7ec5fa31f8e7f2827c38e2ddcf858bb16cec513a63566fa7c0c4c2c63f169`。
- 全output取得は一時memmapを含むため中止し、`--file-pattern`で`artifacts/`と`metrics.json`だけを取得した。Kaggle実行自体への影響はない。

## 判定と次

平均MRR/top3が悪化し、0/5 folds、shuffle差とhidden-likeもFAILしたためpromotion FAIL。TVT-row alignment自体の信号はshift control差に見えるが、Type Well群条件づけの追加価値は示されなかった。条件セル分割後に群ラベル固有の順位情報が残らないことが失敗原因と考える。事前契約どおりbin/df/kの救済を行わずbranchを閉じ、exp313〜320は停止を維持する。次候補は群priorに依存しない既存exp305、続いて既存順序の独立exp321とし、新しい同系救済backlogは追加しない。
