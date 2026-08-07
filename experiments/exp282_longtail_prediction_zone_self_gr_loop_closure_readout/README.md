# exp282_longtail_prediction_zone_self_gr_loop_closure_readout

## 状態

- ルート: `pf_beam`
- 状態: Kaggle CPU v1完了、technical PASS / scientific FAIL、branch closed
- CV: -
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-07-19
- 親実験: `exp090_lateral_self_gr_match_pseudotail_probe`
- 固定予測参照: `exp263_last_anchor_better_candidate_confidence_pair_cache`
- 並行比較: `exp281_exp226_residual_offset_exact_hmm_transition_probe`（結果待ちは不要）

## 仮説

long-tailの大誤差には、一度生じたTVT offsetが持続する区間が含まれる可能性がある。
`md_since >= 1000 ft`のreceiverを、同じwellの予測開始後0～500 ftにあるGR motifへtarget-freeに
再接続できれば、既知TVT区間を使わずに同一TVTらしさを検出できる。まずloop-closure edge自体の
精度と、matched donor側のexp263 fixed予測がreceiver側で改善方向を持つかだけを監査する。

## 変更点

- known `TVT_input` prefixをdonorにせず、prediction-zone donorを`0 <= md_since < 500 ft`、
  receiverを`md_since >= 1000 ft`に固定する。
- rolling mean 5後のGR windowをhalf-window 8 / 15 / 25、stride 3でforward / reverse NCC比較する。
- primary best/second gap、multiscale agreement、3 center以上のsegment supportから等重みconfidenceを作る。
- 全real edgeとwell内shuffled donor controlを凍結しcontent SHAを保存した後にだけtrue TVT、fold、
  hidden-like role、exp263 fixed OOF predictionを結合する。
- この実験は0-booster readoutに限定し、補正予測、hard copy、HMM/PF、inference、submissionを作らない。
- exp281はslow residual-offset HMMの並行仮説であり、本実験の実装・実行条件にはしない。

## 検証方針

- Fold: 保存済みgroup-safe 5 folds
- Group: `well`
- 対象: unknown suffix 3,783,989 rows / 773 wellsのうち固定eligible center
- 主指標: `abs(TVT_receiver - TVT_donor) <= 2/5/10 ft`、real-vs-shuffled lift、
  absolute delta TVT、coverage、orientation、fold、hidden-like、by-well
- high-confidence: well内target-free confidence上位10%
- donor-transfer: edge freeze後だけ、matched donorのexp263 fixed OOF予測をreceiver truthへ評価する診断
- Leakage Check: edge生成stageではwell、row index、MD、GR、`TVT_input`だけを許可し、
  true TVT、target/error/oracle、exp263/exp281予測を禁止する。

## 固定guard

- eligible edge / finite score coverageはともに1.0。
- high-confidence `within10` precisionはpooledで0.60以上。
- all-edge / high-confidence `within10` lift vs shuffledとhigh-confidence median delta改善は各5/5 folds。
- high-confidence receiver coverageはlong-tail receiverの1%以上。
- post-freeze donor-transferはexp263 receiver baselineを5/5 foldsで改善し、pooled RMSE gain 0.10 ft以上。
- hidden-like spatial / typewell-purgedの両方でhigh-confidence `within10` liftが正。
- 一つでもFAILなら補正・parameter rescue・raw-test inferenceへ進まない。

## 実行契約

- active variant / LightGBM config / trained fold / booster: `1 / 0 / 0 / 0`
- HMM / PF variant: `0 / 0`
- parent/control再学習: 0
- well逐次、receiver chunk 256、full pairwise matrixは保存しない。
- Kaggle CPU / GPU off / internet offでversion 1を実行済み。

## 実装

- `*_compact_selfcontained_train.py`をJupytext percent形式で実装し、10章構成の`.ipynb`へ変換した。
- score-stageは`MD`、`GR`、`TVT_input`だけを受け、real/shuffled edgeをgzip保存してlogical
  content SHAを固定した後にだけraw true TVT、exp263 Stage 0 fixed OOF、fold、hidden-like roleを結合する。
- forward / reverse NCC、固定tie-break、multiscale agreement、segment support、well内top 10%
  confidence、stable SHA256 well-local shuffle、overall/fold/scope/orientation/by-well/donor-transfer
  readoutと固定guardを実装した。
- `*_compact_selfcontained_inference.py` / `.ipynb`は常にfail-closedで、submissionを生成しない。
- 2026-07-19の実行依頼を明示承認として、compact train/inferenceを正規notebookへ採用した。

## 静的検証

- exp282 synthetic tests: 6 passed
- exp280/281/282 targeted tests: 18 passed
- Jupytext train/inference round-trip、`py_compile`、ruff: PASS
- `make validate-exp EXP=exp282_longtail_prediction_zone_self_gr_loop_closure_readout`: strict PASS
- `make validate-template`: PASS
- 全体testは178 passed、今回未変更のexp264 config/test status不一致1件だけFAIL

## 結果

| メトリック | 値 |
| --- | --- |
| guard | technical PASS / scientific FAIL |
| high-confidence within10 | 0.554309 vs shuffled 0.551052（+0.003257） |
| high-confidence positive-lift folds | 4/5（required 5/5） |
| donor-transfer RMSE | 15.849509 vs receiver baseline 8.954770（gain -6.894739 ft） |
| donor-transfer improved folds | 0/5 |
| frozen edges | 997,733 / coverage 1.0 |
| runtime | 248.206秒 |
| Public LB | - |
| Private LB | - |

## 所見

### 設計上の判断

- exp281の完了を待たず、prediction-zone loop closureの識別力を独立診断できる。
- edgeのtarget-free freeze境界とpost-freeze truth attachmentを明示した。
- readout通過を補正採用とみなさず、soft correctionは別実験へ分離した。

### 実行結果

- edge/finite-score coverageとtarget-free freeze境界は全PASSした。
- pooled liftは小さく、high-confidenceでもfold 0が負、within10は固定閾値0.60に未達だった。
- donor-transferは全5 foldsで悪化し、matched donorをpseudo-anchorとして使う根拠は得られなかった。
- hidden-like spatial / typewell-purgedのliftは正だったが、主guardの失敗を救済しない。

### リスク / 注意

- GR値単体、近接自己相関、反復地層により偽matchが生じ得る。
- window/stride/confidence/guardは事前固定し、結果を見た救済gridを行わない。
- O(n^2)全行行列を作らず、donor範囲、stride、well逐次、chunkでmemoryを制約する。

## 次

固定契約どおりparameter rescue、補正、inference、submissionへ進まずbranchを閉じる。
window/stride/confidence/donor範囲の救済gridや新規backlogは追加しない。

## 表記

用語は `KAGGLE_DIRECTION.md` の表記方針と `docs/glossary.md` に合わせ、実験名や設定名を除いて日本語優先で記録する。
