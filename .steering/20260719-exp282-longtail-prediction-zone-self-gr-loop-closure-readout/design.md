# 設計

## 結論

初回実験は「補正」ではなく、prediction-zone self-GR edge自体の同一TVT精度と、exp263 fixedの
earlier donor予測がlong-tail receiverで改善方向を持つかを分離する0-booster readoutとする。
exp281とは並行実施可能であり、結果待ちは先行条件にしない。

## 実験範囲

- 対象実験: `exp282_longtail_prediction_zone_self_gr_loop_closure_readout`
- Route: `pf_beam`
- 科学的親: `exp090_lateral_self_gr_match_pseudotail_probe`
- 固定比較基準: exp263 fixed `exp226_w500_50_50`
- 参照する失敗:
  - exp091: self-GR direct candidateは大幅negative。
  - exp134: self-GR qualityを使うhard long-tail gateはnegative。
  - exp223: weak self-GR HMM emissionは一部positiveだがworst-well regressionが大きい。
- positive参照:
  - exp090: known-prefix multiscale self-GR add-onlyは小幅positiveでhalf-window 25が主要だった。
  - exp280: raw-GR shift likelihoodは1000+でもstable shuffledよりfold-stableに良かった。
- 並行比較: exp281 slow residual-offset HMM。入力・decoder・生成物を本実験へ流用しない。
- 変更する変数: donorをknown prefixからprediction-start寄りunknown zoneへ変更し、
  receiverを1000+ long-tailに限定する。
- 固定する変数: raw GRのみのtarget-free matching、exp090由来のrolling mean 5、
  half-window `[8, 15, 25]`、stride 3、truth attachment順序、5 fold、metric、guard。

## 入力契約

### Score stageで許可する入力

- raw train horizontal: well id、row index、`MD`、`GR`、`TVT_input`
- canonical row identity: exp263 Stage 0 candidate cache manifest / partition identity

### Score stageで禁止する入力

- raw horizontal `TVT`
- `target`、`tvt_true`、`error`、`abs_error`、oracle label
- exp263 fixedおよび各primitive prediction/confidence
- exp281 prediction、posterior、error、diagnostic
- fold assignment、distance bucket、exp115 hidden-like role
- training-only formation列

score-stage APIは上記outcome-like列を受け取った場合にfail-closedで停止する。edge tableには
`truth_attached=false`を記録し、row count、well count、schema SHA、logical content SHAを確定する。

## GR前処理

wellごとに逐次処理する。

1. `GR`をnumeric化する。
2. well内で両方向linear interpolationする。
3. finite fallbackはknown prefix GR mean、これがfiniteでなければfull-well GR mean、さらに
   finiteでなければwellをinvalidとして停止する。
4. centered rolling mean window 5、`min_periods=1`を適用する。
5. 各windowをmean 0 / std 1へz-normalizeし、stdには`1e-6`を加える。

## Eligible row契約

- prediction startは最後のfinite `TVT_input` rowの次行とする。
- `md_since = MD - last_known_MD`。
- donor center: `0 <= md_since < 500 ft`かつprediction zone内。
- receiver center: `md_since >= 1000 ft`かつprediction zone内。
- donor / receiver centerはraw row indexの先頭からstride 3で固定する。
- half-window 25が収まらない端はedge候補から外し、eligible center分母と除外理由を保存する。
- known prefix、500～1000 ft、同一または近接区間はdonor/receiverとして使わない。

## Matching契約

### Primary edge

- scale half-window 25のNCCをprimary scoreにする。
- receiver windowとdonor windowのforward / reversedの両方向を比較する。
- donor候補全体のbest / second-best NCCを取得し、best donor row、orientation、score、gapを保存する。
- tieは`score desc -> orientation forward優先 -> donor row index asc`で決定する。

### Multiscale agreement

- half-window 8 / 15でも同じbest donor探索を行う。
- auxiliary scaleのbest donorがprimary donorから25 raw rows以内でorientationも一致した割合を
  `scale_agreement`（0、0.5、1.0）とする。

### Segment support

- receiver centerをrow順に並べる。
- forwardは隣接receiver centerに対してdonor centerが`+stride`、reverseは`-stride`へ進むことを期待する。
- 1 stride以内のずれを許し、3 center以上連続したrunに属するedgeを`segment_supported=1`とする。
- run lengthとorientation flip countを保存する。

### Target-free confidence

well内eligible edgeについて次の4成分を作る。

1. primary best NCCのpercentile rank
2. best-second gapのpercentile rank
3. `scale_agreement`
4. `segment_supported`

4成分の等重み平均を`edge_confidence`とし、well内top 10%をprimary high-confidence bucketへ固定する。
feature weight、quantile、window、strideは同一OOFのtruthを見て変更しない。

## Negative control

- real matchingはRNGなし。
- stable shuffled controlは、real edgeを凍結した後、同じwellのdonor row assignmentだけを置換する。
- seedは`SHA256(experiment_name, seed=42, well_id, "donor_shuffle")`から生成したlocal
  `np.random.default_rng`へ渡す。
- receiver、confidence、coverage、donor pool分布はrealと一致させる。
- global RNGと並列実行順に依存させない。

## Truth / prediction attachmentとreadout

全wellのedge tableを保存しlogical content SHAを確定した後に、別readerでのみ次を結合する。

- raw horizontal true `TVT`（receiver / donor）
- exp263 fixed OOF prediction（receiver / donor）
- fold、distance bucket、hidden-like role

次を集計する。

- `abs(TVT_receiver - TVT_donor)`のmean / median / RMSE / p90 / p95
- within 2 / 5 / 10 ft precision
- real-vs-shuffled lift
- all edge / high-confidence / forward / reverse / segment-supported
- pooled / 5 folds / well / 1000～1500 / 1500～2500 / 2500+ / hidden-like 2面
- exp263 receiver prediction RMSEと、matched donorのexp263 predictionをreceiverへ評価した
  post-freeze donor-transfer readout
- donor-transferのwell別改善/悪化、最大回帰、coverage

donor-transferは診断表だけに保存し、row-wise補正predictionやsubmission候補として保存しない。

## 固定guard

### Technical guard

- canonical row/fold identity一致
- expected folds `[0,1,2,3,4]`
- eligible receiver center edge coverage 1.0
- finite score coverage 1.0
- forbidden score-stage column 0
- edge content SHA確定前のtruth attachment 0

### Scientific guard

- pooled high-confidence within10 precision `>= 0.60`
- all-edge within10 lift vs shuffledが5/5 foldsで正
- high-confidence within10 lift vs shuffledが5/5 foldsで正
- high-confidence median absolute delta TVTがshuffledより5/5 foldsで小さい
- high-confidence receiver coverageがlong-tail receiverの1%以上
- post-freeze high-confidence donor-transferがexp263 receiver baselineを5/5 foldsで改善
- pooled donor-transfer RMSE gain `>= 0.10 ft`
- hidden-like spatial / typewell-purgedの両方でhigh-confidence within10 liftが正

全guard PASSだけが、別実験でsoft correctionを検討する条件となる。部分PASSやoverallだけの改善で
window / stride / quantile / weight / thresholdを救済しない。

## 実行契約

- active audit variant: 1
- model / LightGBM config / trained fold / booster: 0 / 0 / 0 / 0
- HMM / PF well-run: 0 / 0
- parent/control再学習・再生成: 0
- CPU / GPU / internet: Kaggle CPU / off / off
- process: well逐次、receiver chunk 256、全well pairwise matrix保存禁止
- inference / submission: disabled / disabled
- Kaggle push: 実装・静的検証・実行量再確認後にユーザーの別承認が必要

## 再現性設計

- seed policy: real scoreはRNGなし。shuffleだけwell-keyed stable SHA256 local RNG。
- stochastic処理: donor shuffled negative controlのみ。
- 並列処理: 初回はsingle process。将来並列化してもwell local stateだけを使う。
- gzip: raw gzip SHAとdecompressed content SHAを分け、decompressed contentを主証拠にする。
- 入力: raw well file SHA、exp263 manifest/partition logical SHA、hidden-like SHAを記録する。
- 出力: edge schema/content SHA、readout CSV/JSON SHA、config/source/notebook SHAを記録する。
- model / prediction / submission SHA: model・補正prediction・submissionを作らないため対象外。
- deterministic anchor: prediction/submission anchorとは呼ばず、fixed-input diagnosticとする。
- Kaggle package: prepare後にloose config/sourceとbootstrap manifestのSHA、CPU/internet metadataを照合する。

## 予定生成物

- `exp282_longtail_prediction_zone_self_gr_loop_closure_readout_edge_contract.json`
- `exp282_longtail_prediction_zone_self_gr_loop_closure_readout_target_free_edges.csv.gz`
- `exp282_longtail_prediction_zone_self_gr_loop_closure_readout_edge_schema.csv`
- `exp282_longtail_prediction_zone_self_gr_loop_closure_readout_overall_metrics.csv`
- `exp282_longtail_prediction_zone_self_gr_loop_closure_readout_fold_metrics.csv`
- `exp282_longtail_prediction_zone_self_gr_loop_closure_readout_scope_metrics.csv`
- `exp282_longtail_prediction_zone_self_gr_loop_closure_readout_orientation_metrics.csv`
- `exp282_longtail_prediction_zone_self_gr_loop_closure_readout_by_well_metrics.csv`
- `exp282_longtail_prediction_zone_self_gr_loop_closure_readout_donor_transfer_readout.csv`
- `exp282_longtail_prediction_zone_self_gr_loop_closure_readout_input_manifest.csv`
- `exp282_longtail_prediction_zone_self_gr_loop_closure_readout_summary.json`

## リスク

- リークリスク: unknown suffix true TVTやexp263 predictionをedge選択へ混ぜると直接リークする。
  score/readout reader分離、forbidden列guard、truth attachment順序、content SHAで防ぐ。
- 反復GR motif: 異なるTVTでもGRが似る。reverse、multiscale、gap、segment support、shuffled controlで
  識別し、hard copyを禁止する。
- 自己相関: 近接行は自明に似る。donor 0～500 ft、receiver 1000+へ固定して分離する。
- 絶対位置不定: unknown-only edgeは絶対TVTを決めない。初回はedge精度だけを監査し、将来も
  target-free confidenceを持つbase donorなしに補正しない。
- CV/LB不一致: current phaseではLBとCVの不整合があるため、Public LBを使ったthreshold選択をしない。
- ランタイム/メモリ: full unknown-to-unknown O(n^2) matrixは禁止し、donor範囲、stride、well逐次、
  receiver chunkで上限を固定する。
- 再現性: shuffled control以外は決定的。stable key seedとsingle processを初回契約にする。
