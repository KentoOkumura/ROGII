# exp256_segment_local_corridor_near_bucket_signal_attribution_readout

## 状態

- ルート: PF/Beam (`pf_beam`)
- 状態: completed / diagnostic-only
- CV / Public LB / Private LB: 対象外
- inference / submission: disabled
- 作成日: 2026-07-15
- 親実験: `exp250_segment_local_negative_space_gr_corridor_audit`

## 仮説

exp250のnear pooled AUC約0.82が、GR topology固有の広いsignalではなく、
distance、candidate family、wellのbase error構成とrisk飽和で説明されるかを切り分ける。

## 変更点

- exp250 Stage 1の保存済み生成物だけを固定入力にした。
- distance x family conditional AUCを算出し、cross-family比較をpooled AUCから分離した。
- family x wellのpaired AUC差をpair massで加法分解した。
- 0--100 ftの評価weight shareとrisk=1.0飽和率を保存した。
- corridor、candidate、model、control、parameterを再計算していない。

## 検証方針

- Fold: なし。保存済みretrospective diagnostic。
- Group: `well`。
- Pair: `(well, segment_id, candidate)`のreal / shuffled。
- Leakage Check: truth由来bad/good weightは診断labelにのみ使い、prediction / selector / gate / featureを作らない。
- Identity Check: real / shuffledのbad/good weight、segment範囲、paired risk差をfail-closed検証した。

## 実行入口

- train: `exp256_segment_local_corridor_near_bucket_signal_attribution_readout_train.ipynb`
- inference: `exp256_segment_local_corridor_near_bucket_signal_attribution_readout_inference.ipynb`（fail-closed guard）
- Kaggle: `kentookumura/exp256-seglocal-near-signal-attribution-train` v1

## 結果

- near pooled weighted-bucket AUC: real 0.819846 / shuffled 0.773559
- near distance x family conditional AUC: real 0.598678 / shuffled 0.574742
- near weight share: 1.048546%
- family x well conditional AUC: real 0.522220 / shuffled 0.511096
- family x well positive pair-mass share: 0.522241
- pooled risk=1 weight fraction: real 0.188251 / shuffled 0.270472

## 所見

### 良かった点

- exp250 Stage 1を再実行せず6.49秒で原因を切り分けた。
- paired input identity、output SHA、family x well寄与の加法的一致を確認した。
- nearの弱いGR差と、pooled AUCを押し上げるcandidate-family base-rate交絡を分離した。

### 悪かった点

- nearは全評価weightの1.05%しかない。
- nearでAUC算出可能なのは6 / 10 family-bucket strata、4 / 5 familiesだった。
- family x wellでは正負がほぼ半々で、広いwell一般化signalを支持しない。

### リスク / 注意

- segment overlap weightはunique row countではない。
- 本実験は原因切り分け専用で、exp250のhard use / feature化を再開する根拠には使わない。
- threshold/slack/segment grid、near専用rule、candidate変更、ML feature化、raw-test inference、submitは禁止。

## 次

- 対応バックログを完了扱いで削除し、新規候補は追加しない。

