# exp256_segment_local_corridor_near_bucket_signal_attribution_readout セッションノート

## 目的

exp250 の 0--100 ft pooled real AUC 約 0.82 が広い GR topology signal なのか、
distance-conditioned base error、candidate family / well の構成、risk=1 飽和で
説明されるのかを保存済み Stage 1 生成物だけで切り分ける。

## 現在の状態

- Route: `pf_beam`
- 状態: Kaggle CPU train v1完了、diagnostic-onlyでbranch closure確認
- CV / LB: 対象外
- inference / submission: disabled
- 親: `exp250_segment_local_negative_space_gr_corridor_audit`

## 実装前コストガード

- active diagnostic readout: 1
- fixed variants: 2 (`real_gr`, `shuffled_typewell_gr`)
- fixed candidate family: 5
- LightGBM / CNN / HMM config: 0 / 0 / 0
- fold training / booster: 0 / 0
- PF/Beam / corridor 再生成: 0 / 0
- parent/control 再学習: なし
- threshold / slack / segment grid: なし
- runtime: Kaggle CPU、GPU/internet off、single process
- raw-test inference / submission: なし

## 固定入力

- candidate-segment: 291,710 rows / 145,855 paired keys / 773 wells
  - decompressed SHA256: `0fd6241b27fdd778e202d9f81df859fdad1dd228eeea872630261c5a30a48a9a`
- group metrics: 196 rows
  - raw SHA256: `333c3c2a869db976ca0fc42741897bb50cdad42a91fd682e40c1996626fe8d96`
- by-well: 773 rows
  - raw SHA256: `14e226cb05072b3195306b92b7587efbe6300fe6cb1da8e7b5db3aff1a4973b8`
- exp250 summary
  - raw SHA256: `79448ad77882e3ecf6fd95e0044b59965ffe8cc7fa00603459be3e786824a396`

必要な実ファイル確認のため、canonical exp250 kernel output を
`/tmp/kaggle-output/exp250-stage1-v2-full` に取得した。exp250 Stage 1やcorridorは再実行していない。

## 実装契約

- real / shuffled は `(well, segment_id, candidate)` で one-to-one pair する。
- start/end MD、bad/good weight、保存済み paired risk delta の一致を fail-closed にする。
- distance x family conditional AUC は各 stratum の `bad_weight * good_weight` で集約する。
- family x well AUC 差の寄与は `pair_mass / total_pair_mass` で加法分解し、寄与総和を再検証する。
- 0--100 ft AUC は 2 bucket の descriptive weighted mean と明記し、再計算 pooled AUC と呼ばない。
- segment overlap weight は unique row 数ではなく、exp250 contribution contract 内の相対 weight として扱う。

## コマンドログ

### 2026-07-15 実験作成

- `task` は環境に無かったため、同等の `make new-steering` / `make new-exp` を使用した。
- `.steering/20260715-exp256-segment-local-corridor-near-bucket-signal-attribution-readout/` を作成した。
- self-contained Jupytext train source と inference fail-closed guard を実装した。

### 2026-07-15 push前検証

- train/inferenceの`py_compile`、ruff、Jupytext round-trip、notebook JSON検証: PASS。
- `make validate-exp EXP=exp256_segment_local_corridor_near_bucket_signal_attribution_readout`: strict PASS。
- parent exp250 trainは11章 / 3,529行、exp256 trainは10章 / 1,200行。exp256は再生成を除いたreadout専用だが、入力preflight、paired contract、distance集計、family x well寄与、saturation、plots、SHA/metrics保存をnotebookセル上に展開した。
- canonical kernel: `kentookumura/exp256-seglocal-near-signal-attribution-train`。
- title slugとkernel idは一致。private / CPU / GPU off / internet off / run_on_push。
- kernel sourceはcanonical exp250 train 1個だけ。
- source / loose package / bootstrap `config.yaml`はbyte-identical。
- config SHA256: `b119d5cd02cfeaba2fc03dd2c13119000395b419b987edc5ee4b66657a285126`。
- bootstrap 7 filesのmanifest SHAを全件再検証: PASS。
- 実行契約を再確認: active readout 1、variant 2、candidate family 5、model config 0、fold 0、booster 0、PF/Beam再生成0、corridor再生成0、parent/control再学習なし。

### 2026-07-15 Kaggle full readout

- canonical kernel v1をpush。URL: `https://www.kaggle.com/code/kentookumura/exp256-seglocal-near-signal-attribution-train`。
- kernel id_no: `127322012`、status: `COMPLETE`。
- pull-back 23/23 cellsのsourceはlocal packageと一致。
- cell source SHA256: `9f56ec1449b4ce6c8f94e948e9f4932034c60b058e87f02774613685b1b3567c`。
- paired input contract: 291,710 rows / 145,855 pairs / 773 wells、weight / segment / risk-delta identity PASS。
- readout runtime: 6.486863秒。
- near 0--100 evaluation weightは38,299、全体比1.048546%、bad rate 0.281678。
- near pooled AUCの2 bucket weight平均はreal 0.819846 / shuffled 0.773559、差+0.046287。
- distance x family条件付け後はreal 0.598678 / shuffled 0.574742、差+0.023936。pooled realから-0.221168。
- near 10 family-bucket strata中AUC算出可能は6 strata / 4 families。0--50 ftではbeam/likpf/pf_anccのbad weightが0。
- family x well条件付きAUCはreal 0.522220 / shuffled 0.511096、差+0.011124。
- AUC算出可能2,330 strata中positiveは1,138、positive pair-mass share 0.522241。well横断のbroad signalではない。
- pooled q90はreal/shuffledとも1.0。risk=1 evaluated-weight比はreal 0.188251 / shuffled 0.270472。
- output SHA 11件、family x well寄与総和`0.011123891346780063`とAUC差`0.011123891346780046`の一致: PASS。
- summary SHA256: `c2fc5c61980a621089e153e087f04d70842cb3548b4d680beabf292e2f1502a6`。
- family x well gzip decompressed SHA256: `a7205f0e6bdec2af2c4a549fb618b538eedf844701419553ca3152e4edaeb6cc`。
- output取得先: `/tmp/kaggle-output/exp256-v1`。
- decision: `diagnostic_only_no_exp250_route_or_use_change`。
- 結論: nearの弱いGR差は残るが、pooled AUC約0.82の大部分はcandidate-family base rateへ帰属する。exp250 hard use / feature / rule / candidate変更はclosedのまま。

## 再現性

- 新規乱数なし、sorted deterministic aggregation、single process。
- gzip output は `mtime=0`、decompressed content SHA を記録する。
- fixed exp250 inputに対するdiagnostic determinismだけを主張する。
- upstream exp250 / exp072 PF/Beam cacheのstochastic provenanceを継承する。
- model / prediction / submission を生成しないため、それらの SHA は対象外。
- deterministic prediction / submission anchorではない。

## 次のアクション

1. なし。対応バックログを完了扱いで削除し、新規backlogは追加しない。
2. exp250 segment-local signalを`topk_path_confidence_features`へ混ぜない。
