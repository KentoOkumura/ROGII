# exp361_exp333_candidate_path_addone_novelty_audit セッションノート

## 目的

exp333 Stage 1 OOF を exp293 fixed deployable12 へ1本だけ追加し、exp302 と同じ
H512 / whole-well add-one novelty 契約で候補パスとしての価値を再評価する。

## 現在の状態

- Route: `ensemble`
- 状態: Kaggle CPU version 2完了、technical PASS、candidate novelty PASS
- direct参考値: exp226 `9.4271095966`、exp333 `9.0766766609`、
  exp228 `8.9440855008`、exp263 `8.238331715`
- inference/submission: 未承認・対象外

## 比較役割

- 一次 control: `exp226`。exp333 の直接改善と fold identity を確認する。
- 二次 ablation: `exp228`。row-wise residual と segment residual の文脈比較だけに使う。
- 固定 blend 参考値: `exp263`。単体置換 hard gate にはしない。
- 科学的 gate: `exp293` fixed12 に対する exp333 add-one novelty。

## 実行量（push前固定）

| 項目 | 数 |
| --- | ---: |
| scientific candidates | 1 |
| reporting folds | 5 |
| candidate-fold readouts | 5 |
| LightGBM configs | 0 |
| trained folds | 0 |
| boosters | 0 |
| parent/control regeneration | 0 |
| GPU | 0 |

## 2026-07-23 設計・実装

- ユーザーの「それで進めてください」を、0-booster readout の実装と Kaggle CPU audit
  1回の承認として記録した。inference/submissionの承認には拡張しない。
- `docs/legacy/steering/20260723-exp361-exp333-candidate-path-addone-novelty-audit/`
  を先に作成した。
- exp302 の検証済み fixed-bank reconstruction、block assignment、freeze/late-truth join、
  novelty readout を土台にした。
- exp333 OOF は `well_id,row_idx,outer_fold,tvt_pred_stage1` の
  target-free allowlistだけを pre-freeze に読み、同居する `tvt_true` は開かない。
- exp333 OOF file / decompressed SHAをhard固定し、学習時logical prediction SHAは
  upstream evidenceとして記録した。
- fixed12 candidate bank content SHA
  `29477141685662bae7417e788ec5dbe914c2220b7f4cc45ab01befa4e5e3b474`
  と block decompressed SHA
  `b0755c22aa8d791012d3f605e2f1b66063ce9bb6ba46ddd4b48dca77cce032d7`
  を truth 前に照合する。
- direct readout は保存値 parity の technical/context checkだけにし、科学的判定から除外した。
- PASS は `exp333_candidate_path_novelty_supported`、FAIL は
  `close_exp333_candidate_novelty_branch` とする。

## 再現性メモ

- seed policy: 新規乱数なし。保存済み exp226/exp333 fold identity を再利用。
- runtime: Kaggle CPU、1 process、GPU false、internet false。
- exp333 OOF file SHA:
  `70b623d4c839c4f7eb11fb2134aa214ca8f0ce8d6ebe65e723d2fffa95dcc2dc`。
- exp333 OOF decompressed SHA:
  `f2ebc6f6ea243b45fdb785342b8815b3b04947f96d787d3017e5e2be7ff92e5a`。
- exp333 logical prediction SHA:
  `dbb3f41642a2d6a9da704d276ed6398b706059078bcfcaca95e17e5c7af00784`。
- このSHAは学習時のin-memory DataFrame row-hashであり、CSV再読込後のdtypeをまたぐ
  hard contractには使わない。upstream evidenceとして記録し、実ファイル同一性は
  file/decompressed SHA、再読込後の候補はexp361 prediction content SHAで固定する。
- model/submission SHA: 新規 model/submission を生成しないため対象外。
- oracle prediction: 保存しない。

## 未実行

- inference
- submission

## 2026-07-23 Kaggle version 1

- Kernel: `kentookumura/exp361-exp333-candidate-novelty-train` version 1。
- 約162秒で、exp333 OOFの再読込後pandas row-hash
  `b6b50a1e...68de45c`が学習時in-memory hash `dbb3f416...00784`と一致しないため
  fail-closed した。
- file SHA / decompressed SHA は一致した。candidate bank構築後、exp333 loader内で停止し、
  freeze、truth load、oracle集計には未到達。model/booster/inference/submissionは0。
- pandas row-hashはCSV serialization/dtypeをまたぐ永続契約ではないため、upstream SHAは
  evidenceへ降格し、file/decompressed SHAとexp361のpost-read content SHAをhard guardとする。
  科学contract、candidate、block、閾値、実行量は変更しない。

## 2026-07-23 Kaggle version 2 完了

- 同一kernel `kentookumura/exp361-exp333-candidate-novelty-train` version 2を完了した。
- audit完了`234.279773 sec`、最終log`242.406767 sec`。CPU、internet/GPU off。
- 実行量は1 candidate / 5 reporting folds / 5 readouts / LightGBM config 0 /
  trained fold 0 / booster 0 / parent-control regeneration 0。
- technical guardは全PASS。3,783,989 rows / 773 wells、finite 1.0、duplicate 0、
  exp333/exp226 outer-fold parity、candidate bank/block SHA、direct metric parity、
  evaluation truth access before freeze 0を確認した。
- directはgate無効の参考値としてexp333`9.0766766609`、exp226差`-0.3504329356 ft`、
  5/5 folds改善を再現した。exp228/exp263はhard gateにしていない。
- exp293 fixed12へのadd-one novelty:
  - H512 oracle `3.6837626642 -> 3.5506587880`、`+0.1331038762 ft`。
  - whole-well oracle `4.7849038814 -> 4.6827715422`、`+0.1021323391 ft`。
  - H512 strict unique-best `0.1150635675`（896 / 7,787 blocks）。
  - H512 fold改善 `+0.2471545100 / +0.1866714478 / +0.0812791794 /
    +0.0727524629 / +0.0932954333 ft`、5/5。
- 全4 novelty checksがPASSし、decisionは
  `exp333_candidate_path_novelty_supported`。
- exp333 source file/decompressed SHAは固定値と一致。post-read prediction content SHA
  `e9bb5e9f0689facf7d7aa468dda89ca776c11bbec1c97cc02f74ab992a016450`、
  保存gzip decompressed SHA
  `ed7b7a7f281b5d8d0b43b7007a5f640408312f4ed08abd1e417a3372bccd6bff`。
- candidate bank SHA `2947714168...b474`、block decompressed SHA
  `b0755c22aa...32d7`、truth content SHA `e906732705...a8d0`。
- 大容量prediction/blockはダウンロードせず、summary/metrics/readout/manifest等だけ取得した。
  取得したmanifest対象10件のfile SHAは10/10一致した。
- inference/submissionは実行していない。PASSはexp333 current-test候補生成を別承認で
  same-exp実装する価値だけを支持し、単独採用やselector性能を保証しない。

## 次のアクション

1. ユーザーが承認する場合だけ、exp333内へcurrent-test candidate inferenceを実装する。
2. 14,151行のcandidate artifact、feature/model SHA、fold ensemble parityを検証する。
3. submissionは作らず、固定12への組み込み方を別のtarget-free readoutとして設計する。
