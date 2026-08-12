# 要件

## 依頼

局所的な変化と大局的なトレンドを分ける方針の第1段階として、exp226が局所形状を十分に捉えているかを
blockwise offset / slope quotientで監査する実験を設計する。`KAGGLE_DIRECTION.md`のバックログ、
`experiments/exp298_exp226_blockwise_offset_slope_quotient_local_shape_audit/`、このsteeringを作成し、
設計だけを確定する。実装、Notebook採用、Kaggle実行、推論、提出はまだ行わない。

また、後続案2・3・4が別セッションで変質しないよう、exp298配下に正規分岐契約を残す。

## 制約

- Route: `pf_beam`。保存済み物理候補の0-booster監査であり、ML predictor/selectorは使わない。
- 親: `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction`。
- 比較bank: exp293 version 2でSHA固定・support PASSしたdeployable12。
- primary component: `exp226_pre_u = tvt_geop + gr_delta`。`tvt_geop`と`tvt_pred`は診断比較に限定する。
- primary horizon: H256/H512。H128とwhole-wellはsecondary readout。
- block assignment、fold、row identityはexp293を再利用し、新しい境界を作らない。
- candidate/componentとblock assignmentをtruthなしでSHA freezeしてから、別loaderでtrue suffix TVTを接続する。
- oracle offset/slopeは局所形状の診断用nuisanceであり、補正predictionやfeatureとして保存しない。
- Lateフェーズ固有の設計は対象外とする。
- 親/controlの再学習、PF/Beam再生成、候補追加、weight調整、平滑化gridを行わない。
- 再現性は`docs/06_reproducibility.md`に従い、入力、schema、content、bank、block、freeze、readout SHAを記録する。

## 受け入れ基準

- steeringのrequirements/design/tasklistに、仮説、対象成分、数式、freeze順序、PASS/FAIL、禁止事項がある。
- exp298の`config.yaml`、`README.md`、`SESSION_NOTES.md`、`result.md`、`metrics.json`がdesign-only状態を示す。
- `downstream_branch_contract.md`に案2・3・4の開始条件、固定入力、出力、合格条件、禁止事項、分岐順がある。
- `KAGGLE_DIRECTION.md`の未着手バックログにexp298が追加され、契約pathが記載される。
- `experiment_summary.md`にexp298の計画行が追加される。
- `make validate-exp EXP=exp298_exp226_blockwise_offset_slope_quotient_local_shape_audit`と
  `make validate-template`が通る。
- 実装source、Jupytext候補、test、Kaggle package、artifact、prediction、submissionを新規作成していない。

## 2026-07-20 実装時の承認済み改訂

実装後の入力preflightで、exp293固定block assignmentには最終block長1がH128/H256/H512で
`4/2/2 wells`存在すると判明した。1行ではoffsetとslopeを同時に識別できないため、ユーザー承認により
次を固定契約へ追加する。

- exp293のblock ID、境界、horizon、final-short-block inclusionは変更しない。
- affine quotientの評価対象はblock内selected row数が2以上のblockだけとする。
- 長さ1のblockはaffine RMSE、affine rank、block win、strict unique-bestの行数・block数・分母から除外する。
- technical coverage 1.0は「selected row数2以上のaffine-eligible rows」に対して要求する。
- 長さ1のblock数・行数・well数を必ず生成物とsummaryへ記録する。
- 長さ2以上のblockでaffine invalidが1件でもあればtechnical FAILとし、fallbackしない。
- offset-only secondary readout、exp293 block assignment SHA、候補値、PASS rank閾値は変更しない。
