# exp298_exp226_blockwise_offset_slope_quotient_local_shape_audit セッションノート

## 目的

exp226のoverall TVT errorからblockwise offsetと一次slopeを診断上だけ除去し、`tvt_geop + gr_delta`が
H256/H512の局所形状をexp293 deployable12より良く捉えるか監査する。局所sourceが成立した場合だけ、
固定local/global hybrid bankへ進む。

## 現在の状態

- Route: `pf_beam`
- 状態: Kaggle private CPU version 2完了・technical PASS・scientific FAIL・branch close
- CV/LB: diagnostic readoutのみ / 提出対象外
- implementation / active audit / LightGBM config / trained fold / booster / PF-Beam rerun:
  `1 / 1 / 0 / 0 / 0 / 0`
- evaluation fold contract: 5 folds（保存foldのreadoutのみ）
- GPU: なし
- inference/submission: 無効

## 2026-07-20 実装

- ユーザーの「exp298を実装してください」を実装承認として記録した。Notebook採用とKaggle実行の承認には拡張していない。
- exp293 compact self-contained trainの章立てとbank再構築部分を参照し、同一exp内helper importのない
  `exp298_..._compact_selfcontained_train.py/.ipynb`を別名で作成した。
- exp263 candidate-major partitionからdeployable12を同じfloat32演算順で再構築し、exp293 version 2の
  candidate content SHA `29477141...b474`を必須gateにした。
- exp293と同じblock assignmentを再構築し、gzip decompressed SHA `b0755c22...32d7`を必須gateにした。
- exp226 OOFは`well_id/fold/row_idx/suffix_offset/tvt_geop/gr_delta/tvt_pred`だけを`usecols`で読み、
  decompressed SHA `709eb726...c609`、row/fold/identity、finite coverageを確認する。
- exp226 source foldはSHA-hash割当、exp293評価foldは井戸サイズ均衡割当で一致しないため、source foldは
  provenance crosswalkとしてfreezeし、readout/fold guardにはexp293 `outer_fold`だけを使う。
- `P_geop/P_preU/P_postU`をfloat64で構築し、`P_postU`とexp293 `exp226_k16`の最大差`<=0.001 ft`を確認する。
- candidate/component/block/hidden-like/thresholdをcomponent/freeze manifestへ保存してからのみraw train truthを読む。
- blockwise offset/affine quotientはgroup sufficient statisticsから計算し、offset/slope係数や補正pathを
  row-wise生成・保存しない。長さ1 blockはfallbackせず、全候補共通でaffine評価対象から除外する。
- pooled/fold/scope/block/by-well、一次差/二次差、stable rank、strict unique-best比率、固定PASS判定を実装した。
- `P_postU`はpaired diagnosticだけに出力し、rank/block winner bankには重複投入しない。
- fail-closed inference候補を別名で作成し、raw-test inference/submissionが起動しないことをtestした。
- 実装段階では既存の正規train/inference `.ipynb`を上書きせず、別名候補として検証した。
- 親compactとの比較: exp293 trainは8章/約1,960行、exp298 train候補は8章/約2,190行で、
  runtime、bank再構築、component/freeze、truth loader、quotient readout、decision、生成物保存をNotebook上に展開した。
- 実行契約は`1 audit / 0 LightGBM config / 5 evaluation folds / 0 trained folds / 0 boosters /
  0 PF-Beam reruns`。親/control再学習はない。

## 2026-07-20 正規Notebook採用・Kaggle実行承認

- ユーザーの「実行してください」を、compact self-contained train候補の正規train Notebook採用と、
  Kaggle private CPUで固定1 auditをpush・実行する明示承認として記録した。
- 実行対象は`1 audit / 0 LightGBM config / 5 evaluation folds / 0 trained folds / 0 boosters /
  0 PF-Beam reruns`。親/control再学習、GPU、internet、inference、submissionはない。
- compact train候補を正規`exp298_..._train.ipynb`へ採用する。正規inference Notebookは変更せず、
  inferenceをfail-closedのまま維持する。
- Kaggle credential preflightはAPI Token未設定、OAuth credentialとlegacy credential有効。Kaggle CLIは
  OAuth credentialで実行する。credential実値は記録しない。
- 初回package準備は、`runtime.kaggle.bootstrap_files`をリポジトリ起点で指定していたため、実験dirが
  二重に連結されてKaggle送信前に停止した。監査ロジックや入力契約は変更せず、実験dir相対の
  `downstream_branch_contract.md`へ修正して再準備する。
- Kaggle private CPU version 1（kernel id
  `kentookumura/exp298-exp226-local-shape-quotient-audit-train`、id_no `127956072`）は監査本体を完了した。
  bank/component/truth freezeを通過し、H256/H512 primaryを出力、生成物保存後、最後の表示だけが存在しない
  `SUMMARY["support"]`を参照して`KeyError`となりkernel statusはERRORだった。
- version 1の暫定readoutはH256 affine RMSE `0.3482675904552506`・rank 4・post-U
  `0.3041197991208009`・unique-best `0.11363037173741102`、H512 affine RMSE
  `0.7224085770990538`・rank 5・post-U `0.6096467778987567`・unique-best
  `0.11547848426461144`。`audit_passed=false`、branch closeだった。
- 監査計算、入力、閾値、保存内容は変更せず、最終表示を`SUMMARY["decision"]`へ直し、同じkernel IDへ
  version 2をpushして完走させる。version 1はERRORのため最終結果として確定しない。

## 2026-07-20 Kaggle private CPU version 2 完了

- Kernel: `kentookumura/exp298-exp226-local-shape-quotient-audit-train` version 2、id_no `127956072`、
  status `COMPLETE`。監査判定出力は259.830秒、Kaggle log末尾は269.929秒だった。
- 3,783,989 rows / 773 wells、1 audit / 0 LightGBM config / 5 evaluation folds / 0 trained folds /
  0 boosters / 0 PF-Beam reruns、CPU/GPU/internet=`CPU/off/off`で完走した。
- row/well/fold/candidate order、bank/block SHA、finite coverage、allowlist、truth-before-freeze=0、post-U alias、
  offset coverage、affine-eligible coverage、長さ2以上invalid 0、singleton全候補共通除外、oracle係数/
  補正prediction非保存を含むtechnical guardは全PASSした。
- singleton除外はH128/H256/H512/whole-wellで`4/2/2/0 blocks`、各row/well数も`4/2/2/0`。
  affine-eligible row coverageは全候補・全対象で1.0だった。
- `P_preU`のH256はaffine RMSE `0.3482675904552506`、rank 4、post-U
  `0.3041197991208009`、strict unique-best `0.11363037173741102`。H512は
  `0.7224085770990538`、rank 5、post-U `0.6096467778987567`、strict unique-best
  `0.11547848426461144`。
- fold rankはH256が全fold 4位、H512が全fold 5位でtop3は各0/5。H512の1000+、hidden-like spatial、
  hidden-like typewell-purgedもすべて5位だった。
- unique-best比率だけは閾値0.05を通過したが、残るscientific guardはすべてFAIL。technical PASS /
  scientific FAIL / `audit_passed=false`、`branch_closed_without_rescue_grid`を最終判断とする。
- 小規模metrics/manifestだけを選択取得し、Kaggle SHA manifestに記載された取得8ファイルのfile SHAを全件照合した。
  candidate bank content `29477141...b474`、component content `41390811...f10`、block decompressed
  `b0755c22...32d7`、truth content `e9067327...a8d0`、readout content `9dec97ae...4edd`、
  SHA manifest file `25cbecf9...a8e1`を記録した。
- version 1とversion 2のH256/H512 primary、bank/component/truth SHA、判定は完全一致した。

## 2026-07-20 singleton契約改訂（ユーザー承認）

- 保存済みexp226 OOFのwell別suffix行数だけを読み、最終block長が1になるwell数を確認した。
- H128は4 wells、H256は2 wells、H512は2 wells。whole-wellは最小407 rowsでsingletonなし。
- 1行ではinterceptとslopeを同時に識別できないため、ユーザー承認により、exp293のblock ID・境界・SHAを
  変更せず、singletonをaffine RMSE/rank/block win/strict unique-bestの分母から全候補共通で除外する。
- technical coverage 1.0はselected row数2以上のaffine-eligible rowsに対して要求し、長さ2以上でinvalidな
  blockが1件でもあればfallbackせずtechnical FAILとする。
- singleton block/row/well数はpooled/fold/scope/block/by-well/summaryへ記録する。offset-only secondary
  readoutとscientific PASS閾値は変更しない。
- これにより、singletonの存在だけでfull PASS不能になる構造上の停止条件は解消した。scientific PASS/FAILは
  Kaggle private CPU auditの実測結果で判定する。

## 2026-07-20 設計確定

- 次の空き番号`exp298`を確認した。
- `.steering/20260720-exp298-exp226-blockwise-offset-slope-quotient-local-shape-audit/`を先に作成した。
- templateから実験ディレクトリを作成した。自動生成Notebookは未編集・non-canonicalである。
- primary componentを`exp226_pre_u = tvt_geop + gr_delta`へ固定した。
- `tvt_geop`と`tvt_pred`はpaired diagnostic、`tvt_pred`はexp293 `exp226_k16`のaliasとした。
- exp293 version 2のdeployable12 content SHAとblock assignment SHAを比較契約に固定した。
- H256/H512 affine quotientをprimary、H128/whole-well/offset-only/差分metricをsecondaryへ固定した。
- candidate/component/blockのtarget-free freeze後だけtruthを別loaderで接続する順序を固定した。
- oracle offset/slope係数と補正predictionの保存・後続利用を禁止した。
- PASS時はStage 2だけ、FAIL時は救済gridなしでbranch closeとした。
- `downstream_branch_contract.md`に案2・3・4の設計と依存順を固定した。
- downstream contract SHAを`config.yaml`へ記録し、singleton契約改訂後の値へ同期した。
- steering requirements/design/tasklist SHAも`config.yaml`へ記録した。
- Lateフェーズ固有の分岐は対象外とした。

## 固定PASS要約

`exp226_pre_u`について次を全必須とする。

- H256/H512 pooled affine-quotient rankが各3位以内、少なくとも一方で1位。
- 各horizonで4/5 folds以上が3位以内。
- H256/H512でpost-U非悪化。
- H512の1000+とhidden-like 2面が各3位以内。
- H256/H512の少なくとも一方でstrict unique-best block比率`>=0.05`。
- row/fold/component/bank/block/truth freeze/finite/alias parityのtechnical guard全PASS。

## 後続契約要約

1. exp298 PASS時だけStage 2でfixed `S512`、原12 + hybrid12の24候補bankを監査する。
2. Stage 2 PASS時だけStage 3で24候補×registration×reliabilityのsemi-Markov posterior meanを作る。
3. Stage 3が不足しStage 2 supportが強い場合だけ、別承認後にStage 4 outer5×inner4 nested rankerを検討する。
4. exp298またはStage 2 FAILならbranchを閉じる。Stage 4へ自動分岐しない。
5. exp293/exp297 fixed12とexp295 candidate-free SSMは独立のまま変更しない。

詳細と数式は`downstream_branch_contract.md`を正とする。

## コマンドログ

### 実行済み

```bash
make new-steering EXP=exp298_exp226_blockwise_offset_slope_quotient_local_shape_audit
make new-exp EXP=exp298_exp226_blockwise_offset_slope_quotient_local_shape_audit
.venv/bin/pytest -q tests/test_exp298_exp226_blockwise_offset_slope_quotient_local_shape_audit.py
.venv/bin/ruff check experiments/exp298_exp226_blockwise_offset_slope_quotient_local_shape_audit/exp298_exp226_blockwise_offset_slope_quotient_local_shape_audit_compact_selfcontained_train.py experiments/exp298_exp226_blockwise_offset_slope_quotient_local_shape_audit/exp298_exp226_blockwise_offset_slope_quotient_local_shape_audit_compact_selfcontained_inference.py tests/test_exp298_exp226_blockwise_offset_slope_quotient_local_shape_audit.py
make validate-exp EXP=exp298_exp226_blockwise_offset_slope_quotient_local_shape_audit
make validate-template
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp298_exp226_blockwise_offset_slope_quotient_local_shape_audit/exp298_exp226_blockwise_offset_slope_quotient_local_shape_audit_compact_selfcontained_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp298_exp226_blockwise_offset_slope_quotient_local_shape_audit/exp298_exp226_blockwise_offset_slope_quotient_local_shape_audit_compact_selfcontained_inference.py
make prepare-kaggle-notebooks EXP=exp298_exp226_blockwise_offset_slope_quotient_local_shape_audit EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp298-exp226-local-shape-quotient-audit-train --title 'exp298 exp226 local shape quotient audit train' --run-on-push --strict"
make push-kaggle-train EXP=exp298_exp226_blockwise_offset_slope_quotient_local_shape_audit
kaggle kernels pull kentookumura/exp298-exp226-local-shape-quotient-audit-train -p /tmp/exp298_kaggle_pull_v2 -m
kaggle kernels logs kentookumura/exp298-exp226-local-shape-quotient-audit-train
kaggle kernels output kentookumura/exp298-exp226-local-shape-quotient-audit-train -p /tmp/exp298_kaggle_output_v2 --file-pattern '^(metrics\.json|exp298_local_shape_(summary\.json|sha_manifest\.csv|pooled_metrics\.csv|fold_metrics\.csv|scope_metrics\.csv|component_manifest\.json|freeze_manifest\.json|contract\.json|input_manifest\.csv))$'
kaggle kernels output kentookumura/exp298-exp226-local-shape-quotient-audit-train -p /tmp/exp298_kaggle_output_v2 --file-pattern 'exp298_local_shape_(summary\.json|sha_manifest\.csv|pooled_metrics\.csv|fold_metrics\.csv|scope_metrics\.csv|component_manifest\.json|freeze_manifest\.json|contract\.json|input_manifest\.csv)'
```

### 未実行

- local audit、inference、submissionは実行していない。Kaggle auditだけを実行した。
- 親/controlの再学習、PF/Beam再生成も実行していない。

### 検証結果

- exp298専用test: `11 passed`
- Ruff: pass
- Jupytext train/inference round-trip `--test`: pass
- `make validate-exp`: strict pass
- `make validate-template`: pass
- repository全体: `342 passed / 1 skipped / 2 failed`。失敗2件は既存exp296の完了後configと旧test期待の
  不一致で、exp298外のため変更していない。

## 再現性メモ

- seed policy: RNGなし、fixed fold/well/row/candidate order
- stochastic components: なし
- CPU/GPU runtime: Kaggle private CPU single process、GPU/AMP/internet off。監査判定259.830秒
- input SHA: exp226 OOF decompressed `709eb726...c609`、exp293 bank content `29477141...b474`
- block SHA: exp293 decompressed `b0755c22...32d7`
- component SHA `41390811...f10`、truth SHA `e9067327...a8d0`、readout SHA `9dec97ae...4edd`
- model/prediction/submission SHA: 生成しない
- deterministic anchor: fixed-input diagnosticでありsubmission anchorではない

## 次のアクション

1. 固定契約どおりlocal/global decomposition枝を閉じ、Stage 2/3/4へ進まない。
2. component/horizon/quotient/scope/平滑化/weightの救済gridと、本結果起点の新規backlogを作らない。
3. inference/submissionは無効のまま維持し、exp293/exp297 fixed12とexp295独立SSMを変更しない。
