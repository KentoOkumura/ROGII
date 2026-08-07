# exp354_typewell_group_candidate_family_prior_readout セッションノート

## 目的

旧exp316の中核仮説をexp311/312/313/315から切り離し、固定exp293 candidate familyだけを使う
0-model Stage 0として、Type Well群×family error priorのheld-out transferを判定する。

## 現在の状態

- Route: `ml_model`
- 状態: Kaggle CPU version 1完了、fixed real-minus-shuffle gate FAIL、branch closed
- CV: Stage 0 9/10 checks PASS、総合FAIL
- LB: なし

## コマンドログ

- 2026-07-23: ユーザー承認により、旧exp316をreopenしない独立後継scaffoldを作成した。
- 2026-07-23: ユーザーの「exp354を実装してください」をStage 0実装承認として受領した。
- 2026-07-23: compact self-contained train/inference候補とcontract testsを実装した。
- 2026-07-23: 専用test 7件、py_compile、ruff、Jupytext、strict validationをPASSした。
- 2026-07-23: ユーザーの「実行してください」を、compact train候補の正規Notebook採用、
  Kaggle CPU package/push/run、Stage 0完了監視までの承認として受領した。
- push前確認: real prior 1、negative control 1、LightGBM config 0、reporting fold 5、
  trained fold 0、合計booster 0。candidate/PF/Beam再生成0、親control再学習0。
- 初回pushは50文字超のslug/titleでKaggle API `400 Bad Request`となり保存前に拒否された。
  科学契約を変えず短縮し、再packageした。
- canonical kernel:
  `kentookumura/exp354-typewell-family-prior-readout-train`。
- title: `exp354 typewell family prior readout train`。
- `make prepare-kaggle-notebooks ... --run-on-push --no-src --strict`: PASS。
- execution config / package config SHA:
  `a9495a73e25d2fb654d7723fb5bfba5cc69cb4f08c5acd6142b9921f602afeb9`で一致。
- `make push-kaggle-train EXP=exp354_typewell_group_candidate_family_prior_readout`:
  version 1 push成功。
- `kaggle kernels pull ... -m`: id_no `128363177`、private、CPU、internet off、
  exp293/exp065/exp115 kernel sourcesを確認。package/pullの21 cell source content SHAは一致。
- `kaggle kernels logs -f ...`: 正常完了、Stage 0 gate FAILを確認。
- SHA実ファイル確認のためoutputを取得し、
  `kaggle/output/train_v1`へ約1.2 MBのStage 0生成物を保存した。
- 完了後のexp293親契約+exp354 testは18 passed、Jupytext `--test`、
  metrics JSON parse、strict experiment validationもPASSした。

## Kaggle Stage 0結果

- diagnostic runtime: `63.672340 sec`。
- target-free input freeze前truth rows: `0`。
- fit-valid well overlap: `0`。
- held-out group coverage: `0.9805950841`。
- real family rank Spearman: `0.3257889189`、5/5 folds正方向。
- shuffle family rank Spearman: `0.3270790744`。
- real minus shuffle: `-0.0012901555`、固定下限`+0.05`をFAIL。
- hidden-like spatial / typewell-purged:
  `0.3817361922 / 0.3765703933`、両方PASS。
- 10 checks中9 PASS。real-minus-shuffleだけFAILしたため
  `stage_0_failed_close_without_rescue`、Stage 1 eligible=false。

## 生成物SHA確認

- 期待したStage 0 artifact 12件を取得した。
- manifest対象11件のraw SHAと、gzip 3件のdecompressed content SHAを実ファイルで照合した。
- target-free input freeze:
  `b1920c1eb6a855201a91ede193eabdf0fdeead959afb560a455a65c27bb527cd`
- prior schedule freeze:
  `f64975ec2335b8eb5f1bf3e03d8e8a4d3314a9b878219941ded66be583b173f2`
- prior schedule content:
  `1d0f0073ab5a1f997bde06e1fa7366088d56a21fa507083ac8f39db8590d2585`
- readout content:
  `4c384076578107fbff22268222dba4559c09227cd6e8d6f2b7ad6c9f8157b7a8`
- well-family error content:
  `9afa8a83520ef511c3fc721bc2441f8511516433ff675fc2112b03d105d3f8a1`
- model manifest / prediction / submission SHA: 非該当。
- rerun parity: 未実行。

## 解釈

- real prior単体のrank signalはあるが、stable shuffleで消えない。
- native Type Well群の増分signalではなく、global family base rateが主成分と考える。
- 同じreadoutでfamily/support/group/rank metricを救済しない。
- exp353 quality featureは別仮説として残すが、本結果による自動昇格はしない。

## 実行コスト契約

- Stage 0: prior 1 + stable group-label shuffle 1 / reporting fold 5。
- model config / trained fold / booster: `0 / 0 / 0`。
- candidate/PF/Beam再生成、親control再実行: `0 / 0`。
- Stage 1予約40 selector modelsは未実装・未実行。

## 実装SHA

- config:
  `7a2618b94a8ec43ad49d9e8f4af95f354fbd3bca2605fe6945f7aa663ec4e6f4`
- compact train source:
  `f9ebe460200cac04b5372a836d27f23ed547178ca4bfca35137a1dc16372bde3`
- compact inference source:
  `1808552656afa2900797ac639fb40bf3a3f72419865b7dc2de9c7c624759b872`
- contract test:
  `23b0e5dfe7137c9a2ce2f76bd108919172246636ab81384019896ba95d0f9fe8`
- canonical train notebook:
  `af361dcdcd00d8a7033bb8f5c4b3e3052685a5a0321b4fed051a68586ffe5a88`
- compact train notebook:
  `80f99a2c68cc1ba050c23105dc5855b435aa0a61cbe9ea89c0f53e31c4618ef4`
- compact inference notebook:
  `cae928c95f8f821da6b1813f4705e2aeba229bf7dd949e739c6078078100c984`

## 次のアクション

1. exp354を救済grid、Stage 1、inference、submission、再実行なしで閉じる。
2. exp353を判断する場合も、exp353自身のfixed group-shuffle gateを正とする。
3. 次の実験は既存backlogの独立仮説から選ぶ。
