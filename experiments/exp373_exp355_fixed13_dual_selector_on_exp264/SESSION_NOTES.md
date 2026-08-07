# exp373_exp355_fixed13_dual_selector_on_exp264 セッションノート

## 目的

exp355 Stage 1 direct HMM OOFを、既存`exact_hmm`と置換せずcorrected exp264
fixed12へ13本目として追加するdual selector実験を実装する。

## 現在の状態

- Route: ensemble
- 状態: Kaggle CPU version 1完了 / scientific gate FAILでfail-close
- CV: fixed13 hard OOF RMSE `8.695437630439221`
- LB: まだなし
- 親: corrected exp264 Stage C
- candidate parent: exp355 Stage 1 direct HMM
- active variant / objectives / outer / inner: `1 / 2 / 5 / 4`
- planned / trained CPU selector boosters: `40 / 40`
- parent/control再学習: `0`
- GPU / downstream TVT / inference / submission: `0 / false / false / false`
- 実装承認: 2026-07-24のユーザー指示「exp373を実装してください」
- Kaggle run承認: 2026-07-24のユーザー指示「実行してください」
- 正規notebook採用: 同指示によりcompact trainとfail-closed inferenceを正規名へ採用

## コマンドログ

実行したコマンドを時系列で記録します。未実行のコマンドは予定として明記します。

- `make new-steering EXP=exp373_exp355_fixed13_dual_selector_on_exp264`
- steeringのrequirements/design/tasklistを記入。
- `make new-exp EXP=exp373_exp355_fixed13_dual_selector_on_exp264 SOURCE=templates/experiment`
- ユーザー指示によりexp371を構成参照元として実装を開始。
- exp355 Stage 1 SHA manifestからOOF契約を確認:
  raw `28da6ffb17300f7757d51496f2dc56402d477fc5a79e24dec7514e855c960a41`、
  decompressed `3c49f25e138f94c9e09fb551f199fa4f92b0d776899485e67e61e2fcdb83ede3`、
  upstream logical prediction
  `634303f022bced6685367094304da6182fee42815302344469b5919a36cd5e21`。
- `src/exp355_fixed13_candidate_cache.py`へtarget-free loader、SHA resolver、
  global key join、selector-fold repartition、fixed13 readoutを実装。
- candidate/feature contract、config、compact self-contained train、
  fail-closed inference、専用testを実装。
- `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb ...`
  でtrain/inference候補を変換。
- `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test ...`
  をtrain/inferenceともPASS。
- `.venv/bin/python -m py_compile ...`をPASS。
- `.venv/bin/ruff check ...`をPASS。
- `.venv/bin/pytest -q tests/test_exp373_exp355_fixed13_dual_selector.py`:
  `9 passed`。
- exp371 / exp373 / exp264 / exp355の関連回帰:
  `51 passed`。
- `task validate-exp ...`はtaskコマンド未導入で実行不能。
  同等の`make validate-exp EXP=exp373_exp355_fixed13_dual_selector_on_exp264`を
  実行し、strict validation PASS。
- 実装完了時点ではKaggle package、push、runは未実行。
- 2026-07-24、ユーザーが正規notebook採用とKaggle CPU trainを承認。
  実行範囲は`1 variant / 2 objectives / outer 5 × inner 4 =
  40 CPU boosters / control再学習0 / GPU 0 / downstream TVT 0 /
  inference 0 / submission 0`。
- 当初のcanonical kernel id/titleを
  `kentookumura/exp373-exp355-fixed13-dual-selector-on-exp264-train` /
  `exp373 exp355 fixed13 dual selector on exp264 train`とした。
- push前の同slug pullは403、既知のprivate exp371 pullは成功したため、
  認証不良ではなくexp373 canonical kernel未作成と判断。
- 承認反映後の関連回帰は、実装時の`run_approved=false`を固定していた専用test
  1件だけFAIL。科学契約や学習コードではなく承認状態の期待値更新漏れのため、
  `run_approved=true`かつ`approval_consumed=false`、trained booster 0を確認する
  push前testへ更新した。
- compact train / fail-closed inferenceを正規notebookへ採用し、
  canonical SHAをそれぞれ
  `0d30600fabd2ddd3b59fd2635554ff1abf5c3588b7c35dd1a9cd5f70ed323716` /
  `22ffef103a83b575d1c219942f9676b971fef47cbe936538215d91de0c6ac06f`
  と確認した。
- 承認反映後に専用9 tests、関連回帰51 tests、Ruff、Jupytext test、
  template / strict experiment validationを再実行してPASS。
- strict Kaggle train packageを生成し、private / CPU / internet off /
  run-on-push、canonical id/title、3 kernel sources、26 support files、
  bootstrap内configとlocal configのbyte一致を確認した。
- 初回400拒否packageのpush直前SHA:
  embedded config
  `3f534caa05362b5040fa7430d15a87b27c4df1f2633bedcee483b698d93f7ed1`、
  support ZIP
  `5fa228da69e67c54acf37e316082e85bc6f0ff798c58e40cef2a90f5427dee19`、
  metadata
  `0d0eaab8deab7f29379b2a67c7fcaebfe5d72b99ba2eba0c0ed5cdf33f402132`、
  packaged notebook
  `009f2af5f25d18c92130f6a71c02b46ee7f2814820e955010848b9fd056b257d`。
- 上記packageの初回pushはKaggle SaveKernel APIの400で拒否され、runは未開始。
  既知のprivate exp371取得とexp355 source file一覧取得は成功したため、認証や
  kernel source欠損ではない。初回slug/titleがともに51文字で、成功済みexp371の
  短縮slug規約から外れていたため、科学条件を変えずcanonical id/titleだけを
  `kentookumura/exp373-exp355-fixed13-selector-train` /
  `exp373 exp355 fixed13 selector train`へ短縮してpackageを再生成する。
- 短縮slug packageのpushに成功。2026-07-24 04:42:32 UTC時点でversion 1、
  Kaggle id_no `128435229`、status `KernelWorkerStatus.RUNNING`。
  approvalはこのversion 1で消費済みとし、空ログやstatusの一時障害を理由に
  再pushしない。
- 実際にversion 1へpushした短縮slug package SHA:
  embedded config
  `59524e5b7921aa0f47cd00df12c1255461a87cbfe24ed361d405e10406815ec6`、
  support ZIP
  `dbd0096ecd739d5c8057cbff441d84bc00d26591c7208d44c14d82e588536507`、
  metadata
  `e68617cbe856dc3dd3200182c2e769f3933421ce9241d4366d3c77098308ad6c`、
  packaged notebook
  `8e81f283be6402110d9020677adada06550f6276e7aa0d5d568aab9e73c788cc`。
- version 1は`6350.504163695 sec`で`KernelWorkerStatus.COMPLETE`。
  `40/40 CPU boosters / control再学習0 / GPU 0 / downstream TVT 0 /
  inference 0 / submission 0`を確認した。
- Stage Aは650,000 rowsを監査し、153特徴から90特徴をfreeze、compact 77特徴。
  feature schema SHAは
  `908043637ab6af5033d6ae95be0c4505f9e68a7de9f07d857a94e2665a477b8d`。
- Stage Cは40 models / 25 partitions / 18,919,945 compact rows /
  49,191,857 outer-valid candidate-score rows。score guard、leakage audit、
  input/fold technical checksを全PASSした。
- fixed13 hard OOFは`8.695437630439221`、親fixed12
  `8.652531955610227`比`+0.04290567482899377 ft`で悪化し、改善は2/5 folds。
  exp355 top-1 usageはpooled`0.12319221858203076`、5/5 foldsで正。
- hidden-like spatial / typewell-purgedは親比
  `-0.13752472960431916 / -0.12512694887793963 ft`改善したが、
  near / 1000+は`+0.021578925223943557 / +0.04334074320789405 ft`悪化。
  by-well p95は`+1.0082613946415124 ft`、worst `b19b0395`は
  `+29.062586652438384 ft`でscientific gate FAIL。
- decision=`FAIL_CLOSE_FIXED13_SELECTOR_BRANCH`。小型JSON/CSV/manifest
  21件とlogsだけを取得し、巨大parquet、25 partitions、40 model filesは
  ダウンロードしなかった。gate成果物4件とmodel/compact manifestの
  Kaggle summary SHAがローカル取得SHAと一致した。
- 完走状態へlifecycle testを更新後、専用9 tests、関連回帰51 tests、Ruff、
  strict experiment validation、`metrics.json`構文確認を再実行してPASS。

## 変更点

- added candidate: `exp355_dip_rate_hmm`
- 既存`exact_hmm`と派生formulaを維持し、primary domainだけを11→12候補へ増やす。
- fixed fallbackは7候補のまま。
- add-one novelty監査はユーザー判断により省略する。
- exp355 OOFの読み込み列は`well_id,row_idx,fold,candidate_tvt`だけ。
- exp355 `fold`はsaved-exp226 OOF provenanceとしてのみ保持し、model featureには使わない。
- 3,783,989行をglobal key join後にexp263 selector foldへrepartitionする。
- Stage Aは13候補でschemaを再freezeし、Stage Cだけを40 CPU boosters学習する。
- 親exp264 scoreはfit後のpaired readout専用で、selector featureには入れない。

## 再現性メモ

- seed policy: stage/fold/objectiveのimmutable keyからstable SHA256 seedを生成
- stochastic components: selector sampling / LightGBM CPU training
- CPU/GPU runtime: Kaggle CPU、GPU/internet off
- Kaggle kernel id / version:
  `kentookumura/exp373-exp355-fixed13-selector-train` version 1、
  id_no `128435229`
- input evidence: exp355 raw / decompressed / prediction logical SHAをconfigへ固定
- feature schema SHA:
  `908043637ab6af5033d6ae95be0c4505f9e68a7de9f07d857a94e2665a477b8d`
- model manifest SHA:
  `45876171d9d8de5697146abde3120c184466cb36fc8b9bcaa15b22e1f8bf8dce`
- compact manifest SHA:
  `877a731456f2b93ce59c15b39baf32c850a10f565634754eb8d2b3ebf5710a65`
- outer-valid candidate score SHA:
  `694a1800238b80333ea36ae9d8d098e5de28c0a5e0c63b08547ba44c57e8aeb6`
- submission SHA: 対象外
- rerun check: 未実施。version 1を正の科学runとする
- 親compact比較:
  exp371 train 507行 / 8章に対しexp373 train 510行 / 8章。
  入力列とupstream logical SHAの記録追加だけで、章立て・学習面は同等。

## 次のアクション

1. exp355固定13 selector枝をfail-closeし、same-OOF救済を行わない。
2. downstream TVT、inference、submissionへ進めない。
3. 独立候補exp375の結果は別仮説として扱う。
