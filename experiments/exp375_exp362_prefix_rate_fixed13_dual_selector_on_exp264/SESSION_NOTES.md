# exp375_exp362_prefix_rate_fixed13_dual_selector_on_exp264 セッションノート

## 目的

exp362で実際に観測されたprefix-rate-only exact HMM OOFを、corrected exp264
fixed12へ13本目として追加し、別novelty監査を挟まずdual selectorで直接評価する
実験を実装し、Kaggle実行前の契約と静的検証を固定する。

## 現在の状態

- Route: `ensemble`
- 状態: Kaggle private CPU train version 1 完了 / scientific gate FAIL / branch close
- hard OOF CV: `8.78785571000234`
- LB: まだなし
- 親: corrected exp264 Stage C
- candidate parent: exp362保存OOF。ただし候補意味は`prefix_rate_exact_hmm`
- planned active variant / objectives / outer / inner: `1 / 2 / 5 / 4`
- planned / trained CPU selector boosters: `40 / 40`
- parent/control再学習: `0`
- GPU / downstream TVT / inference / submission: `0 / false / false / false`
- 実装承認: 2026-07-24のユーザー指示「exp375を実装してください」
- Kaggle run承認: 2026-07-24のユーザー指示「実行してください」
- 正規notebook採用: 同指示によりcompact trainとfail-closed inferenceを正規名へ採用

## コマンドログ

実行したコマンドを時系列で記録します。未実行のコマンドは予定として明記します。

- `make new-steering EXP=exp375_exp362_prefix_rate_fixed13_dual_selector_on_exp264`
- steeringのrequirements/design/tasklistを記入。
- `make new-exp EXP=exp375_exp362_prefix_rate_fixed13_dual_selector_on_exp264`
- design-onlyのconfig、README、SESSION_NOTES、result、metricsを記入。
- `make update-summary`
- `make validate-exp EXP=exp375_exp362_prefix_rate_fixed13_dual_selector_on_exp264`:
  design-only scaffoldのstrict validation PASS。
- `.venv/bin/python .agents/skills/kaggle-review-exp/scripts/review_exp_docs.py exp375 --root .`:
  core evidence categoriesが揃っていることを確認。
- train / inference notebookは`templates/experiment`から生成された未実装scaffoldのまま。
- ユーザー指示により、この時点で停止する。実装、validation、package、
  Kaggle pushは未実行。ここでいう未実行のvalidationは、将来の実装コードに対する
  test / py_compile / Ruff / Jupytext / contract validationを指す。
- 2026-07-24、ユーザー指示により実装を開始。
- `src/exp362_fixed13_candidate_cache.py`へtarget-free allowlist loader、
  decompressed SHA guard、global key join、selector-fold repartition、
  native confidence、fixed13 paired readoutを実装。
- selector score freeze後だけ動くH512 / whole-well add-one oracle診断を実装。
  学習・scientific gate・閾値選択には使わない。
- `candidate_contract.yaml`、`feature_contract.yaml`、`config.yaml`、
  compact self-contained train、fail-closed inference、専用testを実装。
- `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb ...`
  でtrain/inference候補を変換。
- `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test ...`
  をtrain/inferenceともPASS。
- `.venv/bin/python -m py_compile ...`をPASS。
- `.venv/bin/ruff check ... --select F821,F401,F841,E501`をPASS。
- compact trainの直接path実行はPythonの`sys.path`にrepo rootが入らず
  `ModuleNotFoundError: src`。同じimport-only確認をrepo rootから
  `python -m experiments.exp375_...compact_selfcontained_train`で実行し、
  candidate順序13、compact 77、40 CPU booster契約、`run_approved=false`をPASS。
- `.venv/bin/python -m pytest tests/test_exp375_exp362_prefix_rate_fixed13_dual_selector.py -q`:
  `10 passed`。
- exp375 / exp373 / exp371 / exp264の関連回帰:
  `47 passed, 1 deselected`。
  deselectした1件はexp373の`approval_consumed=false`を固定するpush前
  lifecycle assertionで、exp373側が既に`true`へ進んでいたため。exp373の
  実行記録は本実験から変更していない。
- `task validate-exp ...`はtaskコマンド未導入で実行不能。
  同等の`make validate-exp EXP=exp375_exp362_prefix_rate_fixed13_dual_selector_on_exp264`
  を実行し、strict validation PASS。
- `.venv/bin/python .agents/skills/kaggle-review-exp/scripts/review_exp_docs.py exp375 --root .`:
  collected files全体でcore evidence categoriesが揃っていることを確認。
- `make update-summary`で`experiment_summary.md`を372実験分再生成し、
  exp375を`implementation_complete_train_not_run`として反映。
- 親compact比較: exp371 train 507行 / 8章、exp373 train 510行 / 8章に対し、
  exp375 train 540行 / 8章。native confidenceとpost-freeze診断を追加しつつ、
  章立てとStage A / Stage C orchestrationは同等。
- 正規notebookは未実装scaffoldのまま維持した。compact候補の正規名採用、
  Kaggle package、push、runは未実行。
- 2026-07-24、ユーザーが正規notebook採用とKaggle CPU trainを承認。
  実行範囲は`1 variant / 2 objectives / outer 5 × inner 4 =
  40 CPU boosters / control再学習0 / GPU 0 / downstream TVT 0 /
  inference 0 / submission 0`。
- compact train / fail-closed inferenceを正規notebookへ採用し、canonical SHAを
  それぞれ`03a6df53262f718e057beafd5a94c85f86726fd9b4137c65665b7fa4370672b6` /
  `5ada08a68f7cc6782aba34973cace1daee7330d0bfb73fad3183c3f177b8c8c7`
  と確認した。
- 承認反映後に専用10 tests、Ruff、Jupytext test、strict experiment validationを
  再実行してPASS。関連回帰は47 PASS、exp373がpush後状態へ進んだことによる
  lifecycle assertion 1件のみFAILで、exp375の科学契約には影響しない。
- canonical kernel id/titleはKaggleのslug長制約を避けて
  `kentookumura/exp375-exp362-prefix-fixed13-selector-train` /
  `exp375 exp362 prefix fixed13 selector train`とした。同slug pullは403、
  既知のexp362 source file一覧取得は成功し、canonicalは未作成、
  source OOFはKaggle入力上に存在すると確認した。
- strict Kaggle train packageを生成し、private / CPU / internet off /
  run-on-push、3 kernel sources、26 support files、2 bootstrap dependency files、
  bootstrap内configとlocal configのbyte一致を確認した。
- push直前SHA:
  embedded config
  `dd97bd70c19e0850b91ebce4a51d0c69d19fb7d689526e0fe6ca5bb1db38db07`、
  support ZIP
  `02dbda087917167d751f995efb8a0a567748c6ac4c9d3a1884e506be70dcac90`、
  metadata
  `4d33b7c0945ff3884e2d610304701209f974cc99427974e8ec3d52ddcdf68f02`、
  packaged notebook
  `ba1e5ba30166aa62e1b1ad18380f3e021f278b04bf81b6fe4bb5f2369dec5c50`。
- 初回pushに成功。2026-07-24 05:00:39 UTC時点でversion 1、
  Kaggle id_no `128436686`、status `KernelWorkerStatus.RUNNING`。
  approvalはこのversion 1で消費済みとし、空ログや一時的なstatus障害を理由に
  同じ承認で再pushしない。
- 2026-07-24 06:03:07 UTC、version 1が
  `KernelWorkerStatus.RUNNING`であることを確認。ユーザー指示により継続監視を
  停止した。Kaggle run自体は停止せず、完了連絡後に通常logs、CV、scientific
  gate、post-freeze novelty診断、成果物SHAの取得と記録から再開する。
- ユーザー完了連絡後の2026-07-24 07:07:52 UTCに同canonical version 1を
  `KernelWorkerStatus.COMPLETE`と確認し、通常logs 355行 / 47,948 bytesを取得した。
  notebook科学処理runtimeは`6978.658913637 sec`。
- `FINAL_SUMMARY`から1 variant / 2 objectives / outer 5 × inner 4 =
  40/40 selector models、parent/control再学習0、GPU/downstream TVT/inference/
  submission各0を確認した。
- exp362 OOFは3,783,989 rows / 773 wells、6列allowlist、truth/error pre-freeze
  load 0、decompressed SHA一致、global key join missing 0、source-fold feature利用0、
  native confidence finite率1.0。technical / leakage checkをすべてPASSした。
- Stage Aは153 featuresからall-missing 41 / constant 5 / exact duplicate 17を落とし、
  90 selected features、compact 77。Stage Cは40 models / 25 partitions /
  18,919,945 compact rows / 49,191,857 outer-valid score rowsを完了した。
- selector score guardはexpected-error MAE、within10 logloss/Brierをpooledと
  5/5 foldsでpriorより改善してPASSした。
- fixed13 hard OOFはparent fixed12
  `8.652531955610227→8.78785571000234`、delta`+0.13532375439211286 ft`。
  fold deltaは`+0.4519301745 / +0.1142494463 / +0.0634799591 /
  +0.0004724326 / +0.0223001642 ft`で改善`0/5`。
- near / 1000+は`+0.0371470663 / +0.1486303217 ft`悪化。
  hidden-like spatial / typewell-purgedは`-0.0802141507 / -0.0710709505 ft`
  改善した。
- 追加候補top1率はpooled`0.1152587917`、5/5 foldsで正。
  393/773 wells改善、380悪化、by-well delta p95`+1.0477445674 ft`、
  worst `b19b0395`は`+28.9951164114 ft`。worstでの追加候補top1率は
  `0.0025804232`だけだった。
- post-freeze非gate診断はH512 oracle headroom`0.1626766426 ft` /
  strict unique-best`1236/7787`、whole-well`0.1232133331 ft` /
  `130/773`。候補補完性は確認したがdeployable selector性能とは扱わない。
- scientific gateはusageとhidden-likeをPASSした一方、pooled、fold、near、
  1000+、by-well p95、worst-wellをFAIL。decisionを
  `FAIL_CLOSE_FIXED13_SELECTOR_BRANCH`とした。
- output archive全体は取得せず、scope/usage/by-well/gate/novelty、
  nested metrics/model manifest/compact manifest/reproducibility manifestなど
  小型生成物だけを`kaggle/output/train_v1/`へ取得し、表示SHAと実ファイルSHAを
  照合した。
- 結果反映後、metrics JSON contractと主要9生成物SHAのassertionをPASS。
  専用pytestは`10 passed`、strict experiment validationはPASS。
- experiment reviewerでcore evidence categoriesが揃っていることを再確認し、
  `make update-summary`で`experiment_summary.md`へ
  `completed_fail_closed / CV 8.78785571000234`を反映した。

## 変更点

- planned candidate: `prefix_rate_exact_hmm`
- exp362 OOFの`candidate_tvt`を候補値とし、`candidate_std`と`hmm_loglik`だけを
  標準native confidenceへ写像する。
- `hmm_loglik`はwell内で一定であることを検証し、wellの評価行数で割って
  `loglik_per_row`を作る。
- donor gradient、support/fallback、`mu_rate`、source foldはmodel featureにしない。
- 既存`exact_hmm`と派生formulaを維持し、primary domainだけを11→12候補へ増やす。
- fixed fallbackは7候補のまま。
- 別add-one novelty実験はユーザー判断により省略し、H512 / whole-wellの
  headroomとstrict unique-best率だけをfreeze後の非gate診断へ統合する。

## 再現性メモ

- seed policy: stage/fold/objectiveのimmutable keyからstable SHA256 seedを生成
- stochastic components: selector sampling / LightGBM CPU training
- CPU/GPU runtime: Kaggle CPU、GPU/internet off、`6978.658913637 sec`
- Kaggle kernel id / version / id_no:
  `kentookumura/exp375-exp362-prefix-fixed13-selector-train / 1 / 128436686`
- exp362 prediction logical SHA:
  `bdf616e00bdebb496093d3d05526aebce01381281c4b1c46f7b77e72e57415cb`
- exp362 prediction decompressed SHA:
  `e1d672ff9743b92c33a40bec8d4cf3b0a8c29cdbbb37948992f0809522e3e7ef`
- parent exp264 score SHA:
  `a10b7848127f01bef522f4b17dfd1640c9784956892dc24fc1159e3869500abc`
- exp362 post-read prediction content SHA:
  `fa23301c5b3da1a9846630009e327016b5f131dc1ac370e0c2fa94f9b0561095`
- feature schema SHA:
  `465eee3b936bca4acafbe4c9010e6f744d0b5219c2d0ebeafb6199bdc11c4faf`
- model manifest SHA:
  `03e0277c62c2d315fe5000c9538095449ec73eff4dba71d6a4c311201b1cfbba`
- compact manifest SHA:
  `06a8fbd204c9df14499d0f47f63902faa8c241d1171e71b88cf3b1fd35f36f62`
- outer-valid candidate score SHA:
  `6086a1f1de211a43712cae893669dabbf1958b01edc8a5047d81724db725d67a`
- summary SHA:
  `1e503ff962e7707662baf27d3dab19924ca31fbb24bb5c969dd7170fcdf3f318`
- submission SHA: 今回対象外
- rerun check: exp362 source候補にrerun parityがないため、deterministic anchorはfalse

## 次のアクション

1. 同一OOFのweight / threshold / domain / gate救済を行わず、
   exp375 fixed13 branchを閉じる。
2. downstream TVT、exp362 current-test候補生成、inference、submissionへ進まない。
3. 原因確認が必要になった場合だけ、exp264 / exp371 / exp373 / exp375の保存済み
   candidate scoreを使う0-boosterのincumbent-reranking診断を別承認で検討する。
