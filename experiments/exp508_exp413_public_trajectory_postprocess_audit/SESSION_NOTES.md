# exp508_exp413_public_trajectory_postprocess_audit セッションノート

## 目的

保存済みexp413 Stage D OOFへ、公開実装と同じ固定SG61/p3を最終TVT後処理として適用できるか、
学習・再推論なしで監査する。well-level routingは条件付き後続へ分離する。

## 現在の状態

- Route: `ml_model`
- 状態: Kaggle private CPU Stage A version 1 COMPLETE、promotion FAILで終端閉鎖
- 親実験: `exp413_scale5_likpf_full_replacement_on_exp335`
- 親CV / Public LB参照: `7.884802794404715 / 7.201`
- exp508 CV / LB: `7.878669066831366` / なし
- 実装承認: 2026-08-04ユーザー依頼`exp508を実装してください`
- Kaggle実行承認: 2026-08-04ユーザー依頼`実行してください`
- inference / submission承認: なし / なし

## 設計ログ

- 2026-08-04: 公開source、exp413 final prediction、exp497 strict public-core、既存well selectorのnegative evidenceを確認した。
- 2026-08-04: selectable primaryをwell別SG `61 / 3`の1本へ固定した。
- 2026-08-04: `tau=85` warmup単独とwarmup+SGをreport-only、`selectable=false`へ固定した。
- 2026-08-04: 公開の60/40 direct LikPF blend、well-shape fixed router、SG/tau grid、reanchor、clip、projectionを除外した。
- 2026-08-04: primary gain`0.01 ft`、4/5 folds、固定5 scope、by-well tail、prediction-start continuityの全AND gateを固定した。
- 2026-08-04: well routerはexp508 PASSと独立complementarity evidence後の別expに限定した。
- 2026-08-04: steering、実験scaffold、設計契約、backlog、summaryを作成する。実装コードは作成しない。
- 2026-08-04: ユーザー依頼をStage A実装承認として記録し、Kaggle run、正規Notebook採用、inference、submissionは未承認のまま維持した。
- 2026-08-04: `*_compact_selfcontained_train.py`をJupytext percent形式で実装し、別名`.ipynb`へ変換した。既存の正規train/inference Notebookは上書きしていない。
- 2026-08-04: exp413 Stage D OOF 5生成物、outer fold manifest、hidden-like assignmentのSHA-qualified resolverとfail-fast schema / row-order guardを実装した。
- 2026-08-04: truth/errorを読まない6列OOF allowlistからcontrol / SG61-p3 / tau85 / tau85+SGを生成し、prediction parquetとSHAをfreezeした後だけtruthとhidden-like assignmentを読むphase separationを実装した。
- 2026-08-04: pooled / 5 folds / MD 3 scope / hidden-like 2 scope / by-well tail / first-score-row continuity / trajectory second-difference readoutと固定all-AND gateを実装した。
- 2026-08-04: report-only warmup scoreは`primary_decision_freeze.json`保存後にだけ実行し、primary救済不可とした。
- 2026-08-04: ユーザーの`実行してください`を正規train Notebook採用、Kaggle private CPU package、Stage A runの承認として記録した。inference / submissionは未承認のまま維持した。
- 2026-08-04: compact Jupytext sourceから正規`*_train.ipynb`へ採用し、再度10 tests / py_compile / Ruff / Jupytext round-trip / strict validationをPASSした。
- 2026-08-04: 初回canonical id/titleを`kentookumura/exp508-exp413-public-trajectory-postprocess-audit-train` / `exp508 exp413 public trajectory postprocess audit train`へ一致させ、CPU / internet off / run-on-push packageを生成した。
- 2026-08-04: kernel sourceはexp413 selector/downstreamの2件、hidden-like assignmentはSHA `5f9ac9...`のbootstrap同梱、dataset/model source 0、repository `src/`同梱なしに固定した。
- 2026-08-04: 初回pushはKaggle `SaveKernel 400`。同じIDをpullして403（未作成）、既存exp413 pull成功（credential正常）を確認した。API response本文は`The title cannot exceed 50 characters.`で、55文字titleが原因と確定した。
- 2026-08-04: 50文字以下で仮説を保持するcanonical id/titleを`kentookumura/exp508-exp413-public-sg61p3-audit-train` / `exp508 exp413 public sg61p3 audit train`へ短縮し、id/title slugを一致させた。旧IDは未作成であり重複kernelはない。
- 2026-08-04: canonical短縮packageをpushしversion 1成功。URL `https://www.kaggle.com/code/kentookumura/exp508-exp413-public-sg61p3-audit-train`、id_no `129625989`、push確認`2026-08-03 23:26:41 UTC`。
- 2026-08-04: pull metadataでprivate / CPU (`enable_gpu=false`, `machine_shape=None`) / internet off / kernel sources 2件を確認。statusは`KernelWorkerStatus.RUNNING`。
- 2026-08-04: 同じkernel version 1を完了まで監視し、`KernelWorkerStatus.COMPLETE`を確認した。Kaggle metrics作成時刻は`2026-08-03T23:27:45.379122+00:00`、terminal status確認は`2026-08-03 23:30:45 UTC`。
- 2026-08-04: 3,783,989 rows / 773 wellsでSG61/p3 RMSE `7.878669066831366`、保存exp413 `7.884802794404715`比gain `0.00613372757334929 ft`。5/5 foldsと固定5 scopeはすべて改善した。
- 2026-08-04: by-well p95 / worst deltaは`-0.001344491 / -0.000417966 ft`、`+0.25/+1/+3/+5 ft`悪化wellは全0。first-score-row correction p95 / maxは`0.289606691 / 0.810694404 ft`で安全性gateを全PASSした。
- 2026-08-04: technical / leakage / SHA gateも全PASSしたが、pooled gainだけが固定`0.01 ft`を未達。decision=`FAIL_CLOSE_WITHOUT_SG_GRID_WARMUP_ROUTER_OR_GATE_RESCUE`として終端閉鎖した。
- 2026-08-04: report-onlyはprimary decision freeze後にscoreし、tau85単独`7.886111093377577`、tau85+SG`7.880001331601432`。いずれもprimary救済・候補選択に使っていない。
- 2026-08-04: inference / submissionは実装・実行せず、exp508 PASSを前提とするconditional well routerも作成しない。

## 変更点

- 親exp413の保存最終TVTへ、固定SG61/p3だけをselectable変換として追加する設計。
- tau85 warmup 2本はreport-onlyへ隔離する。
- full public blendとwell routingはexp508から除外する。
- 親prediction、fold、upstream model、physics candidateは変更しない。

## 実行量の事前確認

Stage A push前の確定実行量:

| 項目 | 数 |
| --- | ---: |
| selectable primary | 1 |
| report-only controls | 2 |
| LightGBM config | 0 |
| 学習fold | 0（保存5 foldsをreportのみ） |
| model / booster | 0 / 0 |
| HMM / PF / Beam | 0 / 0 / 0 |
| control再学習 | 0 |
| GPU | 0 |

## 再現性メモ

- seed policy: no RNG、固定source row order、float64
- stochastic components: なし
- CPU/GPU runtime: Kaggle private CPU / GPUなし、internet off
- Kaggle kernel id / version / id_no: `kentookumura/exp508-exp413-public-sg61p3-audit-train / 1 / 129625989`
- input SHA: exp413 OOF `9bd2d17778b3b27d771b12cbff72def8b87e6cdf14062e1c0ba192434cef4a9d`
- fold manifest SHA: `fa41084c5fcb4adffb88d44211b4cc5d2d2f46b5bd4d65828b6af941184b2a6d`
- logical key SHA: `eec221544d109dc265e6491efab589fb9a8f9d883f0de9f9cedac32730744861`
- global / per-well row-order SHA: `d92352a62c15d41bf74eab6eb9ccaca9d324c303a2983b5993fe6c16fafd8a66 / 64c39f7dc8fd910bce09341be5bb0a7ef4bf5dced464301c10c63b820acce1d8`
- postprocess contract SHA: `58afa464063e998f2c4853eb7df2a68784a226ee625a00a8cf1e328280c7dd58`
- primary prediction content SHA: `5caf53f17c52729198eec412cdf7ce46e25f199a24ca842e4efb117e91a67f56`
- promotion gate / reproducibility manifest SHA: `d7e21a243e53f2d5af4e192faf3246e307f4ce92cb2f063bf5e0086582466d1b / 1e68714488b1d7dd938dfe140ab67b00e7f93deb68efb6c520273afbe970b81a`
- model manifest / model SHA: 対象外（学習0）
- submission SHA: 対象外、未承認
- rerun check: 未実行。prediction content SHA一致まではdeterministic anchorとしない。branchは科学FAILで閉鎖済みのためrerunしない。

## 実装ファイル

- `exp508_exp413_public_trajectory_postprocess_audit_compact_selfcontained_train.py`
- `exp508_exp413_public_trajectory_postprocess_audit_compact_selfcontained_train.ipynb`
- `test_exp508_contract.py`

## 親compactとの構成比較

- 参照: exp497 compact self-contained train、8章 / 1,044行。
- exp508候補: 9章 / 1,324行。
- exp508はImports、path/hash、static contract、input checks、truth-free candidate freeze、
  truth-late metrics/gate、report-only、生成物、setup/stopをNotebookセル上に展開しており、
  同一exp helperを呼ぶだけの薄い構成ではない。

## 静的検証ログ

- `.venv/bin/python -m py_compile ...compact_selfcontained_train.py .../test_exp508_contract.py`: PASS。
- `.venv/bin/ruff check ...compact_selfcontained_train.py .../test_exp508_contract.py`: PASS。
- `.venv/bin/pytest -q .../test_exp508_contract.py`: `10 passed`。
- `rg -n "__file__|Path\\(__file__\\)" ...compact_selfcontained_train.py`: 該当0。
- `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test ...compact_selfcontained_train.py`: PASS。
- `task validate-exp ...`は環境に`task`がなく実行不可。skillのfallbackどおり
  `make validate-exp EXP=exp508_exp413_public_trajectory_postprocess_audit`を実行しstrict PASS。
- `make validate-template`: PASS。

## 禁止事項

- 完了済みStage Aのgate緩和、再実行、別slugへの重複push
- SG window/polyorder、tau、blend、clip、reanchor、projectionのgrid
- report-only warmupによるprimary救済
- exp508内のrow/well router、公開固定threshold/map、truth/error gate
- exp413 / exp497 / PF/HMM/Beam / selector / boosterの再実行
- inference、submission

## 次のアクション

なし。exp508は固定pooled gain未達で終端閉鎖し、SG/tau/routerのsame-OOF救済、
inference、submissionへ進まない。現行の独立P1候補を優先する。
