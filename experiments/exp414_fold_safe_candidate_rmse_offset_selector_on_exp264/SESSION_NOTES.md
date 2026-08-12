# exp414_fold_safe_candidate_rmse_offset_selector_on_exp264 セッションノート

## 目的

exp407の真の悪化原因を保存済みOOFから定量的に確定し、その原因を避けながら
候補別RMSEを利用する固定手法を1つ実装・検証する。

## 現在の状態

- Route: `ml_model`
- 状態: implementation-only完了、静的検証中
- CV / LB: 未実行 / 対象外
- 正規train Notebook: template placeholder
- compact self-contained候補: 作成済み
- Kaggle package / run: 未作成 / 未実行
- Stage C / inference / submission: 対象外

## 2026-07-26 根本原因

corrected exp264とexp407のcandidate-score OOF（3,783,989 base rows、
45,407,868 candidate-long rows）を同一key / orderで二回streamした。

- exp407 hard RMSE: `8.66814102464331`
- 親 hard RMSE: `8.587004386703422`
- candidate×fold平均score shiftだけを親へ適用:
  `8.580476914703985`、親以下4/5 folds
- exp407から同平均shiftを除いたrow-local変化だけ:
  `8.673599263270791`、親以下1/5 folds
- exp407 final weightとrow-local score差stdのSpearman:
  `-0.5933870519588776`
- final weightとscore MAE悪化のSpearman:
  `-0.4116699083078634`
- final weightとbinary logloss悪化のSpearman:
  `-0.6037788274520701`
- final weightとcandidate定数shiftのSpearman:
  `-0.07324256737982773`
- 親margin 0.5--2.0のswitchがnet delta SSEの約74%を占めた。

結論は、inverse-RMSE weightingが候補一律のcalibration offsetを作ったのではなく、
低重み候補の共有lossへの寄与を落とし、局所interaction / rankingを分散して
変えたことである。RMSEをbinary objectiveにも同じ重みで使ったことは目的不整合で、
low-weight候補ほどlogloss / Brierも悪化した。

## 2026-07-26 固定treatment

各outer foldのexact sampled fit rowsから候補別RMSE `b_fc`を計算し、
次のadditive residual parameterizationだけを試す。

```text
residual_target = candidate_abs_error - b_fc
pred_abs_error = max(0, residual_prediction + b_fc)
```

`|r_hat - (error - b)| = |r_hat + b - error|`なので、元のunweighted L1を
維持する。target除算はinverse-RMSE weightingと同値になるため禁止した。
sample weight、binary objective、offset scale / clip / subset gridは使わない。

## 実行予算

| variant | objective | outer fold | CPU booster | classifier | control再学習 | GPU |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 5 | 5 | 0 | 0 | 0 |

PF/HMM/Beam再生成、Stage C、inference、submissionも0。

## 実装

- `src/candidate_rmse_offset_selector.py`
  - exact candidate-long order / fit row-ID監査
  - fit-partition RMSE offsetとresidual target
  - unweighted 5-fold LightGBM回帰
  - candidate-score OOF、fold / candidate / bucket / well readout
  - offset / model / OOF manifestと親比較all-AND gate
- `src/candidate_rmse_root_cause_readout.py`
  - global-shift-only / row-local-only counterfactual
  - inverse-RMSE weight dose-response
  - parent margin damage
  - treatmentの親比centered score drift
- compact Jupytext候補は8章、18セル（markdown 10、code 8）。
- 正規Notebookはユーザー承認前のため上書きしていない。

## 静的検証

実行済み:

```bash
.venv/bin/pytest -q experiments/exp414_fold_safe_candidate_rmse_offset_selector_on_exp264/tests/test_exp414_candidate_rmse_offset_selector.py
.venv/bin/python -m py_compile \
  src/candidate_rmse_offset_selector.py \
  src/candidate_rmse_root_cause_readout.py \
  experiments/exp414_fold_safe_candidate_rmse_offset_selector_on_exp264/\
exp414_fold_safe_candidate_rmse_offset_selector_on_exp264_compact_selfcontained_train.py
.venv/bin/ruff check <上記実装とtest> --select F821,F811,F401,E501
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb <Jupytext候補>
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test <Jupytext候補>
```

- dedicated + related selector tests: 32件PASS
- py_compile: PASS
- Ruff F821/F811/F401/E501: PASS
- Jupytext変換 / round-trip: PASS
- import-only static contract: PASS
- 再利用化したroot-cause moduleを実parent / exp407 OOFへ適用し、
  `8.580476914703985` / `8.673599263270791`と9/9原因gate PASSを再現した。
- Notebookローカル実行: 未実施

`make test`は1,215件をcollection中、今回触れていない
`test_exp408_hmm_message_rate_basin_audit.py`が作る`numba.__spec__`なしstubと、
`test_exp411_predictive_filtered_rate_innovation_destick.py`の
`importlib.util.find_spec("numba")`が衝突してcollection errorになった。
exp414を含む関連32件は独立実行で全PASSしており、exp414由来のfailureは0。

## 再現性メモ

- seed policy: 親seed 42と`exp264` deterministic sample keysを継承
- stochastic components: LightGBM subsample / colsampleだけ
- runtime: Kaggle private CPU、internet off、deterministic / force_col_wise
- candidate generation: 保存cache load-only、再生成0
- parent candidate-score OOF SHA:
  `9a91b62599278d4e56d57074df4725d4a09391460458b8eccc02dd50af34d48a`
- exp407 candidate-score OOF SHA:
  `d993b806d92c2462c1509f110669b272b27d48806c0280a2cf54e87c7f32f1e8`
- feature schema logical SHA:
  `aaef4ffdd90667893b099b76a52f1957b22197aea9cee5e5b57bc81048ddd3a4`
- candidate contract file SHA:
  `4f4d3f77db01d7477f9e73066ac311cfdc2c14b15eba84fab9830f4cf5486c20`
- model / offset / OOF / gate SHA: Kaggle実行後に記録
- prediction / submission SHA: 対象外
- deterministic anchor: false。Kaggle versionと出力SHA未記録のため。

## 次のアクション

1. strict experiment validationを完了する。
2. ユーザー承認後だけcompact候補を正規train Notebookへ採用する。
3. private CPU v1を5 boosterで実行し、全gateを判定する。
