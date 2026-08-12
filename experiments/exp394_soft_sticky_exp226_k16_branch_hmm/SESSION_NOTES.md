# exp394_soft_sticky_exp226_k16_branch_hmm セッションノート

## 目的

exp226 geometry-only path と exp355 K16-relative free exact HMM を
soft-sticky 2-branch modelへ統合する設計を確定する。

## 現在の状態

- Route: `pf_beam`
- 状態: `technical_preflight_runtime_failed_closed`
- CV / LB: なし
- source: 3,522行compact self-contained train候補を実装
- helper: 同一exp helper importなし
- test: 専用10件PASS
- Notebook: 別名compact self-contained候補を正規trainへ採用
- Kaggle package / push / fixed16 preflight: version 1完了・runtime gate FAIL
- full OOF / inference / submission: 未承認

## 2026-07-25 fixed16 technical preflight実行承認

- 承認元: ユーザー指示「実行してください」
- 実行scope: technical candidate 1 / 固定16 wells / switching-HMM well runs 16
- reporting fold: 固定well選択が5 foldsを覆うが、fold CV/RMSEは計算しない
- LightGBM config / trained fold / booster: `0 / 0 / 0`
- parent/control rerun / GPU / inference / submission: `0 / 0 / 0 / 0`
- full OOFの将来量: 1 variant / 5 reporting folds / 773 HMM well runs
- full OOFは今回の承認外。technical PASS summary SHAを凍結してから別承認を得る。

## 2026-07-25 Kaggle version 1 結果

- kernel: `kentookumura/exp394-soft-sticky-exp226-k16-branch-hmm-train`
- version / id_no: `1 / 128536142`
- URL: <https://www.kaggle.com/code/kentookumura/exp394-soft-sticky-exp226-k16-branch-hmm-train>
- runtime: `3703.079064 sec`
- scope: `140,721 rows / 16 wells`
- technical gate: `FAIL`
- decision: `technical_blocker_not_scientific_negative_result`
- 唯一のFAIL: projected full runtime
  - 実測投影: `112,736.889439 sec`（`31.3158 h`）
  - 上限: `30,600 sec`（`8.5 h`）
  - 上限比: `3.684212x`
- PASS:
  - completed wells `16 / 16`
  - finite prediction coverage `1.0`
  - H full-grid coverage `1.0`
  - raw / exp226 input identity
  - truth/error/hidden-like pre-freeze read `0`
  - posterior normalization max error `4.529710e-14 <= 1e-8`
  - transition row-sum max error `8.881784e-16 <= 1e-10`
  - projected peak RSS `1.515934 GB <= 25 GB`
- RMSE / CV / LB: 未計算。technical FAILをscientific negativeとは解釈しない。
- model / booster / control rerun / GPU / inference / submission: 全て0。
- summary raw SHA:
  `7f497fe5a44f1bf58d3dab758b6398eeedae986250c6a17b9e6b6c3be3c6321d`
- gate raw SHA:
  `6cd0f5f72a8fbdcf033856b32bd2819bf166ad8accf35aebe8c1c0f01884bab7`
- prediction / branch posterior / schedule content SHA:
  - `b71bf25441d217e992b97dacceaca93ab2bcff2646d7043aafd34f7b39b4710a`
  - `40558f9e94792fca4f2da950a5e547bc5ef1eb7688604a57054f4e15d56c82a1`
  - `c05a0764382c0fb0f366614bff642e2f2ffb41f865190b56d13f0c4e32c9843b`
- output: `/tmp/kaggle-output/exp394_soft_sticky_exp226_k16_branch_hmm/train_v1`
- run flagと`train_run_on_push`は完了後にfalseへ戻す。full OOFは実行しない。

## 2026-07-25 設計判断

- 2 branchはexp226を信頼するE branchと、free exact HMMを信頼するH branch。
- H branchの多峰性は全TVT grid上に保持し、HMM modeを有限列挙しない。
- exp226/K16 rateはGR補正前のH transition meanにだけ使う。
- E branchのGR emissionは`v=tvt_geop`で評価する。
- soft-sticky固定値はinitial 0.5/0.5、base switching length 1000 MD-ft、docking 6 ft。
- 低ランク3D地層場は前提にもfallbackにも置かない。
- 16-well preflightに科学score gateを置かず、full candidateを小標本FAILで止めない。

## 将来の実行量（未承認）

- scientific variant: 1
- reporting folds: 5
- switching exact-HMM well runs: 773
- LightGBM configs / trained folds / boosters: `0 / 0 / 0`
- parent/control reruns: 0
- GPU: 0

## 2026-07-25 実装

- `exp394_*_compact_selfcontained_train.py`へ12章のJupytext実装を追加した。
- exp209のabsolute-TVT gridと41 residual-rate statesを削減せず、Eの1 stateと
  H全stateを同一scaled log-space forward-backwardで周辺化する。
- transitionは各sourceで正規化し、E→E/E→HとH→E/H→Hのrow sumを監査する。
- E→Hは前rowの`tvt_geop`とresidual rate 0からH kernelへ注入する。
- H→Eは次H stateと現`tvt_geop`の6 ft Gaussian docking期待値でswitch hazardを抑える。
- joint posterior mean/std、`gamma_E/H`、H conditional mean/std、expected switch、
  docking、両branch emission、K16 `mu_rate`を保存対象にした。
- fold別longest 3 + duplicate-free global median lengthの固定16-well ledger、
  state-time based runtime投影、peak RSS、finite/full-grid/normalization gateを実装した。
- full OOFはprediction/branch/schedule SHAをfreezeしてからだけtruth、exp263、
  hidden-like roleをjoinし、exp281定義のpersistent-offset/recoveryも含めて評価する。
- preflight PASS summary SHAと別のfull OOF承認がなければfull pathはfail closedする。

## 検証

- `pytest -q experiments/exp394_soft_sticky_exp226_k16_branch_hmm/tests/test_exp394_soft_sticky_exp226_k16_branch_hmm.py`: `10 passed`
- exp355 Stage 1 / exp281 recoveryの隣接回帰を含む実行: `21 passed`
- small synthetic trellisでoptimized switching kernelとdense joint trellisを比較し、
  branch posterior、H position posterior、joint meanが`2e-6`以内で一致した。
- dense forward-backwardは全path列挙posterior/log partitionと`1e-13`以内で一致した。
- Jupytext `--to ipynb --test`: PASS
- `py_compile`: PASS
- Ruff `F821,E9`: PASS
- 親compact比較: exp355 Stage 1は12章/2,226行、exp394は12章/3,522行。
- ローカルNotebook実行、Kaggle package、Kaggle runは行っていない。

## 再現性メモ

- seed policy: RNGなし、fold/well/row/branch/TVT/rate順を固定。
- stochastic components: なし。
- runtime: Kaggle CPU、上限30,600秒 / 25GB。exp355親は18,161.789秒。
- truth freeze: schedule、branch posterior、joint prediction、SHAの後だけsuffix truthを結合。
- SHA: raw identity、decompressed input、schedule、posterior、prediction、contractを記録予定。
- kernel / prediction / submission SHA: 未実行・未生成。
- sourceは`__file__`を使わずNotebook-safe `Path.cwd()`起点。
- gzip出力はraw SHAとdecompressed/logical content SHAを分離する。

## 次のアクション

1. exp394はfixed runtime gateに従い、full 773-well OOFへ進めず閉じる。
2. exp226、exp355、exp263の既存結果は再分類しない。
3. 再訪する場合は別設計で、同一fixed16出力の数値同値性を保ったまま
   `>=3.684212x`のwall-clock短縮を事前に示すtechnical auditだけを検討する。
