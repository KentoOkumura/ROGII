# exp397_prefix_gr_agreement_adaptive_sigma_exact_hmm セッションノート

## 目的

known prefixのGR agreementでexp209 exact HMMのwell-level `sigma_gr` を `1.0 / 1.3` 倍する
単一仮説を事前固定する。Stage 0ではtruth-freeなagreement surfaceのcoverage、非退化、
full-prefix / last-512安定性だけを0-HMMで監査する。

## 現在の状態

- Route: pf_beam
- 状態: stage_0_completed_guard_failed_closed
- CV: まだなし
- LB: まだなし
- 実装: compact self-contained Stage 0 trainと専用testを実装し、正規notebookへ採用済み
- Kaggle: private CPU version 1完了、id_no `128540665`
- 判定: 固定7条件中4 PASS / 3 FAIL、`stage_0_failed_close_without_rescue`
- Stage 1 / HMM / inference / submission: 未実装・未実行のままbranch close

## コマンドログ

2026-07-25:

- `make new-steering EXP=exp397_prefix_gr_agreement_adaptive_sigma_exact_hmm`
- `make new-exp EXP=exp397_prefix_gr_agreement_adaptive_sigma_exact_hmm SOURCE=templates/experiment`
- steering、config、README、SESSION_NOTES、result、backlog、summaryだけを更新。
- notebook編集、helper/test作成、Kaggle package/push/runは未実施。

2026-07-25 Stage 0実装:

- ユーザーの「exp397を実装してください」をStage 0実装承認として扱った。この時点では
  Kaggle package/push/run、Stage 1、inference、submissionには進んでいない。
- Jupytext percent形式の
  `exp397_prefix_gr_agreement_adaptive_sigma_exact_hmm_compact_selfcontained_train.py`
  を新規実装し、candidate `.ipynb` へ変換した。後続の実行指示を受けて正規
  `exp397_prefix_gr_agreement_adaptive_sigma_exact_hmm_train.ipynb` へ採用した。
- notebookは9章 / 1,380行。親exp209にはcompact self-contained版が存在しないため直接比較は
  できない。近い0-HMM監査のexp343 compact train 1,093行と比較し、runtime/config/SHA、
  input preflight、agreement生成、freeze、gate、orchestrationをnotebook上へ展開した。
- horizontal raw readerは`GR / TVT_input`だけを明示的に読み、horizontal `TVT`、
  Formation、error、hidden-like role、exp209 prediction、saved LikPFをfreeze前に読まない。
- full-prefixはpair 64、tailはknown-prefix末尾512 raw rows内のpair 32、std `>1e-6`、
  Pearson `rho_gr=0.50`境界、`1.0 / 1.3`、fallback `1.0`を実装した。
- agreement / coefficientのrow-order-sensitive logical SHA、raw file identity SHA、
  exp226 fold cache decompressed SHA、truth-read 0 ledger、10 Stage 0生成物manifestを実装した。
- Stage 0 gateはcoverage、fallback、poor-group非退化、fold coverage、tail coverage、
  full/tail係数一致、Spearmanの固定7条件をAND判定し、FAIL時のrescueを無効にした。
- `apply_multiplier_to_clipped_parent_sigma` はalready-clipped `[10,60]` だけを受け、
  `1.0 / 1.3`を1回掛ける。`60 × 1.3 = 78`を再clipしないことをtestで固定した。
- helper `.py` は作らず、candidate notebookをself-containedにした。
- 専用test
  `tests/test_exp397_prefix_gr_agreement_adaptive_sigma_exact_hmm.py` を11件追加した。

検証コマンド:

- `.venv/bin/python -m py_compile <compact_train.py> <test.py>`: PASS
- `.venv/bin/ruff check <compact_train.py> <test.py> --select F821`: PASS
- `.venv/bin/ruff check <compact_train.py> <test.py>`: PASS
- `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb <compact_train.py>`:
  candidate notebook生成
- `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test <compact_train.py>`:
  PASS
- `.venv/bin/pytest -q tests/test_exp397_prefix_gr_agreement_adaptive_sigma_exact_hmm.py`:
  `11 passed`
- `task validate-exp ...`: `task` commandが環境に存在せず未実行
- `make validate-exp EXP=exp397_prefix_gr_agreement_adaptive_sigma_exact_hmm`:
  strict validation PASS
- local notebook / Stage 0 data run: リポジトリ規約どおり未実行

## 変更点

- Parent: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- primary agreement: raw finite known-prefix Pearson `rho_gr`
- fixed selector: `rho_gr >= 0.50 -> 1.0`、`rho_gr < 0.50 -> 1.3`
- fallback: finite pair 64未満、std `<=1e-6`、nonfinite相関は `1.0`
- scale: exp209 base `[10,60]` clip後に1回だけ係数を掛け、再clipなし
- Stage 0: truth-free / HMM 0のcoverage・non-degeneracy・full/tail stability audit
- Stage 1候補: 1 variant / 5 folds / HMMは`1.3` wellだけ、最大773 / booster 0 /
  parent control再実行0
- result後のthreshold、multiplier、support、window、bias、emission、blend rescueは禁止

## 再現性メモ

- seed policy: RNGなし、well ID辞書順、raw row順、fold順固定
- stochastic components: なし
- CPU/GPU runtime: Kaggle CPU `39.35975061899995 sec`、GPUなし
- Kaggle kernel id / version:
  `kentookumura/exp397-prefix-gr-adaptive-sigma-hmm-train` / version 1 /
  id_no `128540665`
- input SHA: exp209 HMM decompressed
  `8e2f42367b7b28e73094eae642c57c75dc8a7ebcfbc3826b0f2067b37f7ae5`
- Stage 0 raw identity expected SHA:
  `bbb687a1998092578583ce259309b49031d095bde57cbb26c0ab8808d2379b32`
- reporting fold decompressed expected SHA:
  `709eb726cc30da523f017ed0dbd0371967b88a91ddcf25578eb9356f28e4c609`
- feature content SHA:
  agreement logical `20a69c425a4f85b288b091f65b1bfd6cfb4990548c43a4b1e50108b1f1357d51` /
  coefficient logical `859512d721cd7e543efb2596ae25cead136181e361dd4b64a6fcbc50af14e8bc`
- model manifest / model SHA: modelなし
- prediction SHA: Stage 1未承認のためなし
- submission SHA: inference/submission未承認のためなし
- rerun check: 事後調整・rescue・version 2を行わずterminal-close

## 実行承認とKaggle run

2026-07-25 Stage 0実行承認:

- ユーザーの「実行してください」を、compact candidateの正規train notebook採用と
  Kaggle private CPU Stage 0 package/push/runの明示承認として扱う。
- push前固定量:
  - diagnostic variant: 1
  - reporting folds: 5
  - HMM well-runs: 0
  - model config / trained fold / PF well-run / Beam well-run / booster: 各0
  - parent control再実行: 0
  - GPU: 0、internet: off
- Stage 1 exact-HMM、inference、submissionは未承認のまま無効。
- canonical kernel:
  `kentookumura/exp397-prefix-gr-adaptive-sigma-hmm-train`
- credential preflight: OAuth credentialsとlegacy Kaggle CLI credentialは有効。
  API tokenは未設定だがKaggle CLI実行にはOAuthを使用できる。

2026-07-25 Kaggle push v0:

- initial id/title:
  `kentookumura/exp397-prefix-gr-agreement-adaptive-sigma-exact-hmm-train` /
  `exp397 prefix gr agreement adaptive sigma exact hmm train`
- pre-push pull: `GetKernel` 403。既存private kernelは確認できなかった。
- push: `SaveKernel` 400。Kaggle notebook実行は開始されず、HMM/model/booster各0。
- id/titleのslug自体は一致していたがslug長が57文字だったため、Kaggle側slug解決上限の
  可能性を避け、科学的意味を保った
  `kentookumura/exp397-prefix-gr-adaptive-sigma-hmm-train` /
  `exp397 prefix gr adaptive sigma hmm train`
  に短縮して再packageする。実験番号、コード、input、gate、実行量は変更しない。

2026-07-25 Kaggle private CPU version 1:

- canonical kernel:
  `kentookumura/exp397-prefix-gr-adaptive-sigma-hmm-train`
- URL:
  `https://www.kaggle.com/code/kentookumura/exp397-prefix-gr-adaptive-sigma-hmm-train`
- id_no: `128540665`
- push: 成功、version 1を実行開始
- pulled metadata: private / CPU (`enable_gpu=false`) / internet off /
  exp226 kernel source 1件 / competition sourceあり
- status: COMPLETE

## Stage 0実測

- runtime: `39.35975061899995 sec`
- execution count: diagnostic 1 / reporting folds 5 / HMM 0 / model config 0 /
  trained fold 0 / PF 0 / Beam 0 / booster 0 / parent control再実行0
- wells: 773、full evaluable `773/773 = 1.0`、fallback `0/773 = 0.0`
- poor multiplier: `8/773 = 0.01034928848641656`
  （固定範囲 `[0.10, 0.90]` をFAIL）
- tail evaluable: `773/773 = 1.0`
- full/tail multiplier agreement: `0.666235446313066`（下限`0.80`をFAIL）
- full/tail Spearman: `0.16746641700676126`（下限`0.70`をFAIL）
- minimum per-fold primary coverage: `1.0`
- fold別poor group: `1 / 1 / 2 / 1 / 3` wells
- fold別full/tail multiplier agreement:
  `0.664516 / 0.638710 / 0.664516 / 0.675325 / 0.688312`
- fold別Spearman:
  `0.226586 / 0.235774 / 0.108153 / 0.029084 / 0.230456`
- full-prefix rho median / q05 / q95:
  `0.8140589 / 0.6262308 / 0.9284776`
- tail rho median / q05 / q95:
  `0.648290 / -0.061970 / 0.898081`
- frozen 7 checks:
  primary coverage PASS、fallback PASS、poor-group non-degeneracy FAIL、
  per-fold coverage PASS、tail coverage PASS、multiplier agreement FAIL、
  Spearman FAIL
- decision: `stage_0_failed_close_without_rescue`

## 実行時SHAとleakage監査

- executed package source:
  `d4f43a6554b1caa3d4d4d840b92a1cb1ff722db1576c35536076c8ad45157d7e`
- executed package config:
  `886540b040ea8eeafb8f893bb99f052937072fda33e6f0b9c6fee20ed9ef3884`
- executed notebook:
  `c8be29734a0d329f2c5724843ac121c66b1126679cd9c09f85e935c6abd3e16f`
- package sourceとKaggle outputに保存されたsourceはbyte-identical
- scientific contract:
  `d2af5925416871f393aaffd3e638c0fb01777d7f74abd592479e7ea9890c3053`
- raw well identity:
  `bbb687a1998092578583ce259309b49031d095bde57cbb26c0ab8808d2379b32`
- fold decompressed:
  `709eb726cc30da523f017ed0dbd0371967b88a91ddcf25578eb9356f28e4c609`
- agreement logical:
  `20a69c425a4f85b288b091f65b1bfd6cfb4990548c43a4b1e50108b1f1357d51`
- coefficient logical:
  `859512d721cd7e543efb2596ae25cead136181e361dd4b64a6fcbc50af14e8bc`
- stability logical:
  `8d821046149e6a5218112b8eb2bab9d752dcfa6cba826abdb8432c784ec19d6f`
- fold metrics logical:
  `e44885a76cd0f525993c2ef13a3443b2b819895772ef7c23fdb51a1022c6d859`
- well manifest logical:
  `4df4b295bbb684d17d6dfe3d1723bb9e520d9fcac77e972b16e215e3d38be97b`
- freeze manifest:
  `74895018c460626e57bae17f694e40db0b6d1b44c4f2817f1dbf21bdebb27d79`
- truth rows before freeze: 0
- exp209 control loaded: false、prediction generated: false

## 結論と次のアクション

full-prefixでは`rho < 0.50`が8 wellsしかなく、固定binary selectorはほぼ全wellをno-opへ
割り当てた。さらにlast-512との係数一致と順位相関が弱く、well-level reliability signalとして
安定していない。したがってStage 1の最大773 exact-HMM runsを正当化できない。
threshold、multiplier、support、window、相関種の調整、Stage 1、inference、submission、
version 2、同family rescue backlogを追加せずbranchを閉じる。
