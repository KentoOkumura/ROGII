# exp398_all_well_1p3_sigma_gr_exact_hmm セッションノート

## 目的

exp209 absolute-TVT Gaussian exact HMMで、already-clipped `sigma_gr`を全773 wells一律に
`1.3`倍した候補を、saved exp209 controlとのpaired train-side CVで評価する。

## 現在の状態

- Route: pf_beam
- 状態: train_side_all_well_sigma_x1p3_gate_failed_closed
- CV: `12.710664241676811`
- LB: なし（inference / submission未実行）

## コマンドログ

2026-07-25:

- `make new-steering EXP=exp398_all_well_1p3_sigma_gr_exact_hmm`
- `make new-exp EXP=exp398_all_well_1p3_sigma_gr_exact_hmm SOURCE=templates/experiment`
- ユーザーの「全wellでHMMのGRノイズ幅を1.3倍にする実験を行いたい」を、
  exp397のreopenではなく新規exp398の設計・実装意図として扱った。
- exp389 compact self-contained trainの10章 / 1,884行を構成参照元とした。
  exp398 candidateは同じ10章 / 1,913行で、入力、exact-HMM、truth-late join、
  paired metrics、gate、生成物を維持する。
- 専用testを9件作成し、初回`9 passed`。
- Jupytext train/inference round-trip、py_compile、Ruff、strict experiment
  validationをPASSした。`__file__`依存なし。

## 2026-07-25 実行承認

- ユーザーの「実行してください」により、正規train Notebook採用とKaggle private CPU
  package/push/runを承認済みとして記録した。
- push前実行量:
  - scientific variant: 1
  - HMM well-runs: 773
  - reporting folds: 5
  - model / LightGBM config / trained fold / booster: `0 / 0 / 0 / 0`
  - PF / Beam / parent Gaussian control rerun: `0 / 0 / 0`
  - GPU / TPU: 0、internet: off
- 保存済みexp209 Gaussian HMMとexp072 LikPFをload-only controlとして使う。
- Kaggle CPU runtime見積は`11,520--30,600 sec`、同family実績から約5〜8.5時間。
- inference / submissionは今回の承認に含めない。
- credential preflightはOAuthとlegacy credentialをPASS。API tokenは未設定だが、
  Kaggle CLIはOAuthで利用可能。
- 正規train Notebookへcompact self-contained候補を採用した。
- canonical kernel:
  `kentookumura/exp398-all-well-sigma1p3-exact-hmm-train`
- canonical title:
  `exp398 all well sigma1p3 exact hmm train`
- package metadataはprivate / CPU / GPU・TPU off / internet off / run-on-push /
  competition source 1 / kernel source 3。
- 正本 / loose package / embedded bootstrapのconfig SHA:
  `bec1374cfd2056433af1ed2bee01c5ce2adf148dbd98a32c94fc66d26e63257f`
- 正本 / loose package / embedded bootstrapのtrain source SHA:
  `1a7555296cd1f1e6eab354ed4afa728299d5b23fdd66aec37caf83bf9fc8e0b2`
- 正規train Notebook SHA:
  `1ee856b1d486d51690cfd27652e0ad7d6d2afbe90ff3656c091863499ac1930e`
- push package Notebook SHA:
  `a7e3c7ebb9772156966b13ca282a328adb50d0e001471a059a3659e317b71122`
- bootstrap zip SHA:
  `f81a15d58d734f5723c8f43e5e6c291851696c9b4f2d1cea8fd206b8b5fbf27f`
- kernel metadata SHA:
  `56d67d5c38ca7d85c0b099c5bcfaedb3d303858bc655ca236fe6d20cc18ec415`
- pre-push canonical pullは`GetKernel 403 Forbidden`で、既存kernelを確認できなかった。
- 2026-07-25 04:15:04 UTCにcanonical kernel version 1をpushした。
- URL:
  `https://www.kaggle.com/code/kentookumura/exp398-all-well-sigma1p3-exact-hmm-train`
- pulled metadataでid_no `128542706`、private、CPU、GPU/TPU/internet off、
  competition source 1、kernel source 3、canonical id/titleを確認した。

## 2026-07-25 完了結果

- Kaggle kernel version 1は`COMPLETE`。成果物生成時刻は
  `2026-07-25T09:37:12.263934+00:00`、最終確認は`2026-07-25 10:21:29 UTC`。
- runtime `19324.104064941406 sec`、3,783,989 rows / 773 wells /
  773 HMM runs、finite coverage `1.0`、truth-before-freeze `0`。
- direct RMSEは候補`12.710664241676811`、saved exp209
  `11.938287234887435`、改善量`-0.7723770067893767 ft`。
- fold改善は1/5。raw-observed `-0.592611`、raw-missing `-1.150295`、
  high-missing `-1.559979`、1000+ `-0.862967`、hidden-like spatial
  `-2.053945`、typewell-purged `-2.109669 ft`。
- fixed LikPF 50:50も候補`10.653103994924624`、control
  `10.269692505026358`で`-0.38341148989826657 ft`。
- by-wellは330改善 / 443悪化、p95 delta `+7.038260463380123 ft`、
  worst `e03b45fd`の`+46.04649542033126 ft`。
- scientific gateはFAILし、decisionは
  `all_well_sigma_x1p3_failed_close_without_rescue`。

## 完了後の技術監査

- 実行済みgateの`global_sigma_multiplier_contract_passed=false`だけは科学実装の異常ではない。
  runtimeのin-memory sigmaと、別CSVへ保存・再読込したobservation audit sigmaを
  `rtol=0, atol=0`で比較し、188/773件の最大`2.1316282072803006e-14`という
  float serialization差を拾った。
- 全773 runtime行の`sigma_multiplier`は`1.3`、実効sigmaは
  `14.55161031634424--78.0`で、HMM計算への倍率適用は成立。
- ローカルsourceはruntime baseとの比較とCSV round-trip `atol=1e-12`へ修正し、
  回帰testを追加した。このpost-run audit-only fixは実行済みversion 1に含まれない。
- post-run train source / canonical Notebook SHA:
  `fa4d26fd940c12e650357bffda3c11da5b342502055ff92579a27058b6e42ee1` /
  `df669f20ed93d73ab518afd2b3468e05b0f49a685cac71c0540243626509a2d0`
- terminal inference source / terminal config SHA:
  `cb75db517d6cd7a5f2349f1f9db08d2aef24b2d9c902e451a99c6082d0b68bd9` /
  `e9527154747cd8cf306a041d2a5551cde81ac5d452cc969212aab5630948fc4b`
- post-run監査修正後は専用`10 passed`、共通Notebook test込み`14 passed`。
- 科学gateは監査偽陰性と独立して大幅悪化しているため、version 2を再実行しない。

## 完了成果物SHA

- prediction raw / content:
  `288c1db45873d469ac516defb8f62e2df185ee581023317637a6f1675869a6f4` /
  `937c969e2a240fb02533bcb9b00cec31ce0dd0210e547a37c7df54acbd3f0b23`
- scientific contract logical / file:
  `de38dbeaa3124522b2e3be9e22ae1cf7ea1a51cf157852c624b7a4e0a0ba08c6` /
  `3986898973368cb41b87a5b8c4e899d17208a3e5fc285745c0acba999644522a`
- promotion gate:
  `4b165b4539a1ef05f0d22b924b481a8e8b2e7508a969a89df5948efbc8741cc6`
- overall/fold/scope metrics:
  `ba0474b656298d31d34de42482f88398b0e6e2f63e647a87575861e3dadce6ed`
- by-well metrics / runtime:
  `e708876d2fdd83b173eaa682e26957a14caa0435ac6ef96bd096c43db89e7697` /
  `040c0e06b93409edb77f15210f8ada059351b76cb8c718b78fc9b25039c83aa7`
- observation audit raw / content:
  `74eed87d9a21ff1c1b5a1386efddc4b2d65b7c6a295ebb3f2a9f3978cc101b76` /
  `0a16cfe85f3f434c293f86c4e22ba771f56a5eb1633d1662789260454da1c62e`

## 変更点

- parent `sigma_base=clip(zero-fill population std,10,60)`を維持。
- `sigma_eff=1.3*sigma_base`を全wellへ適用し、post-multiplier clipなし。
- capped Gaussian `-0.5*min(z²,600)`を維持。
- saved exp209 HMM / exp072 LikPF / exp226 fold / exp115 roleをload-onlyで使う。
- exp397のrho selector、threshold、tail stabilityは使用しない。

## 再現性メモ

- seed policy: RNGなし、well / row / grid / rate / variant順固定
- stochastic components: なし
- CPU/GPU runtime: Kaggle private CPU予定、GPU/internet off
- 実行量: scientific variant 1 / 773 HMM well-runs / reporting folds 5 /
  model config・trained fold・booster・PF・Beam・parent control rerun各0
- runtime見積: `11,520--30,600 sec`、過去同family実績約5〜8.5時間
- Kaggle kernel id予定:
  `kentookumura/exp398-all-well-sigma1p3-exact-hmm-train`
- input / prediction SHA: 実行時に記録
- model manifest / model SHA: modelなし
- submission SHA: inference/submission無効

## 次のアクション

1. [x] source/test、Jupytext、pycompile、Ruff、strict validationを完了した。
2. [x] 正規train notebookを採用し、実行量確認後にprivate CPU version 1を完走した。
3. [x] 固定gateをFAIL判定し、救済・inference・submissionなしでbranchを閉じた。
