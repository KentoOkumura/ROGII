# exp283_self_gr_topk_alternative_mode_branch_future_evidence_readout セッションノート

## 目的

self-GR過去matchをabsolute donor copyではなくtop-3 alternative mode proposalとして扱い、未来256行の
typewell evidenceがproposalを識別できるかを0-boosterで分離監査する。

## 現在の状態

- Route: `pf_beam`
- 状態: Kaggle CPU version 2完了、technical PASS / scientific FAILで不採用確定
- CV / LB: なし
- active variant / LightGBM config / trained fold / booster: `1 / 0 / 0 / 0`
- HMM / PF regeneration: `0 / 0`
- parent/control再学習: 0
- GPU / inference / submission: off / disabled / disabled
- Kaggle push approval: true（2026-07-19 ユーザー指示「実行してください」）

## コマンドログ

### 実行済み

```bash
make new-steering EXP=exp283_self_gr_topk_alternative_mode_branch_future_evidence_readout
make new-exp EXP=exp283_self_gr_topk_alternative_mode_branch_future_evidence_readout
```

上記は初回のtemplate作成である。その後、2026-07-19のユーザー依頼により実装を進めた。

### 実装・静的検証

```bash
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb \
  experiments/exp283_self_gr_topk_alternative_mode_branch_future_evidence_readout/*compact_selfcontained*.py
.venv/bin/python -m py_compile \
  experiments/exp283_self_gr_topk_alternative_mode_branch_future_evidence_readout/*compact_selfcontained*.py
.venv/bin/ruff check \
  experiments/exp283_self_gr_topk_alternative_mode_branch_future_evidence_readout/*compact_selfcontained*.py \
  tests/test_exp283_self_gr_topk_alternative_mode_branch_future_evidence_readout.py
.venv/bin/pytest -q \
  tests/test_exp280_exp226_shift_likelihood_separability_readout.py \
  tests/test_exp282_longtail_prediction_zone_self_gr_loop_closure_readout.py \
  tests/test_exp283_self_gr_topk_alternative_mode_branch_future_evidence_readout.py
make validate-exp EXP=exp283_self_gr_topk_alternative_mode_branch_future_evidence_readout
```

検証結果:

- exp283専用合成test: 7 passed。
- exp280/282/283関連test: 19 passed。
- ruff / `py_compile` / Jupytext train+inference round-trip: PASS。
- strict `validate-exp`: PASS。
- repository full test: 192 passed / 1 failed。FAILは今回未変更のexp264で、configのinference status
  `corrected_inference_v4_complete`とtest期待値`user_authorized_2026_07_19`の既存不一致だった。
- Kaggle prepare / push / run、ローカルnotebook実行、推論、提出は行っていない。

実生成物のread-only input preflightでは、exp209 full 3,783,989行をローカルの制限メモリへ展開中に
exit 137（OOM）となった。最初の意味のある失敗はpandas full-frame loadであり、コード例外やSHA不一致
ではない。事前のheader確認で実列`id, well, md_since, hmm_mean_tvt,
hmm_minus_likpf_mean`、raw SHA `b50b4d1e...b64b`、decompressed SHA `ee3b548b...3f4`はconfigと一致した。
ローカルnotebook smokeへは進まず、Kaggle CPU high-memory runtimeを正とする。

### Kaggle CPU v1 実行承認

2026-07-19のユーザー指示「実行してください」を、固定済みscientific contractのKaggle CPU
train-side readoutを1回実行する明示承認として記録した。実行量はactive audit variant 1 / LightGBM
config 0 / trained fold 0 / booster 0 / HMM・PF regeneration 0 / parent-control再学習0で、GPU、
inference、submissionはいずれも使わない。初回full-name canonical候補は
`kentookumura/exp283-self-gr-topk-alternative-mode-branch-future-evidence-readout-train`とした。

credential preflightはKaggle CLI用OAuth credentialsとlegacy credentialの存在を確認した。primary
API tokenは未設定だが、OAuth credentialsがCLI認証に利用可能なため、このtrain pushの前提を満たす。

## 正規notebook採用とKaggle package監査

ユーザーの実行承認後、compact self-contained train / fail-closed inference sourceから正規
`*_train.ipynb` / `*_inference.ipynb`をJupytextで再生成した。trainは23 cells、inferenceは8 cellsで、
template stubを正規実装へ置換した。採用後のexp283 testは7 passed、Jupytext round-trip、`py_compile`、
ruff、strict `validate-exp`、`validate-template`はいずれもPASSした。

strict packageはcanonical id/titleを同時指定して生成した。train package notebookはbootstrap 1 cell +
正規23 cells、cell output 0で、package内`config.yaml`はrepo側とbyte一致した。metadataはprivate、CPU、
internet off、run-on-push true、competition source 1件、固定kernel source 5件である。

- initial full-name kernel: `kentookumura/exp283-self-gr-topk-alternative-mode-branch-future-evidence-readout-train`
- initial title: `exp283 self gr topk alternative mode branch future evidence readout train`
- regular train notebook SHA: `2683f1aa...bbf`
- regular inference notebook SHA: `953f7214...2ee`
- packaged train notebook SHA: `f062f632...3cc`
- packaged metadata SHA: `8d7efcc2...ee6`

### Kaggle CPU v1 初回push失敗とslug短縮

初回full-name packageのpushはKaggle `SaveKernel 400 Bad Request`で停止し、実行は開始されなかった。
同じ73文字kernel idのmetadata pullは403で、Kaggle側に作成済みkernelは確認されなかった。id/titleの
slug自体は完全一致しており、exp270の53文字、exp275の51文字などで再現しているKaggle側の長いslug
制約パターンと一致する。

同じexp283・同じ科学条件・同じ入力・同じ実行量のまま、内容を保持した49文字の短縮canonical名
`kentookumura/exp283-self-gr-topk-future-evidence-readout-train` / `exp283 self gr topk future evidence
readout train`へ揃えて再packageする。別実験・別仮説には分岐しない。

短縮packageもstrict prepare、exp283 tests 7件、strict `validate-exp`、config byte parityをPASSした。
metadataはprivate / CPU / internet off / run-on-push true、competition source 1件、kernel source 5件を
維持する。実行package config SHAは`a150c68d...a53`、notebook SHAは`2a8f1862...e0f`、metadata
SHAは`00417b88...e9f`である。

短縮canonical kernelへのpushは2026-07-19T15:19+09:00に成功し、Kaggle version 1を開始した。
server metadata pullも成功し、id_no `127849798`、private、CPU（`machine_shape=None`）、internet off、
competition source 1件、固定kernel source 5件を照合した。実行URLは
`https://www.kaggle.com/code/kentookumura/exp283-self-gr-topk-future-evidence-readout-train`。

## Kaggle CPU v1 技術失敗とv2修正

version 1は118秒でtechnical failureとなり、scientific readoutには到達しなかった。最初の意味のある
例外は`build_target_free_identity()`の`exp226 and exp263 outer-fold identities differ`である。

調査の結果、exp263 `outer_fold`はStage 0 cache builderがwell row数から独立に再構築した保存・評価用
partitionであり、exp226 OOFのsource-model foldではない。固定式`exp226_w500_50_50`のcandidate値は
exp263 partitionに依存しないため、行単位一致は期待されず、target-free/leakage contractにも不要だった。

v2では行単位一致guardを削除し、exp226 foldとexp263 partitionがそれぞれwell内で一定か、双方が期待
partition集合を被覆するかを検証する。event/fold guardは引き続きexp226 OOF foldを正とする。K、event、
horizon、proposal、evidence、threshold、input、booster数は変更しない。synthetic testを1件追加し、
scientific contractを変えないtechnical retryとして同じkernel idのversion 2へ進む。

v2 preflightはexp283 tests 8/8、ruff、`py_compile`、Jupytext round-trip、strict `validate-exp`、
package config byte parityをPASSした。server側v1 metadataもpush前にpullし、同じid_no `127849798`、
private、CPU、internet off、competition source 1件、kernel source 5件を確認した。

- v2 train source SHA: `e2840692...9441`
- v2 regular train notebook SHA: `f501eec7...5b72`
- v2 config SHA: `f62cdf61...ee5c`
- v2 packaged notebook SHA: `24a01433...10d`
- metadata SHA: `00417b88...e9f`
- push: 2026-07-19 15:29 JST、同じcanonical kernelのversion 2として成功

version 2はv1の118秒停止点を越えて`RUNNING`を維持した。2026-07-19のユーザー指示により
Codex側のpollingを停止し、Kaggle実行自体は継続する。完了連絡後に同じkernelのlogsを取得し、
scientific guard・SHA・実験記録の確定を再開する。

## Kaggle CPU version 2 完了と生成物監査

ユーザー完了連絡後に同じcanonical kernelのlogs / status / metadataを再取得した。Kaggle statusは
`COMPLETE`、kernel id_noは`127849798`、version 2、private CPU、internet off、competition source 1件、
固定kernel source 5件である。実行時間は1,331.408秒（約22分11秒）。3,783,989 rows / 773 wells、
4,397 events / 13,191 proposals / 103,624 evidence rowsを処理し、booster、HMM/PF regeneration、
inference、submissionはいずれも0で完走した。

technical guardは全PASSした。identity / event / proposal / evidence coverageは1.0、5-fold coverage、
branch identity uniqueness、truth-before-freeze=0を確認した。proposalはtop-3 within10
`0.755288`、shuffled `0.722083`、lift `+0.033204`でpooled guardをPASSし、liftも5/5 foldsで正。
branch-choice AUCはpooled `0.622168`、fold 0--4で`0.613938 / 0.638057 / 0.605266 /
0.613139 / 0.638743`となり、0.60 guardを5/5でPASSした。

しかしfuture-evidence選択はbase RMSE `8.221613`から`14.606586`へ悪化し、gainは
`-6.384973 ft`。fold別gainも`-8.096628 / -6.502113 / -5.571780 / -5.966414 /
-5.893711 ft`でnonregressing foldは0/5。base unique-best 2,435 eventsのfalse-switch率は
`55.5647%`で5% guardを大幅超過した。hidden-like spatial / typewell-purged gainも
`-7.174194 / -7.125766 ft`。768 event wells中、gain正108 / 0が23 / 負637、worst well
`af7a59ce`は`-48.601538 ft`だった。oracle-best RMSE `6.547182`のheadroomは残るが、固定した
target-free verifierは安全に選べない。

Kaggle outputはfold / hidden-like / by-well / SHA監査のため`/tmp/exp283-v2-output`へ一時取得した。
summary内のmetric CSV 8件のSHA、target-free events / proposals / future-evidenceとpost-freeze readout
2件のraw SHAを実ファイルで照合し、5 gzipのdecompressed SHAも一致した。summary SHAは
`8de2db0b5d73f31fc66171893f3355db267376ff39392050ca880ac2bf82fe99`。freeze content SHAはevent
`e4e5c159...bbed`、proposal `2d1d38ac...c31d`、evidence `61e261ad...455`である。大きな生成物は
リポジトリへ保存していない。

scientific checksはproposal lift 2件とAUCだけPASSし、selected H256 gain、5/5 nonregression、
false-switchをFAILしたため総合FAIL。decisionは`close_without_rescue_grid_or_decoder_connection`。
K/window/horizon/veto/margin/threshold救済、decoder接続、inference、submissionへ進まない。exp284は
別の明示overrideでstandalone実行済みだが、exp283からscientific promotionは付与しない。

## 実装境界

- 正規`*_train.ipynb` / `*_inference.ipynb`は、ユーザーの実行承認後にcompact sourceから採用済み。
- v2 compact train sourceは2,291行・10章・23 cells、inferenceは105行・3章・8 cellsである。
- v2 train source / regular notebook SHA: `e2840692...9441` / `f501eec7...5b72`。
- inference source / notebook SHA: `c9685905...16c3` / `9764180a...5046`。
- 親exp282 compact trainは1,630行・10章であり、exp283はsafe input、event、proposal、evidence、
  post-freeze readout、orchestrationを欠かさず同等以上の章立てにした。
- 同じexp内helper importとnotebook unsafeな`__file__`参照はない。
- exp209 enriched cacheにdirect `likpf_mean`列がない実スキーマへ合わせ、
  `hmm_mean_tvt - hmm_minus_likpf_mean`で同値復元してmanifestへ記録する。
- event / proposal / evidenceを順にgzip・schema/content SHAでfreezeし、3 SHAが揃うまでraw TVTと
  hidden-like role readerはfail-closedする。
- real pipelineはRNGなし。shuffleだけwell/event/sourceのstable SHA256 local RNGを使う。
- multiscale agreementはNCC51優先順位を変えないtie-breakとして`mean(NCC17, NCC31)`に固定した。
- H128/256/512、real/shuffled、proposal recall/MRR、branch-pair AUC、selected/base RMSE、
  false switch、fold/stratum/source/orientation/1000+/hidden-like/by-wellを保存する。

## 固定scientific contract

- K=3、primary horizon=256 rows、diagnostic horizon=128/512 rows。
- 4 target-free ambiguity strata、256-row refractory、causal trailing proposal window。
- known prefix + 256 rows以上前のprediction donor bank、forward/reverse、deterministic tie-break。
- exp263 fixed base常時保持、exp226 increment、exp209 typewell verifier、geometry veto only。
- event/proposal/evidence freeze後にtruth attach。
- proposal lift +0.02 / 5 folds、AUC 0.60 / 5 folds、H256 RMSE gain 0.10、false switch 5%をguardする。

## 再現性メモ

- real pipeline RNGなし。shuffleだけstable SHA256 local RNG。
- single process、sorted well/event/donor order。
- gzipはdecompressed content SHAを主証拠とする。
- event/proposal/evidenceのschema/content SHAを別々に保存する設計。
- model/prediction/submissionは作らないためSHA対象外。
- deterministic anchorではなくfixed-input diagnosticとして扱う。

## 次のアクション

exp283はnegative diagnosticとして閉じる。同一仮説の救済grid、decoder接続、推論、提出は行わない。
exp284のstandalone結果にかかわらず、`triggered_fixed_horizon_self_gr_multibranch_hmm_recovery`は
exp283全guard要件を満たさないためpromotionしない。
