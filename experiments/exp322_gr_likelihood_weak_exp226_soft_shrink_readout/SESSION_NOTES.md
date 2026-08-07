# exp322_gr_likelihood_weak_exp226_soft_shrink_readout セッションノート

## 目的

GR matchingが曖昧な時だけexp226へ寄せる案を、exp263固定blendに対する1つのbounded shrinkとして反証可能にする。

## 現在の状態

- Route: `pf_beam`
- 状態: Kaggle CPU train version 2完了、`INCONCLUSIVE_COVERAGE`、branch closed
- active candidate / diagnostic control: `1 / 1`
- model / LightGBM config / trained fold / booster: `0 / 0 / 0 / 0`
- HMM / PF / Beam / K16再生成: `0 / 0 / 0 / 0`
- parent/control再学習: なし
- Kaggle package / push / run: 完了 / version 2完了 / 完了
- inference / submission: 無効 / 無効

## 2026-07-21 設計確定

```bash
make new-steering EXP=exp322_gr_likelihood_weak_exp226_soft_shrink_readout
make new-exp EXP=exp322_gr_likelihood_weak_exp226_soft_shrink_readout
```

- steeringを先に作成し、その後design-only scaffoldを作成した。
- 親をexp263固定`exp226_w500_50_50`、shrink先をexp226 K16に固定した。
- GR弱区間はfinal HMM posteriorではなく、exp280互換のpre-transition raw-GR emission scoreから定義した。
- H512 blockでouter-train margin q20以下 AND entropy q80以上を必須にした。
- shift 0 rank top3 OR zero gap q20以下をexp226 admissibilityとして追加した。
- raw GR観測率`>=0.80`、near 250 ft veto、`alpha=0.25`、最大移動`10 ft`を固定した。
- well内非zero circular-shift gateをmatched negative controlに固定した。
- truth late-join、SHA、coverage、overall/fold/subset/scope/p95/worst/control guardを固定した。
- 実装、Jupytext source、Notebook編集、test、Kaggle package/push/runは行っていない。

## 2026-07-21 train-side実装

ユーザーの実装開始指示により、凍結設計を変更せず次を追加した。

- compact self-contained Jupytext train sourceと別名Notebook。
- exp263 manifest/partition SHA、exp226 decompressed content SHA、anchor/fold/row identity、raw MD identity、hidden-like SHAのhard guard。
- exp280互換`-0.5*min(zscore^2, 600)`の13-shift Gaussian raw-GR score、H512 margin/entropy/zero rank/zero gap/observed share。
- 4-fold outer-train quantile、real gate、stable SHA256 well内circular control、near 250 ft veto、alpha 0.25 / 10 ft clip。
- target-free score/gate/predictionのschema/content SHA freezeと、別APIだけで行うtruth late join。
- overall/fold/activated/1000+/hidden-like 2面/by-well/controlの固定decision。
- exp263 virtual blend float contract、式、tie、fold境界、禁止列、near veto、clip、control、gzip primary SHA、late joinのunit test 11件。

canonical train/inference Notebookは明示採用前のため上書きしていない。親exp280にcompact版はないため、self-contained正規train sourceと比較した。exp280は9章/1,165行、exp322候補は10章/約1,580行で、exp263 input guard、outer-train gate、bounded shrink、late decisionを追加した分だけ厚く、薄いhelper entrypointではない。

```bash
.venv/bin/pytest -q tests/test_exp322_gr_likelihood_weak_exp226_soft_shrink_readout.py
.venv/bin/ruff check experiments/exp322_gr_likelihood_weak_exp226_soft_shrink_readout/exp322_gr_likelihood_weak_exp226_soft_shrink_readout_compact_selfcontained_train.py tests/test_exp322_gr_likelihood_weak_exp226_soft_shrink_readout.py
.venv/bin/python -m py_compile experiments/exp322_gr_likelihood_weak_exp226_soft_shrink_readout/exp322_gr_likelihood_weak_exp226_soft_shrink_readout_compact_selfcontained_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp322_gr_likelihood_weak_exp226_soft_shrink_readout/exp322_gr_likelihood_weak_exp226_soft_shrink_readout_compact_selfcontained_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp322_gr_likelihood_weak_exp226_soft_shrink_readout/exp322_gr_likelihood_weak_exp226_soft_shrink_readout_compact_selfcontained_train.py
make validate-exp EXP=exp322_gr_likelihood_weak_exp226_soft_shrink_readout
make test
```

- unit test: `11 passed`
- ruff / py_compile: PASS
- Jupytext conversion / round-trip: PASS
- strict experiment validation: PASS
- exp322 + exp280 + exp263 cache + Kaggle Notebook関連テスト: `35 passed`
- repository full test: `442 passed / 1 skipped / 2 failed`。失敗2件はいずれも既存`tests/test_exp296_exp223_self_gr_known_tvt_support_gate.py`で、完了済みexp296 config（`completed_train_side_guard_failed_closed` / `execution.run_variant: false`）に対し、旧テストが`kaggle_cpu_*` / run承認済み状態を期待している不整合。exp322変更箇所とは独立しているため修正していない。
- `task validate-exp`は環境に`task`実行ファイルがないため使わず、同等の`make validate-exp`で確認した。
- ローカルNotebook/full data実行、Kaggle package/push/runは行っていない。

## 2026-07-21 Kaggle CPU実行承認・push準備

ユーザーからKaggle CPUでのpackage / push / runについて明示承認を得た。compact self-contained候補をcanonical train Notebookへそのまま採用し、科学設定を変えずに実行する。

- 実行量: `1 candidate / 1 matched control / 5 fold strata`
- 学習量: `0 LightGBM config / 0 trained fold / 0 booster`
- 再生成量: `0 HMM / 0 PF / 0 Beam / 0 K16 well-run`
- parent/control再学習・prediction再生成: なし / なし
- runtime: Kaggle private CPU、GPU/TPU/internet off
- canonical kernel: `kentookumura/exp322-gr-weak-exp226-shrink-readout-train`
- title: `exp322 gr weak exp226 shrink readout train`
- inference / submission: 今回の承認対象外、無効のまま

認証事前確認ではKaggle CLI用OAuth credentialを確認した。最初に予定した58文字slug/titleはKaggle `SaveKernel`がHTTP 400で拒否し、runは作成されなかった。既存実験で確認済みの48文字制約に合わせ、科学contractを変えず42文字の上記canonical名へ短縮する。元slugの事前確認は`kernels pull`が403、同slugの`kernels list`がNot foundだった。canonical採用後に専用unit test `11 passed`、Jupytext round-trip、strict experiment validationを再確認した。

短縮slugのstrict packageは`--no-src`で再生成し、private、CPU、GPU/TPU/internet off、run-on-push true、3 kernel sources、埋め込みapproval trueを確認した。`Kernel version 1 successfully pushed`を確認後、誤再push防止のためローカル`execution_contract.kaggle_push_approved`をfalseへ戻した。Kaggle側metadataは`id_no=128089589`、private、CPU、GPU/internet off、3 kernel sourcesで一致した。初回logsは空で、version 1を継続監視する。

### Kaggle CPU v1 precompute failureとv2最小修正

version 1は約44秒で`ValueError: exp226/exp263 identity mismatch in fold`により停止した。`load_exp226_safe`内のSHA、row/well identity確認後、raw well scoring開始前だったため候補予測・metrics・artifactは未生成である。

ローカル保存済みexp226 OOFとexp263の決定的fold builderを照合すると、両者は別のwell-group splitだった。exp226行数はfold 0〜4で`742514 / 770907 / 746011 / 746131 / 778426`、exp263 readout行数は`757738 / 756650 / 756255 / 757101 / 756245`、同じfold番号の行率は`0.187411750`だった。これはlabel permutationではない。一方、exp263 cache manifestは同じexp226 OOF decompressed SHA `709eb726...e4c609`をsourceとして明記し、cached `exp226_k16`との値parityは既存hard guardで確認する。

親exp263とのreadoutではexp263保存`outer_fold`を正とする。exp226元foldは各wellで一意、fold set `[0,1,2,3,4]`、行数contingencyを別監査し、exp263 foldとの一致は要求しないよう修正した。新しいsplit、refit、予測変更、threshold/gate/guard変更はない。専用testを2件追加し、`13 passed`、ruff、Jupytext round-trip、strict validationをPASSした。

同じcanonical slugへ`Kernel version 2 successfully pushed`を確認した。private CPU、GPU/TPU/internet off、3 kernel sources、`1 candidate / 1 control / 5 exp263 strata / 0 model / 0 booster`はversion 1と同じ。push後にローカルapprovalをfalseへ戻し、version 2だけを監視する。

### Kaggle CPU v2完了・branch closed

Kaggle status `COMPLETE`、773/773 wells、3,783,989 rows、7,787 H512 blocks、固定13 shiftsを確認した。runtimeは`195.331601 sec`。technical hard checksはrow/well/fold coverage、finite score/prediction、exp263 formula parityを含め全PASSし、cached exp226 anchor parityも`0.0 ft`だった。

- decision: `INCONCLUSIVE_COVERAGE`
- changed: `4,870 rows / 0.001287002 / 10 wells / 5 folds`
- coverage: minimum row fraction `0.01` FAIL、minimum wells `50` FAIL
- overall: `8.238331715 -> 8.239202313`、delta `+0.000870598 ft`
- folds: 改善`1/5`。fold 0〜4 deltaは`+0.000514717 / +0.000169465 / +0.000090227 / +0.004085765 / -0.000564897 ft`
- activated subset: `7.744743179 -> 8.433567710`、delta `+0.688824530 ft`
- circular control: RMSE `8.237948157`、real gain minus control gain `-0.001254155 ft`
- scopes: near parity PASS、1000+ `+0.000966632 ft` FAIL、hidden-like 2面は発火0でdelta `0.0`
- by-well: improved / same / worse `2 / 763 / 8`、p95 `0.0`、worst `8c167025 +0.261431339 ft`

coverage不足だけでなくoverall、4/5 folds、activated subset、1000+、worst well、negative controlも不支持だった。固定停止条件どおりalpha、quantile、block、clip、emission、selectorのsame-OOF救済は行わず、branchを閉じる。inference / submission、新しい救済backlogはなし。

outputは性能値とSHAの記録に必要なため完了後だけ`/tmp/kaggle-output/exp322_gr_likelihood_weak_exp226_soft_shrink_readout/train_v2`へ取得した。raw SHAとgzip decompressed SHAを再計算し、Kaggle summaryと一致した。

- target-free contract: `e0b23f2d5852202ab4b4e5aca98cedccc5bcc331888233d0e4af8de0ee389b5f`
- scores decompressed: `adfa974e14a2bcbf481d98784c31e9959fbeee86dc91945007e162adfc8581ce`
- gate decompressed: `8d42a676cf314bf2fb055f6997cd3f243bf128651ec2a253ea447f7a393f2f9a`
- prediction decompressed: `8e335ef58235c44a0cbaae893ee4447054067021f14d3f5858e61d82785ad2c1`
- 最終検証: dedicated `13 passed`、exp322 + exp280 + exp263 cache + Kaggle Notebook関連`37 passed`、ruff / py_compile / Jupytext round-trip / strict validation PASS。

## 根拠

- exp263 fixed: OOF `8.238331`、Public LB `7.800`。
- exp280: top1/top3/MRR/signがshuffleを5/5 foldsで上回ったが、top1 `0.189547`、sign `0.498523`。
- exp281: exp263 fixed比`+1.589088 ft`、0/5 folds、worst `+30.961675 ft`。
- exp133: ambiguous rate `0.5668566`で、ambiguous側がbase model悪化領域ではなかった。
- exp177: Beam ambiguity replacementはbaseline比`+0.242886 ft`、worst `+22.519193 ft`。

## 再現性メモ

- seed policy: real score/gate/shrinkはRNGなし。controlだけwell IDからstable SHA256でcircular offsetを決定。
- stochastic components: matched negative controlのみ。global RNGなし。
- runtime: Kaggle private CPU、GPU/TPU/internet off。v2 `195.331601 sec`で完了。
- kernel id / version: `kentookumura/exp322-gr-weak-exp226-shrink-readout-train` / v2 COMPLETE、id_no `128089589`。
- input/schema/content SHA: version 2 summaryとdownload後の再計算で確認済み。
- prediction SHA: decompressed `8e335ef58235c44a0cbaae893ee4447054067021f14d3f5858e61d82785ad2c1`。
- model/submission SHA: 非該当。
- deterministic anchor: いいえ。train-side readoutでありinference未設計。

## 実行量契約

- scientific candidate: 1
- matched control: 1
- fold strata: 5
- LightGBM config / trained fold / booster: `0 / 0 / 0`
- decoder/parent rerun: 0
- 親実験の保存済みpredictionとcacheだけを読む。

## 次のアクション

exp322は完了・不採用としてbranchを閉じる。救済grid、inference、submissionは行わない。
