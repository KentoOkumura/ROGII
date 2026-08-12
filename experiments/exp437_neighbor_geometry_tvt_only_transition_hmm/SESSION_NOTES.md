# exp437_neighbor_geometry_tvt_only_transition_hmm セッションノート

## 目的

exp435のTVT-only HMMの状態遷移へ、`-ΔZ`だけでなくfold-safeな周辺井戸geometryの
一行増分を入れるStage 0を実装する。

## 現在の状態

- Route: `pf_beam`
- 状態: `stage0_fail_closed`
- CV / LB: なし
- 実装承認: 2026-07-29のユーザー指示`exp437を実装してください`
- Kaggle Stage 0: version 1完了、technical PASS / mechanism FAIL
- train notebook: 9 code cells / 11 markdown cellsのcompact self-contained実装
- inference notebook: raw-test geometry再生成未設計のためfail-closed
- package / push / Stage 0 run: 完了
- Stage 1 / inference / submission: 未実施・不適格

## 2026-07-29 設計セッション

ユーザーの意図:

```text
exp435の状態遷移にdzだけでなく周辺の井戸情報も入れる。
バックログ、実験ディレクトリ、steeringを作成して設計を確定する。実装はまだ行わない。
```

設計確定内容:

- 実験番号を`exp437`、親をexp435、geometry evidence parentをexp226とした。
- exp435のTVT-only probability state、GR emission、grid、noise、forward-backwardを固定した。
- 保存済みfold-safe exp226 `tvt_geop`の隣接差だけをtransition centerへ入れる。
- scientific candidateは`neighbor_geometry_direct_transition` 1本だけとした。
- rate state / rate mixture / branch stateは復活させない。
- exp226 final prediction、GR correction、U projection、blend、selectorは使わない。
- fixed32 Stage 0とfull OOF Stage 1を分け、Stage 1には別承認を要求した。
- exp355、exp394、exp436とは異なる介入であることを固定した。
- truth-late、read時allowlist、logical content SHA、fail-close gateを固定した。

## 変更点

- exp435比ではtransition centerだけを変更する。
- exp226比では`tvt_geop`の絶対予測を最終出力へコピーせず、その隣接差だけを
  exp435 HMMの遷移に使う。
- 実験配下の正規notebookからscaffoldのcode cellを除き、未実装状態をfail-closedにした。

## 2026-07-29 実装セッション

ユーザーの明示指示:

```text
exp437を実装してください
```

実装内容:

- `exp437_neighbor_geometry_tvt_only_transition_hmm_compact_selfcontained_train.py`
  をJupytext percent形式で作成し、正規train notebookへ採用した。
- exp226 OOFはread時に
  `well_id,row_idx,suffix_offset,tvt_geop,fold`だけをallowlist指定し、
  gzip decompressed SHA一致を必須とした。
- exp435 Stage 0保存予測はfull logical SHA一致を確認し、
  `dz_only_r0_prediction`だけをmechanism controlとして使う。
- `mu_geo(0)=tvt_geop(0)-last_known_TVT_input`、以降は隣接差とする
  geometry scheduleを実装した。
- exp435のTVT grid、start prior、GR emission、`sig_p`、5-cell kernel、
  forward-backwardを固定し、rate/branch stateなしのdirect transitionを実装した。
- 32 wellsすべてのschedule/prediction/diagnostic SHAをfreezeしてからだけ
  manifest role/foldとsuffix truthを読むtruth-late ledgerを実装した。
- exp226 geometryと保存exp435 dz-onlyに対するtechnical/mechanism AND gate、
  by-fold、by-well tail、runtime/RSS projectionを実装した。
- inferenceはStage 1 promotionとraw-test geometry regenerationの別設計まで
  必ず失敗するguard notebookにした。

親compactとの比較:

- exp435 compact train: 2,452行、9章。
- exp437 compact train: 1,714行、9章。
- exp437は親の章スロットをすべて維持し、rate mixture / persistent episode readoutを
  geometry schedule / saved-control mechanism readoutへ置き換えた。実験内helper
  importや`__file__`はない。

実装時の実行量contract:

- active variant: 1
- HMM config: 1
- Stage 0: 32 wells / 32 candidate HMM well-runs
- Stage 1最大: 773 candidate HMM well-runs（未実装・未承認）
- parent/control rerun: 0
- LightGBM config / trained fold / booster: 0 / 0 / 0
- PF / Beam / GPU: 0 / 0 / 0

## 予定実行量

| 項目 | Stage 0 | Stage 1最大 |
| --- | ---: | ---: |
| scientific variant | 1 | 1 |
| candidate HMM well-runs | 32 | 773 |
| parent/control HMM reruns | 0 | 0 |
| fitted ML model | 0 | 0 |
| LightGBM config / trained fold / booster | 0 / 0 / 0 | 0 / 0 / 0 |
| PF / Beam / GPU runs | 0 / 0 / 0 | 0 / 0 / 0 |

Stage 0は16 persistent + 16 matched control / 156,088 suffix rowsの
mechanism preflightであり、CVまたはpromotion evidenceではない。Stage 1は
Stage 0全gate PASSと別のユーザー承認が必要である。

## コマンドログ

設計scaffold作成:

```bash
make new-steering EXP=exp437_neighbor_geometry_tvt_only_transition_hmm
make new-exp EXP=exp437_neighbor_geometry_tvt_only_transition_hmm
```

`task` runnerは環境に存在しなかったため、リポジトリの`Makefile`入口を使用した。
設計文書とmetadataの静的検証以外にtrain/inferenceコマンドは実行していない。
Kaggle package/push/runも実行していない。

実装検証:

```bash
.venv/bin/python -m py_compile \
  experiments/exp437_neighbor_geometry_tvt_only_transition_hmm/exp437_neighbor_geometry_tvt_only_transition_hmm_compact_selfcontained_train.py \
  experiments/exp437_neighbor_geometry_tvt_only_transition_hmm/exp437_neighbor_geometry_tvt_only_transition_hmm_compact_selfcontained_inference.py \
  experiments/exp437_neighbor_geometry_tvt_only_transition_hmm/tests/test_exp437_neighbor_geometry_tvt_only_transition_hmm.py
.venv/bin/ruff check \
  experiments/exp437_neighbor_geometry_tvt_only_transition_hmm/exp437_neighbor_geometry_tvt_only_transition_hmm_compact_selfcontained_train.py \
  experiments/exp437_neighbor_geometry_tvt_only_transition_hmm/exp437_neighbor_geometry_tvt_only_transition_hmm_compact_selfcontained_inference.py \
  experiments/exp437_neighbor_geometry_tvt_only_transition_hmm/tests/test_exp437_neighbor_geometry_tvt_only_transition_hmm.py --select F821,E9
.venv/bin/pytest -q experiments/exp437_neighbor_geometry_tvt_only_transition_hmm/tests/test_exp437_neighbor_geometry_tvt_only_transition_hmm.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test \
  experiments/exp437_neighbor_geometry_tvt_only_transition_hmm/exp437_neighbor_geometry_tvt_only_transition_hmm_compact_selfcontained_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test \
  experiments/exp437_neighbor_geometry_tvt_only_transition_hmm/exp437_neighbor_geometry_tvt_only_transition_hmm_compact_selfcontained_inference.py
```

結果:

- py_compile: PASS
- Ruff F821/E9: PASS
- exp437専用test: 8 passed
- train / inference Jupytext round-trip: PASS
- 親exp435 + exp437関連test: 19 passed
- strict `make validate-exp`: PASS
- 実ファイルinput contract: fixed32 32 wells / 156,088 rows、exp226 geometry
  32 wells / 156,088 rows、forbidden read 0、decompressed SHA一致
- ローカルHMM Stage 0実行: 0
- Kaggle package / push / run: 0

ローカルinput contract確認の初回は`.venv`にNumbaが入っていないため
`ModuleNotFoundError: No module named 'numba'`でimport時に停止した。HMMは実行せず、
contract testと同じnon-execution stubでresolverだけを再実行して上記の実ファイル
row/SHA/allowlist一致を確認した。Kaggle側依存や本体importは変更していない。

全repo `pytest -q`はexp437へ到達する前のcollectionで、既存のexp297、exp301、
exp333、exp336、exp349のconfig/root解決エラー5件により停止した。exp437固有8件と
直接親exp435を合わせた19件はPASSしており、この既存collection failureは本実装では
変更していない。

## 再現性メモ

- seed policy: RNGなし。fold / well / row / variant / reduction順を固定する。
- stochastic components: なし。
- runtime: Kaggle private CPU想定、GPU / internet off。
- worker policy: 1 worker、Numba 4 threadsを設計値として固定。
- exp226 OOF decompressed SHA:
  `709eb726cc30da523f017ed0dbd0371967b88a91ddcf25578eb9356f28e4c609`
- exp435 fixed32 prediction logical SHA:
  `aa79810f6c189dd7fbb9d53b8c172a4a051d29ac1780ee4696237e8c24e214c3`
- fixed32 manifest SHA:
  `fbbc62b7cb79e16a7fb436f3a9d11f8975e935ad2475a17e2dec4fd7b142e4d6`
- candidate schedule / prediction / diagnostic / metrics SHA: 未実装・未実行のため未生成。
- deterministic anchor: 初回runでは主張せず、同一設定rerunのschedule/prediction
  logical SHA一致後だけ再判定する。

## 次のアクション

1. 今回はStage 0実装完了で停止する。
2. Kaggle package/push/runは別承認とする。
3. 実行する場合も新candidate 1本×32 wellsだけとし、controlを再実行しない。
4. Stage 0の全gate PASS時だけ、Stage 1の実行可否を相談する。

## 2026-07-29 Stage 0実行承認

ユーザーの明示指示:

```text
実行してください
```

2026-07-29 12:32:35 UTC時点のpush前計算契約:

| 項目 | 数 |
| --- | ---: |
| active scientific variant | 1 |
| HMM config | 1 |
| Stage 0 candidate wells / HMM well-runs | 32 / 32 |
| Stage 1 wells / HMM well-runs | 0 / 0（未実装・未承認） |
| parent/control HMM rerun | 0 |
| fitted ML model / LightGBM config / trained fold / booster | 0 / 0 / 0 / 0 |
| PF / Beam / GPU run | 0 / 0 / 0 |

保存済みexp226 geometryとexp435 fixed32 predictionをcontrolとして使い、既存
baseline/controlは再実行しない。今回の承認範囲はcanonical private CPU kernelの
package、push、fixed32 Stage 0実行、logsによる結果確認までとする。Stage 1、
raw-test regeneration、inference、submissionは未承認のまま維持する。

push前検証:

- strict `make validate-exp`: PASS
- exp437専用test: 8 passed
- Ruff: PASS
- canonical kernel id:
  `kentookumura/exp437-neighbor-geometry-tvt-only-transition-hmm-train`
- title: `exp437 neighbor geometry tvt only transition hmm train`
- metadata: private / CPU / internet off / run_on_push
- kernel sources: exp226 train、exp435 trainの2件
- package notebook SHA:
  `d0abaef6886a12b53f5d6f204e6c60cfe83c2a28c4a4385113e3ad71f0af7635`
- canonical / loose / bootstrap config SHA:
  `cb92062203da45990fa67fa3dc615bc34ed3f2bbbe7c2b2fecb4749dcae6364f`
- bootstrap train source SHA:
  `cde657d1fd1e5c4423e0270ef0685b67c85ff5061c93f13bb0c8d4b707e97a2b`
- bootstrap fixed32 manifest SHA:
  `fbbc62b7cb79e16a7fb436f3a9d11f8975e935ad2475a17e2dec4fd7b142e4d6`
- loose / bootstrap config、train source、fixed32 manifestはbyte一致。

初回push:

```text
2026-07-29 12:32 UTC
kentookumura/exp437-neighbor-geometry-tvt-only-transition-hmm-train
400 Client Error: Bad Request / SaveKernel
```

id末尾とtitle由来slugは一致していたが54文字だった。参照元exp226 / exp435
kernelは`pull -m`で存在を確認し、元exp437 slugは`pull -m`が403でKaggle上に
作成されていないことを確認した。Kaggle slug長制約を避けつつ意味を維持するため、
同じexp437内で`only`だけを省いた49文字のcanonical id/title
`kentookumura/exp437-neighbor-geometry-tvt-transition-hmm-train` /
`exp437 neighbor geometry tvt transition hmm train`へ短縮してpackageを再生成する。
別実験や別科学variantは作らない。

## 2026-07-29 Kaggle Stage 0結果

- canonical kernel:
  `kentookumura/exp437-neighbor-geometry-tvt-transition-hmm-train`
- version / id_no: `1` / `129056603`
- status: `COMPLETE`
- 1 candidate / 32 wells / 32 HMM well-runs
- parent/control HMM rerun、ML model、LightGBM config、trained fold、
  booster、PF、Beam、GPU: 0
- elapsed: `39.153269533 sec`
- candidate HMM total: `9.336708374 sec`
- full 773-well projection: `225.539861659 sec`
- peak RSS: `0.415966034 GB`

Technical gateは全項目PASS:

- 32 wells / 156,088 rows / 5 folds
- duplicate / missing row 0、finite coverage 1.0
- source / manifest fold match 1.0
- forbidden geometry column、truth、roleのfreeze前read 0
- first-difference parity最大`0.0 ft`
- transition row-sum最大誤差`4.440892e-16`
- posterior normalization最大誤差`2.220446e-15`
- prediction readback logical SHA、runtime、RSS PASS

Mechanism結果:

| Scope | Candidate | exp226 geometry | 差 |
| --- | ---: | ---: | ---: |
| all32 | 13.019009088 | 9.267204778 | +3.751804309 |
| matched control 16 | 7.771561732 | 8.719886308 | -0.948324576 |
| persistent 16 | 16.592455298 | 9.768805034 | +6.823650264 |

- matched candidateは保存exp435 dz-only `17.133652291`も改善した。
- exp226 geometry比の改善foldは`2/5`。
- paired by-well delta p95は`+21.699228790 ft`、worstは
  `+24.452435654 ft`。
- mechanism gateはmatched-control 2項目だけPASS、残り5項目FAIL。
- decision:
  `stage0_fail_closed_without_same_oof_rescue`
- Stage 1 eligible: false。

主要SHA:

- prediction decompressed/logical:
  `9eead0755e11fc5093ffedff59c9f4f3aeee3c7b5755d493af498fd4589bc2d8`
- schedule manifest:
  `6d42fe8928c0f142902a3dd47679d64cdbac801eef5439c7dddb216d8e92b2d2`
- prediction manifest:
  `d190c8fb9a92eb9ed3606205fcb9be1213f227a38b9529b0f07f95be5b3fb263`
- diagnostic manifest:
  `a80b193bea1e4786cde21b1718dc1a3f172f681591d064aee04f23c8328e5646`
- well metrics:
  `e7e8950984e33644dcfaf90c54db0ed8aa327719bef2a6c70f113afd2ab9f6a8`
- input manifest:
  `2e1cc2426d53461aeb35b5f5c5fe9dd2caa2c51d18e90f8e3438d7b96058edca`

科学値、fold別値、SHA、生成物パスはKaggle logs / notebook cell outputで
確認できたため、output archiveは取得していない。初回runのみなので
deterministic anchorとは呼ばない。

最終判断:

- matched controlではgeometry transition + GR emissionが有効だったが、
  仮説対象のpersistent 16、特にfold 0/1でmode slipを大幅に悪化させた。
- exp435の大幅悪化を`-ΔZ`中心だけに帰属する仮説は不支持。
- contractどおりscale / clip / noise / emission / grid / subset / gate /
  blend / selector rescue、再実行、Stage 1、inference、submissionなしで閉じる。
- exp438 / exp439は本結果のsame-OOF救済ではなく独立仮説として扱う。
- canonical configは`run_hmm/create_prediction=false`、
  `kaggle_stage_0_authorized/run_approved=false`へ再ロックした。
