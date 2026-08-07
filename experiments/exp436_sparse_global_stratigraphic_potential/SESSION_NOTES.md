# exp436_sparse_global_stratigraphic_potential セッションノート

## 目的

outer-trainのformation contactから6つの`U_k(X,Y)` global surfacesを復元し、
固定formation集合のpotential差だけで状態遷移を作る物理候補を実装する。

## 現在の状態

- Route: `pf_beam`
- 状態: `stage0_fail_closed`
- CV / LB: なし
- 実装承認: 2026-07-29のユーザー指示`exp436を実装してください`
- 実行承認: Stage 0のみを実行済み。将来のrun lockは再び閉鎖
- compact self-contained train候補 / 11 contract tests: 実装済み
- 正規train notebook: compact候補を採用
- 正規inference notebook: markdown-only placeholderを維持
- package / push / run: Kaggle private CPU version 2まで実施済み
- Stage 1 / Stage 2 / inference / submission: 未実行・未承認

## 2026-07-29 訂正セッション

ユーザー確認により、当初作成した単一`P(X,Y)`は第2案を正しく表していないと判定した。
設計を次の地層面別契約へ訂正した。

- formationは`ANCC / ASTNU / ASTNL / EGFDU / EGFDL / BUDA`の6面。
- sourceはouter-trainのfirst contactだけ。outer-validは完全除外。
- contact値は`U_contact=TVT_contact+Z_contact`。
- 1 global sparse surface / formation / fold、合計30 field fits。
- target生formation列とGRは使わない。
- anchorと全control pointsでsupportを満たす固定formation集合`K_w`、最低4面。
- primaryは`K_w`のanchor差を等重み平均し、row-wise k切替を禁止。
- Stage 0 resource/integrity、Stage 1 prefix rolling-origin、
  Stage 2 truth-late direct OOFの順序とAND gateを固定。

## 予定実行量

- scientific candidate: 1
- single-formation report-only paths: 6
- reporting folds / global surface fits: `5 / 30`
- sparse solve: 初期L2 + Huber更新5回、最大180
- fitted ML model / LightGBM config / trained ML fold / booster: `0 / 0 / 0 / 0`
- HMM / PF / Beam / GPU: `0 / 0 / 0 / 0`
- parent exp226 control再生成: 0

この量は設計値であり、push前に再確認する。

## 2026-07-29 実装セッション

ユーザーの明示指示:

```text
exp436を実装してください
```

実装内容:

- Jupytext percent形式の
  `exp436_sparse_global_stratigraphic_potential_compact_selfcontained_train.py`
  を作成した。正規notebook placeholderは明示採用前なので上書きしていない。
- exp226保存OOFはpre-freezeに
  `well_id,row_idx,suffix_offset,fold`だけを`usecols`で読み、decompressed SHAを
  固定した。`tvt_true/tvt_pred`はcandidate bundle freeze後だけ読む。
- source first contactはMD昇順の最初のexact zero / sign crossingを線形補間し、
  `U_contact=TVT_contact+Z_contact`を1 well ×1 formation最大1 nodeにした。
- graphは同一formation内の別well 8近傍、4,000 ft以内、Gaussian weightの
  stable undirected unionとし、formation間edgeを作らない。
- surfaceはformation/foldごとにHuber observation、edge smoothness、weighted
  graph-Laplacian bending、ridgeを組み、初期L2 + 5 IRLS updatesをSciPy LSQRで解く。
- targetは64 ft control pointsで各surfaceを最大16 / 最低8 source wellsから
  Gaussian queryし、anchorと全control pointsでsupportを持つ固定`K_w`だけを使う。
- primaryは`K_w`のpotential anchor差を等重み平均する。6 formation pathは
  report-onlyで、row-wise switch、fallback、selector、blendはない。
- Stage 0は全fold contact/node/edge/component census、fold 0の6面 solve、
  固定16 wells queryとfull runtime/RSS projectionを行う。
- Stage 1は既知prefix最後512 ftのrolling-originをconstant-`U` nullと比較する。
- Stage 2はStage 1 PASSと別承認後だけsuffix truth / exp226 control /
  hidden-like rolesをlate joinし、固定scopeとby-well tailをAND判定する。
- Stage 0 / 1 / 2の実行承認フラグを独立させ、未承認段階へ進まない。

contract tests:

- implementation/run lockと実行量
- first exact/sign-crossing、`U=TVT+Z`
- same-formation stable graph / distance / duplicate
- Huber IRLS + LSQR smooth field
- query source supportと固定6面等重みanchor差
- target allowlist / truth-late ledger
- Stage 1 / Stage 2 AND gate
- notebook-safe pathとread-time allowlist

専用pytestは`10 passed`。構文、Ruff F821、Jupytext変換/testもPASSした。

親compact比較:

- 親exp226にはcompact self-contained train sourceがない。
- exp436 compact trainは2,682行、10章、11 code / 12 markdown cellsで、
  Imports、path/SHA/role-read、
  input contract、contact/graph、sparse solver、fixed-support query、
  Stage 0、Stage 1 freeze、Stage 2 truth-late、guarded orchestrationを持つ。
- 同一exp helper import、`__file__`、薄い`main()` entrypointはない。

実装時の実行量contract:

| 項目 | 値 |
| --- | ---: |
| scientific candidate | 1 |
| formation report-only path | 6 |
| reporting folds / global fields | 5 / 30 |
| sparse solves最大 | 180 |
| parent/control再生成 | 0 |
| ML model / LightGBM config / trained fold / booster | 0 / 0 / 0 / 0 |
| HMM / PF / Beam / GPU | 0 / 0 / 0 / 0 |

## コマンドログ

初回設計ではsteering、config、記録文書、markdown-only notebook placeholderだけを
更新した。今回、compact train候補、専用test、設定・記録更新、静的検証を追加した。
Kaggle train/inference、package/push/runは実行していない。

全体`pytest -q`も起動したが、exp436 testの実行前のcollectionで既存
`exp297 / exp301 / exp333 / exp336 / exp349`の5 moduleが各自のconfig contract
不一致により停止した。exp436専用10 tests、strict experiment validation、
template validationは独立にPASSしており、この5件の既存実験ファイルは変更していない。

## 再現性メモ

- seed policy: RNGなし、固定fold/formation/well/contact/node/edge/query順。
- stochastic components: なし。
- CPU/GPU runtime: Kaggle CPU version 2完了、GPU off、internet off。
- input / contact / graph / solver / decision SHA: version 2で生成・記録済み。
- query / prediction SHA: 6面contract不成立のため0-row生成物だけを記録。
- deterministic anchor: 同一設定rerunのlogical content SHA一致まで主張しない。
- model / submission SHA: 対象外。

## 解釈

Stage 0は入力/leakage/resourceをPASSしたが、BUDA first contactが全fold 4–6 wellsで
固定最小32をFAILした。6つの`U_k`をすべて必要とするexp436 contractは成立せず、
科学評価前にbranchを閉じた。

## 次

exp436は再実行せず、Stage 1以降へ進まない。固定5面案は別仮説・別承認とする。

## 2026-07-29 Stage 0実行承認

ユーザーの明示指示:

```text
実行してください
```

この指示を、compact train候補の正規train notebook採用、Kaggle CPU package作成、
canonical kernelへのpush、Stage 0実行、完了監視の承認として扱う。Stage 1、Stage 2、
inference、submissionは承認対象に含めず、ロックしたままにする。

push前の実行量再確認:

| 項目 | 実行量 |
| --- | ---: |
| scientific candidate | 1 |
| formation report-only path | 6 |
| reporting folds | 5 |
| Stage 0で実際にsolveするfold / field | fold 0 / 6 |
| Stage 0 sparse solve | 36（6 fields × 初期L2+5 IRLS） |
| full Stage 0 projection上のglobal field / sparse solve上限 | 30 / 180 |
| ML model / LightGBM config / trained ML fold / booster | 0 / 0 / 0 / 0 |
| HMM / PF / Beam / GPU | 0 / 0 / 0 / 0 |
| parent exp226 control再生成 | 0 |

認証確認ではKaggle CLI用OAuth credentialとlegacy credentialを検出した。
headless API tokenは未設定だが、今回使用するKaggle CLIのOAuth認証は利用可能。

実行:

```text
make prepare-kaggle-notebooks EXP=exp436_sparse_global_stratigraphic_potential \
  EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp436-sparse-global-stratigraphic-potential-train --title 'exp436 sparse global stratigraphic potential train' --run-on-push --strict"
make push-kaggle-train EXP=exp436_sparse_global_stratigraphic_potential
```

- strict package作成: PASS
- push: Kaggle kernel version 1
- kernel id / id_no:
  `kentookumura/exp436-sparse-global-stratigraphic-potential-train` / `129058940`
- private / GPU / internet: `true / false / false`
- kernel sources: exp226 train、exp115 hidden-like assignment train
- version 1状態: `ERROR`

version 1の最初の意味のあるtraceback:

```text
ValueError: surface has 5 source wells, below 32
```

分類はcode-side fail-close処理不足。32 wellsという固定gateを緩める対象ではなく、
data support不足をStage 0 `FAIL`として保存すべき箇所が例外終了していた。
formationごとのpreflight solveを例外捕捉し、失敗理由、nodes、edges、components、
sparse solve数0をsolver manifestへ残す。6面すべて揃わなければtarget queryを
実行せず、coverage/support gateを0として正常なfail-closed decisionを保存する。
なおlogsでYAMLの未引用`null` keyがPython `None` keyになっていたため、Stage 1用keyを
文字列`"null"`へ修正する。Stage 0/1/2の条件値は変更しない。

## 2026-07-29 Kaggle Stage 0 version 2完了

- kernel:
  `kentookumura/exp436-sparse-global-stratigraphic-potential-train`
- version / id_no / terminal: `2 / 129058940 / COMPLETE`
- notebook log上の最終status: `stage0_fail_closed`
- Stage 1 / Stage 2 / inference / submission: `null / null / false / false`
- 将来のpackage / push / execution / Stage 0 authorizationはfalseへ戻した。

Stage 0判定:

| gate | 観測値 | 判定 |
| --- | ---: | --- |
| rows / wells / folds | 3,783,989 / 773 / 5 | PASS |
| source-valid overlap | 0 | PASS |
| target GR / formation / suffix truth reads | 0 / 0 / 0 | PASS |
| duplicate contact/node/edge keys | 0 | PASS |
| finite source coverage | 1.0 | PASS |
| source wells / formation最小 | 4（下限32） | FAIL |
| fold 0 surface solve成功率 | 5/6 = 0.833333 | FAIL |
| queried wells / query rows | 0 / 0 | FAIL（6面未成立のため未実行） |
| reported projected runtime（query未実行） | 175.435738 sec | PASS・参考値 |
| projected peak RSS | 0.549873 GB | PASS |
| sparse solves | 30（5面 ×6） | 上限36内 |

formation別source contact wells（fold 0–4）:

| formation | wells |
| --- | --- |
| ANCC | 596 / 598 / 595 / 596 / 595 |
| ASTNU | 615 / 614 / 615 / 616 / 616 |
| ASTNL | 616 / 616 / 616 / 618 / 618 |
| EGFDU | 592 / 594 / 596 / 595 / 595 |
| EGFDL | 557 / 555 / 561 / 558 / 557 |
| BUDA | 5 / 4 / 4 / 5 / 6 |

fold 0ではANCC–EGFDLの5面が各6 sparse solvesを完了し、BUDAは
`surface has 5 source wells, below 32`として0 solveでmanifestへ保存された。
固定6面contractが揃わないためqueryは実行していない。formation除外、contact定義、
minimum support、aggregationを変更せず、事前登録policyどおり閉じた。
runtime値はcensusと5面solveを主に反映し、full query込みの投影ではないため、
promotion根拠には使わない。

生成物証拠:

- artifact bundle SHA256:
  `5b1a20bda01409bb562c241611f023568ab811f9be8683ff7456f212a56da6d2`
- Stage 0 decision SHA256:
  `af50b420c899c58854a6721b1fdf011b05472746eff86871739f54af5275fbd8`
- contact census SHA256:
  `a3bf39a16f43977435e3d3b837245eacbbe69f53bdb477378587c80824e46b0e`
- solver manifest SHA256:
  `2ce1068b8a787c126e714b792f1b115397fd8a8fdc5962a7a92dec9448bfffde`
- fold identity decompressed SHA256:
  `7bd19d46c8608c18c8bb0238b7bce7d55e897746bd3442ce0e4d69fb96702817`

Kaggle output archive全体は取得せず、実ファイル確認が必要な小さい
`metrics.json`、Stage 0 decision、contact census、solver manifestだけを取得した。

解釈:

5面のsource supportとCPU resourceは十分だったが、BUDA first contactだけが全foldで
4–6 wellsと極端に疎で、6面global potential familyは実行可能条件を満たさない。
exp436を再実行・救済しない。BUDAを事前除外する固定5面contractを検討する場合は、
target-free evidenceに基づく別実験・別承認とする。
