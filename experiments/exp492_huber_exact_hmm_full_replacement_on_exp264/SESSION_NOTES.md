# exp492 セッションノート

## 目的

exp389 Huber exact HMMをexp264の元の`exact_hmm`と全面置換し、
総数12候補のままcorrected strict nested dual selectorを評価する。

## 現在の状態

- Route: `ensemble`
- 状態: Stage C完了、scientific gate FAIL、post-readout error、branch close
- CV: `8.639368546`（parent比`-0.013163410 ft`、3/5 folds改善）
- LB: 未提出
- 実行コード: 別名Jupytext source + ipynb候補、共有helper、専用test
- executable notebook candidate: 1
- canonical executable notebook: 1

## 設計時点の実行量

- active variant: 1
- LightGBM objectives: 2
- outer folds: 5
- inner folds: 4
- planned CPU selector boosters: 40
- trained boosters: 40
- 親/control再学習: 0
- GPU boosters: 0
- downstream TVT / inference / submission: 0 / 0 / 0

## 変更契約

- 12 candidate ID/order/domainを維持する。
- changed 4:
  `exact_hmm`、`exp226_k16__exact_hmm`、`likpf_mean__exact_hmm`、
  `exp226_w500_50_50`
- unchanged 8はparent値と完全parityを要求する。
- exp264 corrected Stage A 88列schemaを同じ名前・順序で使う。
- exp389をglobal key join後にexp263 selector foldへ再分割する。

## 設計根拠

- exp389: `11.938287235 -> 11.852741130`、gain `0.085546105 ft`、5/5 folds。
- tail: by-well p95 `+0.002234351 ft`、worst `+1.750248202 ft`で不合格。
- exp392 fixed13: `8.652531956 -> 8.769791682`、`+0.117259726 ft`、
  2/5 folds。Huber top1 91,035 rowsでも既存候補rerankingが観測された。

## 再現性メモ

- seed: 42
- sampling: stable SHA256 keys
- HMM/PF/Beam再生成: 0
- exp389 decompressed SHA:
  `f5d44d9d9ee380bb7ea408006030363efbe8fcdb3573cfa18031b2d31c617f90`
- parent feature schema logical SHA:
  `aaef4ffdd90667893b099b76a52f1957b22197aea9cee5e5b57bc81048ddd3a4`
- model / outer-valid score SHA: post-readout ERROR後のoutputで大きい生成物が
  保持されず未回収
- scientific gate SHA:
  `63f59978f4d28deb1044d0799d49bd461fc2c1719ca9eab45b4a6680af9d2ce1`
- deterministic anchor: false。独立再実行一致前は昇格しない。

## コマンドログ

2026-07-30に`make new-steering`と`make new-exp`でscaffoldだけを作成した。

2026-07-31のユーザー依頼で実装を承認された。次を実装した。

- `src/exact_hmm_full_replacement.py`
- `exp492_huber_exact_hmm_full_replacement_on_exp264_compact_selfcontained_train.py`
- 対応する別名compact ipynb候補
- `tests/test_exp492_huber_exact_hmm_full_replacement.py`

実装時検証:

- `py_compile`: PASS
- Ruff `F821`: PASS
- 専用test: `9 passed`
- 関連exp264/exp392を含むtest: `36 passed`
- Jupytext round-trip `--test`: PASS
- `make validate-exp EXP=exp492_huber_exact_hmm_full_replacement_on_exp264`: strict PASS
- `__file__` notebook-safe check: PASS
- canonical train notebook markdown-only維持: PASS

Notebook構成比較:

- 親exp264 train source: 7章 / 465行
- exp392 fixed13 compact source: 8章 / 540行
- exp492 compact候補: 9章 / 625行
- exp492ではauthorization、固定入力/SHA、fixed12 Stage A、Stage C、
  saved control科学readout、feature importance、再現性保存をNotebook上で追える。

Kaggle package作成・push済み。train version 1はterminal ERROR。

2026-07-31のユーザー依頼「実行してください」を、次の固定scopeに対する
canonical採用、Kaggle package、private CPU train push/runの明示承認として記録した。

- approved scope:
  `fixed12_huber_replacement_stage_a_stage_c_1_variant_2_objectives_outer5_inner4_40_cpu_boosters_no_control_retraining`
- active variant: 1
- LightGBM objectives / configs: 2
- outer / inner folds: 5 / 4
- total CPU selector boosters: 40
- saved parent/control retraining: 0
- GPU booster: 0
- downstream TVT / inference / submission: 0 / 0 / 0
- canonical kernel:
  `kentookumura/exp492-huber-exact-hmm-full-replace-exp264-train`
- accelerator: CPU
- internet: disabled

この承認でcompact self-contained候補をcanonical train notebookへ採用する。
`kaggle-review-exp`のコストガードに従い、40 booster以外は実行しない。

Kaggle package作成:

```bash
make prepare-kaggle-notebooks \
  EXP=exp492_huber_exact_hmm_full_replacement_on_exp264 \
  EXTRA_ARGS="--notebook train \
  --kernel-id kentookumura/exp492-huber-exact-hmm-full-replace-exp264-train \
  --title 'exp492 huber exact hmm full replace exp264 train' \
  --run-on-push --strict"
```

- metadata: private、CPU、internet disabled、run_on_push
- competition source: `rogii-wellbore-geology-prediction`
- kernel sources: exp263 / exp389 / exp264
- generated notebook: 21 cells / 10 code cells（bootstrap含む）

初回は実験ディレクトリ名を非短縮のまま使った55文字の
`exp492-huber-exact-hmm-full-replacement-on-exp264-train`へpushしたが、
Kaggle `SaveKernel`が詳細なし400を返した。title由来slugとの一致は確認済み。
同slugを`kernels pull -m`すると403で、Kaggle側にkernelは作成されていなかった。
Kaggleの50文字制約へ合わせ、意味のある要素を保持した48文字のcanonical
`exp492-huber-exact-hmm-full-replace-exp264-train`へ短縮し、packageを再生成する。

再package後のstrict validationと専用test 9件はPASS。次でpushした。

```bash
make push-kaggle-train EXP=exp492_huber_exact_hmm_full_replacement_on_exp264
```

- push result: `Kernel version 1 successfully pushed`
- kernel URL:
  `https://www.kaggle.com/code/kentookumura/exp492-huber-exact-hmm-full-replace-exp264-train`
- Kaggle `id_no`: `129217774`
- version: `1`
- pushed at: `2026-07-30 23:17:16 UTC`
- pulled metadata: private / CPU / internet disabled / machine_shape `None`
- output archive: 科学gate、scope、usage、by-well、schema SHA回収のため取得

Kaggle監視:

- RUNNINGを同じslugで監視し、途中のCLI logs空は既知挙動として再pushしなかった。
- 一時的なDNS失敗後、`KernelWorkerStatus.ERROR`を確認。
- version 1の通常logsを取得した。
- notebook elapsed `6112.213 sec`で、Stage C 40/40 boosterと科学readout後に停止。

ERROR原因:

```text
KeyError: 'Column not found: gain'
```

`nested_feature_importance_by_objective_outer_inner.csv`は
`importance_type` / `importance`のlong-formだが、Notebookの事後診断セルが
`gain`列を期待していた。科学gate保存後のfeature importanceセルだけの不具合。
canonical sourceは`importance_type == "gain"`をfilterして`importance`を平均する
実装へ修正し、Jupytext round-trip、`py_compile`、Ruff F821を再度PASSした。

## Kaggle version 1結果

- technical checks: all PASS
- selector score guard: PASS
- hard primary: `8.652531955610227 -> 8.63936854552658`
- delta: `-0.01316341008364752 ft`
- improved folds: `3/5`（必要`4/5`）
- fold delta:
  `[-0.102580979, -0.007842910, +0.052712491, +0.037968822, -0.044357407]`
- near 0--250 delta: `-0.012444018 ft`
- distance 1000+ delta: `-0.014342934 ft`
- hidden-like最大delta: `-0.096364940 ft`
- by-well p95 delta: `+0.381470357 ft`（上限`+0.25`）
- worst well: `d2f3b1ab`
- worst-well delta: `+4.254514134 ft`（上限`+0.25`）
- changed-family top1: `937,102 / 3,783,989 = 24.764924%`
- fixed fallback report-only:
  `8.238331546 -> 8.222215557`（`-0.016115990 ft`）
- decision: `FAIL_CLOSE_FIXED12_HUBER_REPLACEMENT_SELECTOR`

回収した主要SHA:

- input contract:
  `3bdbed3fc12ac387e5c635910af4fb805f7233a944f38b443b7ee67962a44ebe`
- feature schema:
  `b91ec1517a82641fe4d96f41c97872151f273a8bbfcb537284f91d47aacf1035`
- compact schema:
  `e3a677610899cb33bf58262f4cf02f650300c8c2207c46b53588d3418162ea74`
- nested compact manifest:
  `0fcbf89d0b774d85c1b210c0900a2c264b3d2b77cc14dd0f041b55843b8f1cf3`
- scientific gate:
  `63f59978f4d28deb1044d0799d49bd461fc2c1719ca9eab45b4a6680af9d2ce1`

科学結論は回収済み。追加versionはさらに40 CPU boosterを学習し、承認scopeを
超える。加えて凍結gateがFAILしているため、version 2はpushしない。

## 次のアクション

branchを閉じ、weight、threshold、domain、gate救済、downstream、inference、
submission、追加rerunを行わない。exp493の独立fixed12 Student-t結果を待ち、
well-tail不安定性を回避するtarget-free continuous risk readoutへ進むか判断する。
