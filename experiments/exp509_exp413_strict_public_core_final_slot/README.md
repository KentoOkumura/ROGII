# exp509_exp413_strict_public_core_final_slot

## 状態

- ルート: `ensemble`
- 状態: prediction-only候補実装済み、正規notebook未採用、未実行
- 最終提出枠: 第1枠、private一般化優先
- CV: 新規実行なし。参照candidate `7.874488150`、exp413 `7.884802794`
- Public LB: 未提出。exp413参照値`7.201`
- Private LB: なし
- Submit ID: なし
- 作成日: 2026-08-04
- 親実験: `exp413_scale5_likpf_full_replacement_on_exp335`
- auxiliary: `exp497_strict_public_core_fold_safe_ensemble_on_exp413`

## 仮説

exp497 strict public-coreは科学的promotion gateを通過しなかったが、5つのmeta-fold係数は
すべて正で、exp413と異なるtrajectoryの平均的相補性は確認できた。最終提出をexp413単独に
しないというportfolio判断の下では、事前固定deployment係数の中央値`0.13716473330712417`
だけを使う小blendが、公開系の補完性を取り込む最も保守的な構成になる。

```text
prediction = 0.8628352666928758 * exp413
           + 0.13716473330712417 * strict_public_core
```

## 変更点

- 保存済みexp413 predictionとexp497 strict public-core predictionだけを使う。
- exp413/exp497のmodel、PF/Beam、feature、selectorを再学習・再生成しない。
- Gold、guarded contact、same-well lookup、router、最終postprocessを入れない。
- exp497のgate FAILを維持し、reference-only final-slot overrideとして扱う。

## 検証方針

- 新規CV: なし。exp497 Stage Eのmeta5 OOFとFAIL判定を再利用する。
- Technical: input SHA、ID one-to-one、sample order、finite、固定weight、formula parityの全AND。
- Leakage: strict public-core component以外のpublic output/overlayをfail-closedで拒否する。
- Selection: 差分readoutやPublic LBでweightを変えない。

## 実装

- Jupytext起点の`*_compact_selfcontained_inference.py/.ipynb`を実装した。
- raw hidden testからexp413とstrict public-coreを動的再生成し、保存済みboosterだけで推論する。
- exp413が途中で作る`submission.csv`は`artifacts/exp413_intermediate_submission.csv`へ隔離する。
- exp497 v1で一律`0.001 ft`を超えたparityを、strict `0.002`、exp413/blend `0.02 ft`の
  component別historical-equivalence契約へ分離した。これは候補値やweightを変えない。
- 最終式はcomponentをCSV境界で読み戻した後、float64で1回だけ評価する。
- component予測、差分readout、入力manifest、再現性manifest、`submission.csv`をtechnical
  gate通過後だけ生成する。外部提出処理は持たない。

## 実行量

- scientific variant: 1
- 新規model/config/fold/booster: `0 / 0 / 0 / 0`
- 読み込む保存model: exp497 booster 40 + Ridge 2、exp413 booster 75
- PF/Beam: hidden test componentの動的再生成だけ。新規探索・再fitは0
- 親/control再学習: `0`

## 検証

- 専用test: `6 passed`
- 依存するexp497 test: `30 passed`
- Jupytext round-trip、`py_compile`、Ruff `F821/F401/F811/E501`: PASS
- strict experiment validation、template validation: PASS
- bootstrap dependency 24 files: 欠落0
- 親compact比較: exp497 343行/7章に対し、exp509候補694行/8章
- repository-wide: `1861 passed / 8 skipped / 4 failed`。失敗4件は既知の対象外exp293/296。

## 現在のblocker

- 正規`*_inference.ipynb`への採用、Kaggle package/push/run、output取得、提出は未承認。
- 初回Kaggle実行前なので、hidden/current-testの生成物SHAとtechnical gateは未評価。

## 所見

- positive evidenceはexp497の5/5 positive meta weightと小幅pooled改善。
- negative evidenceはfold、hidden-like、well-tail gate FAILであり、科学的anchorへは昇格しない。
- 最終portfolio第1枠としてのみ、固定小weightとtechnical gateを条件に利用する。

## 参照ファイル

- `config.yaml`: 固定weight、入力SHA、禁止事項、technical gate
- `ensemble_contract.yaml`: blendの機械可読契約
- `output_contract.md`: 将来生成物の契約
- `SESSION_NOTES.md`: 設計判断とblocker
- steering: `../../.steering/20260804-exp509-exp413-strict-public-core-final-slot/`

## 次

別承認があれば候補を正規inference notebookへ採用し、Kaggle T4/private/internet-off packageの
bootstrap readbackを行う。push前inventoryは新規booster 0、保存booster 115、Ridge 2で再確認する。
