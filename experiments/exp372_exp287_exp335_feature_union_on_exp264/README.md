# exp372_exp287_exp335_feature_union_on_exp264

## 状態

- ルート: `ml_model`
- 状態: train科学gate FAILを保持したsaved-model CPU inference・scoring完了
- CV / Public LB / Private LB: `8.071563865` / `7.587` / 未確定
- 作成日: 2026-07-24
- 完了日: 2026-07-25
- スコア確認日: 2026-07-26
- 親実験: `exp264_exp263_candidate_confidence_dual_selector`
- 統合元: `exp287_fold_safe_formation_74_addonly_on_exp264` /
  `exp335_signed_residual_meta_on_exp264`

## 仮説

exp287のformation 74列とexp335のsigned residual 23列は、foldごとの強みが異なる。
両方をcorrected exp264の同じdownstream面へadd-onlyで投入すれば、単独親より安定した
pooled CV改善が得られる可能性がある。

## 変更点

- 固定: clean 273、saved compact 74、fold、target、3 LightGBM config、seed。
- 追加: saved fold-safe formation 74、saved strict-nested signed residual 23。
- 最終特徴: `273 + 74 + 74 + 23 = 444`。
- 学習量: 1 variant × 3 configs × 5 folds = 15 GPU boosters。
- 親control、単独親、selectorの再学習: 0。
- formation / signed train feature再生成: 0。

## 検証方針

- Fold: corrected exp264 outer 5 fold
- Group: well
- Score rows: `TVT_input`欠損行
- Leakage check: id/well/fold/role完全一致、manifestとpartition SHA、schema freeze前truth読込0
- 比較: 保存済みexp264 / exp287 / exp335 OOF
- promotion: pooled/fold/scopeのincremental utilityとexp264基準のtail guardをAND判定

## 実行結果

正規train NotebookをKaggle
`kentookumura/exp372-exp287-exp335-feature-union-train`へpushした。
version 1はprefit loader adapter不足の`KeyError: compact_features`でbooster開始前に停止した。
74 unique列をexp264 loaderの`compact_features`契約へ明示変換する修正後、
同一15-booster契約のversion 2をKaggle T4で再実行し、15/15 boosterを完走した。

- pooled CV: `8.071563864946972`
- best standalone exp287: `8.136708220359452`
- 改善: `0.06514435541248 ft`
- technical gate: PASS
- incremental utility gate: FAIL
- tail promotion gate: FAIL
- promotion gate: FAIL

pooled上限と4/5 fold条件はPASSしたが、`mid_250_1000`がbest standalone比
`+0.048399545 ft`で固定上限`+0.02 ft`を超えた。tailはexp264比by-well p95
`+2.198026177 ft`、worst `fb03ae90 +13.023263266 ft`で、clean273比悪化well数も
`+1/+3/+5 ft = 157/53/23`と全上限を超えた。

詳細は`result.md`と`kaggle/output/train_v2/artifacts/metrics.json`を正とする。

## 実装・成果物

Jupytext起点の正規train Notebookと`src/feature_union_pipeline.py`は、
target/errorを開く前の444列schema freeze、3入力manifestと全partition SHA、
formation logical float32 SHA、`id/well/fold/role` alignment、15 model slot、
technical / incremental utility / tail promotionの固定AND gateを含む。

version 2 outputはOOF、metrics、model manifest、SHAの実ファイル監査に必要なため取得した。
主要10成果物と15 model fileのSHAはすべてmanifestと一致した。

- feature schema SHA:
  `049800d626b04f16fbf08eb33e8a980ecbe62008402ff7b24f3e77e04e6ef4e9`
- model manifest SHA:
  `e0d7f85c34d5c64410fe1b2e641669ee1887346a4cbd754579d0dd7e15875b5a`
- OOF SHA:
  `635dea78b9bf7ad07a1bef267d37e4e2d1707f648799c1590715d4255c02e6f8`

## 推論結果

Kaggle CPU canonical inference version 4を完了した。raw test 3 wells / 14,151 rowsから
12候補、88 selector特徴、clean273、saved74、formation74、signed23を再生成し、
40 parent selector、20 signed selector、15 union TVT modelで予測した。model fitは0。

- runtime: `459.376 sec`
- 最終特徴: `273 + 74 + 74 + 23 = 444`
- 予測範囲 / 平均 / 標準偏差:
  `11591.696289–12239.309570 / 11905.273438 / 278.501831`
- submit-check: PASS（14,151行、`id,tvt`、ID内容・順序一致、重複/NaN/Inf 0）
- submission SHA:
  `3688de824db2ae0ff1002fb9c2c9ed8543ed09d4e5bbfdd45d7bbf3c9c7eacdd`
- prediction decompressed SHA:
  `5f18bcaf8cdd6952652155c6029c8045272b0b052a69ac8157bbf170aad4bc54`
- Code submission: `ref 54975325`、`COMPLETE`、Public LB `7.587`
- 提出時刻: `2026-07-25 12:28:12.460000 UTC`
- 外部submitはユーザー確認後に観測したもので、Codexは実行していない

## リスク / 注意

- 平均CVの相補性は確認できたが、fold 1とmid-rangeで単独親より不安定だった。
- exp287とexp335で既知だったworst-well riskはunionでも解消せず、さらに拡大した。
- Public LBはexp335 `7.517`より`+0.070`、exp287 `7.530`より`+0.057`悪化した。
- GPU LightGBMのbitwise reproducibilityは主張しない。
- scientific FAIL後の同じOOFでのfeature/config/weight/gate救済は禁止する。

## 所見

formationとsignedの両familyは5/5 foldsでpositive gainを持ち、pooled CVも明確に改善したため、
unionの平均的相補性は支持された。しかし固定scopeとtailの安全条件を同時に満たせず、
train-side promotionには不適格である。実際のPublic LBも両単独親より悪化し、
CVの平均改善はPublic LBへ転移しなかった。

## 次

推論、提出形式検証、scoring記録を完了した。trainの科学FAILとsame-OOF rescue禁止を維持し、
exp335 `7.517`をML Public-LB anchorとして継続する。exp372の追加提出は行わない。
