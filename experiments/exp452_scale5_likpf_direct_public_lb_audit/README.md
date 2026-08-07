# exp452_scale5_likpf_direct_public_lb_audit

## 状態

- ルート: `pf_beam`
- 状態: Public LB `8.797`確定、ユーザー提出ref `55149125`、Codex submit 0
- train-side OOF: `10.914522073`
- Public LB: `8.797`
- Private LB: -
- Kaggle kernel: `kentookumura/exp452-scale5-likpf-public-audit-inference` v1
- Submit ID: `55149125`（ユーザー提出、Codexは未提出）
- 作成日: 2026-07-30
- 親実験: `exp417_scale5_seed_aggregation_promotion_audit`

## 仮説

固定temperature-5 seed aggregationは算術平均よりOOFを`0.680376 ft`改善した。
by-well tail guardでは不採用だったが、単体Public LBでは改善が転移する可能性がある。

## 変更点

- 新しい候補は作らない。
- exp417の`likpf_scale_5_x1p0`を単体の`tvt`として出力する。
- exp413 v4のhidden-compatible raw-test generatorをparity参照にする。
- blend、selector、gate、postprocess、ML学習を行わない。

## 検証方針

- Fold: train-side evidenceは既存5-foldを読み取り専用で参照
- Group: `well_id`
- Public LB: exp434の同じSHA256 seed familyの`likpf_mean`と比較
- Leakage Check: suffix TVT/error/fold/hidden-like roleをPF生成前に読まない
- 再現性: stable SHA256 per-well × seed index、公開surface完全parity、SHA記録

## 実行入口

- 評価Notebook:
  `exp452_scale5_likpf_direct_public_lb_audit_inference.ipynb`の1本
- train Notebookはtemplate scaffoldのみで、実行・実装対象ではない
- compact self-contained sourceから生成済み。公開3 wellsの専用function testで
  exp413 v4 `likpf_scale_5`とのfloat32完全parityを確認済み
- Kaggle private CPU version 1を実行済み。`submission.csv`は`/tmp`取得後に
  submit-check PASS。ユーザーが後から外部提出し、ref `55149125`との対応を確認済み

## 結果

| メトリック | 値 |
| --- | ---: |
| OOF RMSE | 10.914522073 |
| OOF gain vs arithmetic LikPF | 0.680375810 |
| by-well delta p95 | +2.941688483 |
| worst-well delta | +25.311274575 |
| 公開参照function parity max abs | 0.0 |
| Kaggle rows / wells | 14,151 / 3 |
| fallback rows / wells | 0 / 0 |
| submit-check | PASS（FAIL 0 / WARN 0） |
| Public LB | 8.797 |
| SHA256 arithmetic LikPF control | 9.807 |
| Public LB改善 | 1.010 |

## 所見

### 良かった点

- 5/5 foldsと全固定scopeで算術平均LikPFを改善した。
- exp413 v4にhidden-compatible raw-test generationの実績がある。
- Public LBでも同一SHA256 seed familyのarithmetic LikPF controlを`1.010`改善し、
  OOFと方向が一致した。

### 悪かった点

- 257/773 wellsが悪化し、p95とworst-well guardを大きく破った。

### リスク / 注意

- exp413 Public LB `7.201`はML最終予測であり、scale-5 PF単体のLBではない。
- seedの実装差は別Monte Carlo realizationを作るため、公開parityをhard gateにする。
- Public LBを見てtemperatureやseedを変更しない。

## 次

- 追加run、rerun、再提出、LBを見たパラメータ変更は行わない。
- exp417のtail FAILを維持し、Public LB結果から自動昇格しない。
