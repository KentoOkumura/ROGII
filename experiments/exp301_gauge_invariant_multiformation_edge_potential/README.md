# exp301_gauge_invariant_multiformation_edge_potential

## 状態

- ルート: `pf_beam`
- 状態: Kaggle CPU version 2完了・Stage 0 FAILでbranch closed
- CV / Public LB / Private LB: Stage 1未実行 / 対象外 / 対象外
- Submit ID: なし
- 作成日: 2026-07-20
- 親実験: なし。独立physical family
- 比較: exp226、exp289、exp293
- 設計の正: `.steering/20260720-exp301-gauge-invariant-multiformation-edge-potential/`
- 案2/案3の正: `reserved_followup_contract.md`

## 仮説

6 formationの絶対値ではなく同一well内のedge差分だけを使えば、formation topとwell固有datumが消える。
この差分を共通2D scalar fieldとして積分し、known prefix末尾でanchorすれば、GR、既存prediction、candidate
selectorなしで未知suffix TVTの新しいdirect physical candidateを生成できる。

## 変更点

- `U=TVT+Z`とし、6 formationの`delta S`中央値だけを観測する。
- 250 ft active grid、bilinear basis、二階差分regularizerでintegrable potentialを復元する。
- outer-train-only inner edge holdoutで固定3 lambdaから選び、outer-valid truthでは調整しない。
- exp289のANCC絶対面、latent datum、known-prefix fit、fault-risk/cutを継承しない。
- direct qualityに加え、exp293 fixed12へのH512 add-one oracleでcandidate noveltyを判定する。
- 案2/案3は実装せず、開始条件と禁止事項だけを本実験配下に固定する。

## 検証方針

- Fold: exp226保存OOFの5-fold well identityを再利用。
- Group: `well_id`。outer-valid fold全体をdonor/scale/inner選択から除外。
- Score rows: `TVT_input.isna()`の3,783,989 rows / 773 wells。
- Allowed valid geometry: `MD/X/Y/Z/TVT_input`。GRとformation 6列は禁止。
- Stage 0: formation別/median6 edge identity RMSE `<=0.02 ft`、support/coverage/leakage guard。
- Stage 1 direct PASS: exp226 9.4271095966から`>=0.20 ft`改善、5/5 folds、1000+/hidden-like/p95非悪化、worst delta`<=+0.25 ft`。
- Candidate novelty PASS: exp293 fixed12 H512 oracleを`>=0.10 ft`改善、4/5 folds、strict unique-best block`>=2%`。
- 両方PASSした場合だけ、別承認後のinferenceまたは案2/案3を検討する。

## 実行入口

- 学習notebook: `exp301_gauge_invariant_multiformation_edge_potential_train.ipynb`
- 推論notebook: `exp301_gauge_invariant_multiformation_edge_potential_inference.ipynb`
- 正規train notebookにはユーザー承認済みcompact self-contained版を採用した。正規inference notebookはplaceholderのまま。
- 実装元: `exp301_gauge_invariant_multiformation_edge_potential_compact_selfcontained_train.py` / `.ipynb`。
- train notebookはStage 0を先に判定し、PASS時だけStage 1へ進む。完了後はconfigの実行gateをfalseへ戻した。
- Kaggle kernel: `kentookumura/exp301-gauge-edge-potential-train` version 2、id_no `128007163`。
- local execute、再push、inference、submissionは禁止。

## 実装内容

次をcompact self-contained train sourceへ実装した。

- outer-valid/testのgeometry-only safe loaderとforbidden-column guard。
- stride 16の6 formation edge identity、250 ft active grid、bilinear basis、4-neighbor component support監査。
- active-grid `xx/yy/xy`二階差分、component zero-mean gauge、Huber 1.345 IRLS KKT solver。
- SHA256 stable 3-way inner edge holdoutによる固定3 lambda選択と5 outer fold OOF生成。
- OOF content SHA freeze後だけのraw truth / exp226比較 / hidden-like / exp293 fixed12 H512 add-one診断。
- solution、edge/grid/solver structure、gzip raw/decompressed、prediction logical contentのSHA manifest。
- gauge shift、affine recovery、formation permutation、valid poison、fold/same-name exclusion、no-donor component、stable SHAのunit tests。

## 結果

| メトリック | 値 |
| --- | --- |
| Stage 0 identity | PASS。formation最大 `0.008132852 ft`、median6最大 `0.007869666 ft` |
| Query component donor coverage | FAIL。pooled `0.982164`、fold最小 `0.969525` |
| Active component donor coverage | FAIL。fold最小 `0.92` |
| Leakage / row identity / runtime guards | PASS |
| Stage 1 solver / Direct OOF / H512 novelty | 未実行・未生成 |
| Public / Private LB | 対象外 |

## 所見

### 確定した点

- 中核観測は6 formationのwithin-well edge differenceで、絶対formation値ではない。
- Stage 1はsmooth/no-fault/no-GR/no-MLの単一direct solverである。
- 数値contract、leakage境界、success/failure policyは実装前に固定し、Kaggle Stage 0で再現した。
- 6 formation edge identityは全foldで閾値を大きく下回った。
- 250 ft / 4-neighbor / halo 1 gridでは、donor constraintのないcomponentが全foldに残った。
- Stage 0 FAIL policyどおりsolver fitは0で、truth join、OOF、candidate noveltyへ進まなかった。
- 案2/案3は`reserved_followup_contract.md`以外の解釈で開始しない。

### 未評価の点

- donor coverageを満たす別connectivity contractでのspatial field、direct OOF、candidate novelty。

### リスク / 注意

- smooth fieldがfaultを跨ぐとlong-tailを悪化させうるが、本実験内でfault cutを救済追加しない。
- donor constraintのないquery componentは既存予測へfallbackせずtechnical FAILにする。
- 同じOOFでgrid spacing、halo、adjacencyを変えて救済しない。

## 次

本branchを閉じる。再訪は別承認・別実験のtruth-free component connectivity readoutに限定し、inference、submission、
reserved proposal 2/3は開始しない。

## 表記

用語は`KAGGLE_DIRECTION.md`の表記方針と`docs/glossary.md`に合わせ、実験名や設定名を除いて日本語優先で記録する。
