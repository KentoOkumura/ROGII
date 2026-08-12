# 要件

## 依頼

exp209 exact HMM の位置状態を、モデルを変えずに
`TVT_t` から `U_t=TVT_t+Z_t` へ単純に座標変換し、親と同じ結果になることを
確認する。

`exp445_tvt_to_u_coordinate_parity_exact_hmm` として、今回の作業範囲は
アイデアバックログ、steering 3文書、実験ディレクトリ、設定・記録文書の
作成までとする。実装、正規Notebookの変更、Kaggle package / push / run、
inference、submissionは行わない。

2026-07-29の追加依頼「exp445を実装してください」により、作業範囲を
compact self-contained train候補、inference禁止guard、専用testの実装まで
拡張した。正規Notebook採用、Kaggle package / push / run、inference、
submissionは引き続き未承認である。

2026-07-30の追加依頼「実行してください」により、compact候補の正規Notebook
採用、Kaggle private CPU package / push、fixed32 Stage 0の1回実行までを
承認済みとする。inferenceとsubmissionは引き続き未承認である。

## 検証仮説

親の固定TVT格子を`P_j`、既知の各rowのZを`Z_t`とする。
candidateのU状態値をrowごとに

```text
U_t,j = P_j + Z_t
```

と定義すれば、`U_t,j-Z_t=P_j`であるため、これは離散state indexを含めた
厳密な再ラベルである。親とcandidateで、emission、初期prior、rate process、
position/rate transition、forward/backward message、log-likelihood、
smoothed posterior、TVT readoutが一致しなければならない。

candidateは固定absolute-U格子を使ったexp438とは異なる。`P_j+Z_last`を全rowで
固定せず、同じindex `j` のU状態値を既知の`Z_t`だけ平行移動する。

## 制約

- Route: `pf_beam`
- 親: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- 比較参照: `exp438_u_state_fixed_lattice_exact_hmm`
- 科学variant: 1つのcoordinate-parity candidateだけ。
- 親の固定TVT index格子、0.35 ft step、cell数、band、開始indexを固定する。
- rate状態は親と同じ`r_U=d(TVT+Z)/dMD`を使う。
- row-dependent U値は`P_j+Z_t`で決定し、data-driven regrid、補間transport、
  re-anchor、absolute-U固定を行わない。
- 親のindex-space position mean
  `r_current*delta_MD-delta_Z`をcandidateでも維持する。これはmoving U grid上の
  physical edge
  `(U_t,k-U_(t-1),j)-r_current*delta_MD`
  と厳密に同じkernelである。
- GR emission、missing処理、prefix calibration、prior、noise、rate transition、
  arrival-rate積分、5-cell support、forward-backward、posterior meanを変更しない。
- suffix truth、fold、well role、episode label、errorはparity判定に使用しない。
- CV、RMSE改善、LB改善、promotion evidenceを生成しない。
- 再現性: `docs/06_reproducibility.md`に従い、入力、格子、kernel、posterior、
  predictionのlogical/content SHAを記録する。
- compact self-contained候補と専用testの実装は承認済み。
- 正規Notebook採用、Kaggle package / push / Stage 0 v2は追加承認に基づき
  完了済み。run flagは完了後に再ロックした。

## 禁止事項

- exp438の固定absolute-U格子`P_j+Z_last`。
- candidate index transitionを`r_current*delta_MD`へ変えること。
- state数、grid step / phase / band、rate、noise、emission、prior、readoutの変更。
- interpolation、row-adaptive cell追加・削除、boundary rescue、fallback。
- truth/error/foldに基づくwell・row選択。
- blend、selector、PF、Beam、ML、inference、submission。
- parity FAILをparameter調整や許容値緩和で救済すること。

## 受け入れ基準

- steering、実験scaffold、config、記録文書、バックログが
  `exp445_tvt_to_u_coordinate_parity_exact_hmm`で整合する。
- `experiment.route=pf_beam`を明記する。
- `U_t,j=P_j+Z_t`と、candidateのindex-space transitionが親と同じ
  `r_current*delta_MD-delta_Z`になる理由を明記する。
- exp438との差を「fixed absolute U」対「row-shifted coordinate relabel」として
  明記する。
- 将来のtechnical parity gateを事前固定し、性能gateを置かない。
- 予定実行量を、candidate 32 + paired parent 32 = 64 HMM well-runs、
  1 reporting-free manifest、ML / booster / PF / Beam / GPU各0として記録する。
- compact self-contained候補を正規notebookへ採用し、Kaggle Stage 0で
  16/16 technical gatesをPASSする。
- deterministic anchorと呼ぶのは、独立rerunでinput / posterior /
  prediction SHAが一致した後だけとする。
- gzip生成物を比較する場合はraw `.csv.gz` SHAではなくdecompressed content
  SHAを主証拠にする。

## 次

fixed32 Stage 0 v2まで完了。初回成功runはdeterministic anchorとせず、
独立rerunは別承認がある場合だけ行う。inference / submissionへは進まない。
