# exp349_exp287_u_boundary_continuity_fade

## 状態

- ルート: `ml_model`
- 状態: `kaggle_cpu_v2_complete_fail_close_no_rescue`
- CV: 親`8.136708220` → 候補`8.135096925`（改善`0.001611295 ft`）
- Public / Private LB: 未提出
- 作成日: 2026-07-22
- 親実験: `exp287_fold_safe_formation_74_addonly_on_exp264`

## 仮説

exp287の未知suffix先頭で`U = TVT + Z`に局所的なdatum段差が残っているなら、現在wellの既知prefix末端だけを基準に、最大8 ftの逆向き補正を240 MD-ftで指数減衰させることで、遠方形状をほぼ維持しながら境界近傍RMSEを改善できる。

## 変更点

保存済みexp287 OOFへ、次の固定式を1 variantだけ適用する設計である。

```text
gap_U = parent_pred[first_hidden] + Z[first_hidden]
        - (TVT_input[last_visible] + Z[last_visible])
move(d) = -clip(gap_U, -8, 8) * exp(-d / 240)
candidate(d) = parent_pred(d) + move(d)
```

公開6.594系Notebookのcontact reconstruction、same-well overlap、PF/Beam、model-package、branch hedgeは使用していない。固定式だけをKaggle private CPUで監査した。

## 検証方針

- Fold: exp287保存OOFのouter 5 foldsを再利用する。
- Group: well。
- 親: exp287 OOF 3,783,989 rows / 773 wells / CV 8.136708220、SHA固定。
- Leakage Check: 親予測とraw `MD/Z/TVT_input`だけでcandidateとdiagnosticを作り、SHA freeze後にのみ`actual_tvt`をjoinする。
- Technical: suffix／ID parity、finite、最大8 ft、単調減衰、境界U gap非増加、SHA readbackを全件要求する。
- Scientific: pooled `>=0.020 ft`、4/5 folds、0--240 ft `>=0.050 ft`改善に加え、遠方、hidden-like、p95、worst-wellの非悪化guardをAND評価する。
- FAIL時: cap/tau/threshold/parent/gridを救済せず閉じる。

## 実行入口

`*_compact_selfcontained_train.py/.ipynb`にStage A target-free生成／SHA freezeとStage B late-truth評価を実装し、正規train Notebookへ採用した。Kaggle kernel version 2で完了した。`*_compact_selfcontained_inference.py/.ipynb`はfail-closedのままで、predictionとsubmissionは生成していない。

## 結果

- 3,783,989 rows / 773 wells、technical gateは全PASS。
- 5/5 folds改善、0--240 MD-ftは`0.110004 ft`改善。
- hidden-like spatial / typewell-purgedは`0.002099 / 0.002240 ft`改善。
- by-well median / p95 / worst deltaは`-0.000363 / +0.010451 / +0.063651 ft`。
- pooled改善は`0.001611 ft`で、事前下限`0.020 ft`をFAIL。
- 最終判定: `FAIL_CLOSE_NO_RESCUE`。

## 所見

### 良かった点

- 0 model・0 boosterで現行ML submitted anchorを直接評価できる。
- current well自身の既知prefixだけを使うため、same-name train wellがないhidden rerunでも式が成立する。
- 変更が単一式で、作用範囲と失敗原因を監査しやすい。
- 境界0--240、全5 folds、hidden-like 2面は一貫して改善し、tail safety gateも通過した。

### 悪かった点

- 1000+の改善は`0.000002 ft`で実質ゼロだった。
- pooled改善は`0.001611 ft`だけで、anchor更新に必要な`0.020 ft`の約8.1%に留まった。
- always-on direct correction単独では未知suffix全体への寄与が小さすぎる。

### リスク / 注意

- 公開Notebookの低Public scoreを、このU補正単独の効果と解釈しない。
- 公開Notebook記載の495/773 well監査は独立再現できていない。
- `cap=8 / tau=240`以外を同じOOFで試すとpost-hoc tuningになるため禁止する。

## 次

direct fixed U-boundary fadeをno-rescueで閉じる。cap/tau/threshold/distance/well gate/blendを同じOOFで調整せず、inferenceとsubmissionへ進めない。continuity再訪は独立したtarget-free feature／selector仮説ができた場合だけ別途検討する。
