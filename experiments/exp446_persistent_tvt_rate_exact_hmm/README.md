# exp446_persistent_tvt_rate_exact_hmm

## 状態

- ルート: `pf_beam`
- 状態: fixed32 Stage 0 `stage0_fail_closed`
- 優先度: 低-中・P3
- CV / Public LB / Private LB: なし
- 作成日: 2026-07-30
- 親実験: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`

## 仮説

exp209の持続rateを`U-rate=d(TVT+Z)/dMD`ではなく
`TVT-rate=dTVT/dMD`として直接持てば、既知の`Z`勾配をrate dynamicsから分離でき、
rateの0方向under-responseとforward transition/prior hysteresisを減らせる。

これはexp445の同値な座標再ラベルではない。TVT-rateにexp209と同じ
zero-centered dynamicsを直接適用するため、予測が変わる科学的treatmentである。

## 変更点

```text
parent:
  state = (TVT, U-rate)
  delta_TVT = U-rate * delta_MD - delta_Z

candidate:
  state = (TVT, TVT-rate)
  delta_TVT = TVT-rate * delta_MD
```

- prefix初期rateは`median(delta_TVT_input/delta_MD)`。
- 41-state gridは`±max(0.10, abs(q0)+0.04)`。
- `momentum=0.998`、`sig_r=0.002`の親local kernelをTVT-rate上で使う。
- TVT grid、position kernel、GR emission、prior、posterior readoutは親固定。

## 検証方針

1. `Z`一定のsynthetic sentinelでexp209との数値パリティを確認する。
2. small-state brute-force、kernel normalization、truth-late、SHAを確認する。
3. Stage 0 fixed32の候補32 HMM runsでmechanism AND gateを判定する。
4. 全PASSと別承認がある場合だけStage 1の773 wellsへ進む。

Stage 0では保存exp209 controlを使い、parent rerunは0。fixed32はCVではない。
rate/grid/noise/emission/gate/blend/selectorのsame-OOF救済は行わない。

## 実行量

| 段階 | candidate HMM | parent rerun | model / booster / PF / Beam / GPU |
| --- | ---: | ---: | ---: |
| Stage 0 | 32 | 0 | 0 |
| Stage 1 | 773 | 0 | 0 |

Stage 0はKaggle private CPU version 1で完了した。Stage 1、rerun、
inference、submissionはfail-closedである。

## 実行入口

- 学習 notebook: `exp446_persistent_tvt_rate_exact_hmm_train.ipynb`
- 推論 notebook: `exp446_persistent_tvt_rate_exact_hmm_inference.ipynb`
- compact self-contained train/inferenceを正規notebookへ採用済み。
- Kaggle version 1は32 wells / 156,088 suffix rowsを完走した。
- Kaggle Notebook実行を正とし、ローカル実行はユーザーが明示した
  `--allow-local` smoke debugだけに限定する。

## 再現性

- RNGなし。well、row、position、TVT-rate、edge、message、reduction順を固定する。
- input、rate grid/kernel、joint transition、posterior、prediction、
  diagnostic、metricsのSHAを保存する。
- gzipはdecompressed content SHAを主証拠にする。
- 初回成功runをdeterministic anchorにしない。

## 所見

- technicalはruntime projectionだけFAILして`17/18 PASS`だった。
- mechanismは`0/7 PASS`。under-response share削減`-0.061091`、
  forward / persistent SSE削減`-0.306441 / -0.214831`だった。
- matched control pooled / p95は`+7.159063 / +16.310622 ft`悪化した。
- known-Z forcingを外す物理的リスクが実データで顕在化したためbranchを閉じる。

## 次

Stage 1、rerun、inference、submissionへ進めない。rate/span/noise/grid/
emission/prior/gate/blend/selectorのsame-fixed32救済も行わない。
