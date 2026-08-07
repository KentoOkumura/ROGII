# exp431_affine_ar1_seed_evidence_reaggregation 結果

## 状態

`closed_prerequisite_failed`。未実装、未実行で、CV、LB、生成物、Kaggle kernelはない。

## 仮説

exp427 が示す affine uncertainty + AR(1) covariance の追加識別力を seed evidence に移すと、固定128 seedの PF 集約を改善できる。

## 固定設定

- PF 親: exp404
- prerequisite: exp427 complete PASS
- PF: x1.0、500 particles、128 seeds、trajectory variant 1
- candidate: `affine_ar1`
- controls: `identity_iid_matched`、`affine_iid`、`identity_ar1`、保存 exp404 T=5
- evidence: proper block/run log-density の総和
- aggregation: centered softmax、temperature 5.0

## 結果

| メトリック | 値 |
| --- | --- |
| exp427 technical gate | FAIL |
| exp427 scientific gate | FAIL |
| exp427 eligible block率 | `0.721073584`（必要値`>=0.75`） |
| exp427 `affine_ar1` MRR | `0.386090045` |
| exp427 matched / saved MRR | `0.388002620 / 0.388146378` |
| exp431 PF well-runs | `0` |
| exp431 particle starts | `0` |
| CV | 未実行 |
| Public LB | 未提出 |
| Private LB | 未提出 |

## 再現性

- deterministic anchor: false
- prerequisite kernel:
  `kentookumura/exp427-affine-ar1-whitened-gr-readout-train` version 2
- exp427 scientific contract SHA:
  `75241052d0bdeba3dcbad6548167bb1193f4375b1035e8de625591d4fdb24773`
- exp427 target-free bundle SHA:
  `3cae530e8c2629eea16468383ae06edc3e971d1ed77fb3a4d8d71d4043ba8a4d`
- trajectory / evidence / prediction SHA: 未生成

## 解釈

exp427はtechnical / scientificともFAILした。特にeligibleな5,615 blocks上でも
primary MRRがmatched / saved controlを下回ったため、coverage gateだけを緩和しても
exp431を実装する根拠にはならない。事前登録したno-rescue分岐に従って閉じた。

## 次

なし。exp431を再開せず、PF replay、推論、提出へ進まない。
