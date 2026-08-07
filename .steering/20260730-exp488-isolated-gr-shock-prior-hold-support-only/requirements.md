# 要件

## 依頼

ユーザーの「安全に性能検証するための事前固定した対照群はいらないので次に進む」
という指示に従い、exp482を改変せず、zero-shock well対照群を要求しない
support-only分岐でStage A1を実行する。

## 制約

- Route: `pf_beam`
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- exp482のraw-shock、message agreement、current-emission conflict、
  row-local LOO readout、親exp209 HMMは変更しない。
- manifestはraw-only censusのshock count降順、suffix rows降順、well ID昇順の
  top32 support wellsとし、zero-shock controlは0。
- truth/fold/errorは32 wellsのmessage、trigger、prediction SHA freeze後に読む。
- Stage 0はshock-enriched mechanism readoutであり、CVやpromotion evidenceではない。
- Stage 1、inference、submissionは別承認まで無効。

## 受け入れ基準

- scientific candidate 1、raw census 773 wells、unchanged exp209 message HMM
  replay 32 wellsで完走する。
- candidate state HMM、保存parent再生成、LightGBM/model/booster/PF/Beam/GPUは0。
- support32、parent parity、normalization、finite coverage、trigger support、
  truth-late、runtime/RSS、科学gateをすべて記録する。
- FAIL時はexp488内でthreshold/window/output/control定義を救済しない。
- deterministic anchor として扱う場合は、feature content SHA、model SHA、prediction SHA、submission SHA、Kaggle kernel version が記録されている。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
