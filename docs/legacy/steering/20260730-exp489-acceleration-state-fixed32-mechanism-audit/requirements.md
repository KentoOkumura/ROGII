# 要件

## 依頼

exp458 の高速 scaled probability-space engine は、exp444 保存出力との frozen
閾値を超えたものの、最大差は prediction mean 0.000105 ft、std
0.000064 ft、acceleration posterior 0.000009 だった。ユーザーの
「わずかであれば次に進む」という明示判断を、exact parity の PASS への
書き換えではなく近似 engine 使用の waiver として記録し、exp444 で凍結した
fixed32 Stage 0B mechanism audit を実行する。

## 制約

- Route: `pf_beam`
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- exp458 v2 の fixed4 予測・acceleration posterior・diagnostic を SHA 検証して
  再利用し、残り 28 well だけを同一 engine と同一 scientific contract で計算する。
- truth、role、fold、episode、cause、exp209 control は全32 wellの予測と
  diagnosticをfreezeした後にだけ読む。
- exp458のFAIL、凍結gate、scientific parameter、runtime engineを変更しない。
- Stage 0B は mechanism preflight でありCVや昇格根拠ではない。
- Stage 1、inference、submissionは実行しない。

## 受け入れ基準

- 32 well / 156,088 suffix rowを一意に揃え、有限値coverageを1.0にする。
- posterior acceleration nonzero mass、future true rate curvatureとの符号一致、
  persistent/forward-cause episode SSE、persistent well/fold改善数、
  matched exp209 control safetyをexp444の凍結閾値で判定する。
- 全technical/mechanism gateの結果をfail-closedで保存する。
- deterministic anchor として扱う場合は、feature content SHA、model SHA、prediction SHA、submission SHA、Kaggle kernel version が記録されている。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
