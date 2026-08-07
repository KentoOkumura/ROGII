# 要件

## 依頼

exp355の保存済みStage 1 direct exact-HMM OOFを、corrected exp264のfixed12
candidate bankへ13本目として追加し、同じdual-objective selectorを再学習する。
ユーザー判断によりadd-one novelty監査は省略し、既存`exact_hmm`の置換や既存formulaの
再計算は行わない。

## 制約

- Route: `ensemble`
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- fixed12の候補順序、既存候補値、fixed fallback 7本、selector fold、2目的、
  sampling、LightGBM設定は変更しない。
- 追加候補はexp355 Stage 1のdirect HMM予測1本だけとし、既存`exact_hmm`と
  `exp226_k16__exact_hmm`、`likpf_mean__exact_hmm`、`exp226_w500_50_50`
  は保存値を維持する。
- exp355の保存元outer foldはOOF provenanceとして保持し、selector featureには使わない。
  `well_id,row_idx`でglobal key join後、親と同じexp263 selector foldへrepartitionする。
- selector以外の親/control再学習、GPU学習、downstream TVT、inference、submissionは行わない。
- 実行量は1 variant、2 objectives、outer 5 × inner 4、合計40 CPU selector boosters。

## 受け入れ基準

- 3,783,989行 / 773 wells / 13候補のkey、finite、fold、SHA契約が成立する。
- feature freeze前にexp355同居truth/error列を読み込まない。
- outer-valid wellsをinner fit/early stoppingから除外し、40 modelと25 compact partitionsを生成する。
- 2目的のselector scoreがouter-train priorを改善する。
- exp355候補のprimary top1利用率とfold別利用率を記録する。
- saved parent fixed12 selectorとのpooled RMSEを主判定とする。fold、near、1000+、
  hidden-like、by-well p95、worst-wellは安全性readoutとして記録するが、
  ユーザー指定により今回のselector学習開始前の監査条件にはしない。
- downstream TVT、inference、submissionへ自動移行しない。
- deterministic anchor として扱う場合は、feature content SHA、model SHA、prediction SHA、submission SHA、Kaggle kernel version が記録されている。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
