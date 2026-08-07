# 設計

## アプローチ

1. exp072 v2 と同じ `stable_sha256_per_well` policy で raw current test から
   196 列の Pixiux/public replay feature frame を再生成する。
2. exp228 の target-free generator を再利用し、exp333 の固定 config で
   anchor、U projection / disagreement、GRWR の 129 row feature を生成する。
3. exp226 inference v1 の predictionを ID で結合し、未知suffixを同一の
   `numpy.linspace + searchsorted(side="left")` K16 segmentへ割り当てる。
4. train と同じ有限値 float64 mean と structural定義で、3 well × 16 =
   48 segment の136列を作る。
5. exp333 Stage 1 train v1 の outer fold 0..4 modelをSHA検証後に読み、
   各segment offsetを予測して5 model等重み平均を行い、rowへ定数broadcastする。
6. `exp333_candidate_tvt = exp226_tvt + mean_segment_offset` を候補artifactとして
   保存する。候補の選択、他候補とのblend、submission生成は後段へ持ち越す。

## 実験範囲

- 対象実験: `exp333_exp226_k16_segment_residual_offset_target`
- Route: `ensemble`
- 親実験: `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction`
- downstream根拠: `exp361_exp333_candidate_path_addone_novelty_audit`
- 変更する変数: dataset split を saved train OOF から current test へ変更し、
  5 fold componentを等重みensembleする。
- 固定する変数: K16、129 row feature、136 model feature、5 model、
  exp226 v1 base、aggregation、offset適用、CPU、no submission。

## 再現性設計

- seed policy: exp072 v2 の SHA256 well/split seedをそのまま使う。
- stochastic 処理の有無: raw-test replay 内の PF / likelihood-PF は stochastic
  だが、各well固有seedで固定する。LightGBMは推論のみ。
- PF/Beam / likelihood-PF / seed bagging の有無: PF ANCC、PF Z、
  likelihood-PF、deterministic Beamのcurrent-test再生成を行う。PF seeds=128、
  particles=500。
- 並列処理と乱数の関係: global seed順序に依存せず、well単位のstable seedを使う。
  n_jobs=8に固定し、ID / well / row / segmentをmergesortで正規化する。
- CPU/GPU runtime と deterministic flags: Kaggle CPU、GPU=false、internet=false、
  LightGBMは保存済みboosterのpredictのみ。
- train cache / test feature regeneration の SHA 記録方針: raw-test replay metadata、
  row schema/content SHA、segment content SHA、gzipはdecompressed content SHAを保存する。
- model manifest / prediction / submission SHA 記録方針: model manifest、
  feature schema、5 modelを固定SHAで照合し、fold component / ensemble / candidate
  content SHAを保存する。submission artifact / SHAは作らない。
- Kaggle package bootstrap 確認方針: exp072 public replay sourceとexp228 target-free
  generatorをpackageへ固定コピーし、exp333 Stage 1 train v1とexp226 inference v1を
  kernel sourceとしてmountする。prepare後のconfig/source一致とmetadata sourceを確認する。

## リスク

- リークリスク: current testのTVT truth / formation label / selector scoreを読まない。
  raw testはMD/X/Y/Z/GR/TVT_input prefixのみをsource generatorが利用する。
- CV/LB 不一致リスク: exp333はdirect promotionに失敗済みであり、今回の出力は
  exp361で支持された候補パスに限定する。単独予測や提出候補とは解釈しない。
- ランタイム/メモリリスク: raw-testは14,151行、3 wellのみだが、773 train wellから
  imputerを構築する。CPU 8 thread、runtime上限30,600秒で実行する。
- 再現性リスク: sourceやモデルの取り違えをSHA、feature順、saved train summary parity、
  exp226 submission SHAでfail-closedにする。
