# exp510_exp413_exact_public_preoverride_hedge

## 状態

- ルート: `ensemble`
- 状態: hidden-safe version 4 code submission COMPLETE、候補notebook未採用
- 最終提出枠: 第2枠、Public分布hedge
- CV: なし
- Public LB: `7.201`（ref `55231514`）。Kaggle UIは`matching your best`と表示し、
  exp413 ref `55080377`と公開3桁で同値。full-precision scoreはAPI非公開のため完全一致は未確認
- 親実験: `exp413_scale5_likpf_full_replacement_on_exp335`
- public source: `degnonguidi/public-score-rogii-lb-7-159`

## 仮説

公開sourceの高いPublic LBにはdual-pipelineの相補性とPublic固有処理が混在している。
可視pre-override componentだけをexp413へ事前固定10%加えれば、Public分布への別系統hedgeを
持ちながら、境界後のwell固有処理を遮断できる。

## 固定式

```text
public_preoverride = 0.55 * sp45_projection_submission
                   + 0.45 * submission_B
prediction = 0.90 * exp413 + 0.10 * public_preoverride
```

可視sourceのfinal blend cell直後をfreezeし、そこから後の処理は候補sourceへ取り込まない。
weightは再fitせず、全行float64の固定式だけを使う。

## 実装

- Jupytext候補source:
  `exp510_exp413_exact_public_preoverride_hedge_compact_selfcontained_inference.py`
- 候補notebook:
  `exp510_exp413_exact_public_preoverride_hedge_compact_selfcontained_inference.ipynb`
- 正規`*_inference.ipynb`はplaceholderのまま。採用は別承認とする。
- archived sourceからPF/Beam、空間prior、projected-SP45、Pipeline-B feature/postprocessだけを抽出した。
- projected-SP45はsample全IDを埋めることを必須とし、元sourceのPipeline-A tabular fallbackを削除した。
- Pipeline Bは`fleongg/rogii-claude-models-pub` version 1の`features.json`と3 boosterをSHA照合して読む。欠落・複数候補・SHA不一致時は停止する。
- exp413 v4をdynamic competition sample上で再生成し、parent source SHA、生成artifact SHA、ID契約を照合する。
- 生成したexp413 CSVを読み戻してからblendし、従来のexact serialized component boundaryを維持する。
- PF乱数は`SHA256(base_seed, split, family, well, seed_index)`から作り、global RNGとthread順序に依存しない。
- test側の空間priorでも同じwell IDを除外する。
- public output CSVのcopy、artifact欠落時のbooster学習、weight tuning、router、最終postprocessはない。

## artifact preflight

- `fleongg/rogii-claude-models-pub`: version 1、runtime必須
  - `features.json`: `ea9042f8...96308`
  - `lgb0.pkl`: `a6451b3c...01e1`
  - `lgb1.pkl`: `4d61ab16...f547`
  - `lgb2.pkl`: `1ee24121...acf5`
- `phongnguyn23021656/koolbox-offline`: version 1。候補sourceではimportせずlineage記録のみ。
- `ravaghi/wellbore-geology-prediction-artifacts`: version 6。Pipeline-A fallbackを削除したためruntimeでは使わない。
- archived metadataにある残り3 datasetとkernel sourceもruntime非依存として`config.yaml`へ記録した。

## 検証方針

- honest OOFがないためCV improvementを主張しない。
- archived source SHAと可視final cell境界を静的に固定する。
- source/model/input/feature/prediction/submission SHA、stable seed、ID/order/finite/duplicate/fallback、
  float64 formula parityを全ANDで確認する。
- Kaggle run後はoverall、well、MD horizon、start continuityのtruth-free差分だけを記録する。
- rerun SHA一致前はdeterministic anchorと表記しない。

## 実行量

- scientific variant / final blend: `1 / 1`
- new model/config/fold/booster: `0 / 0 / 0 / 0`
- 保存model読込: `78`（exp413 75 + Pipeline B 3）
- inference-time booster training / parent retraining / GPU: `0 / 0 / 0`
- test wellあたりPF seed run: `48 + 3 = 51`
- test wellあたりbeam path run: `21`
- public sampleの3 wellsではPF `153`、beam `63`。hidden totalはpush前に動的well数から記録する。

## 検証

- dedicated contract tests: `14 passed`
- source SHA / source boundary /入力ファイル式を固定した。
- 候補ASTに禁止route、学習route、public notebook output pathがないことを確認した。
- fixed formula、ID順序、duplicate/nonfinite/missing、model SHA cardinality、exp413 gzip/content SHA、stable seedを検証した。
- Jupytext round-trip、`py_compile`、Ruff `F821/F401/F811`、strict experiment validationを実施する。

## 判断

Kaggle version 4は14,151 rows / 3 wellsを385.11秒で完走し、fallback / duplicate / nonfinite 0、
formula parity 0.0を満たした。exp413はdynamic sampleから再生成され、final content / submission SHAは
version 2と完全一致した。ただしtechnical PASSは科学的採用を意味しない。対応するhonest OOFがなく、
exp497のpublic-coreはtail gateをFAILしているため、第2枠の高リスクhedgeという位置付けを維持する。
修正版はref `55231514`として提出・scoring完了し、Public LBは`7.201`だった。exp413単独と
公開3桁で同値であり、Public分布hedgeによる測定可能な改善は確認できなかった。技術実装は再監査で
PASSを維持するが、honest OOFがないため科学的promotionやanchor更新は行わない。

## 所見

version 2のhidden失敗原因だった公開test固定sidecarを削除し、version 4でexp413の動的再生成と
serialized component boundaryを検証した。ref `55231514`は627分後にCOMPLETEし、Kaggle UIの
`Your latest submission scored 7.201, matching your best.`表示とCLI/APIの`publicScore=7.201`を確認した。
APIは表示3桁より細かい値を返さないため、生RMSEの完全一致か同一表示bucketかは判別できない。
