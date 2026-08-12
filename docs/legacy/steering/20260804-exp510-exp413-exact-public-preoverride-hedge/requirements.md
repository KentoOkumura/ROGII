# 要件

## 依頼

最終提出2枠のうちPublic分布へのhedgeを担う第2枠として、LB 7.159公開notebook系の
dual-pipeline出力を、guarded contact overrideとGold visible-prefix overlayの直前でfreezeし、
exp413へ固定10%だけ混合する。公開コードの相補性は残すが、Public-LB固有のwell補正は入れない。

## 制約

- Route: `ensemble`
- 親anchor: `exp413_scale5_likpf_full_replacement_on_exp335`
- public source: `degnonguidi/public-score-rogii-lb-7-159`の保存済みsource。
- public component boundaryを可視コードのfinal blend cellに固定する。

```text
public_preoverride = 0.55 * sp45_projection_submission + 0.45 * submission_B
prediction = 0.90 * exp413 + 0.10 * public_preoverride
```

- 公開sourceで作られる`submission_A.csv`は最終cellに使われていないため、本実験でも代入しない。
- `ENABLE_GOLD_OVERLAY=False`を強制し、guarded contact override cellの前でcomponentをfreezeする。
- Gold、contact override、train/test same-well lookup、public well ID規則、Q0522/A27、固定branch shiftを禁止する。
- weight 0.10を再fit/gridせず、LB、current-test差分、well別診断で変更しない。
- public output CSVのcopyは禁止し、code-competition hidden test上でraw testから再生成する。
- optional artifact欠落時のinference-time再学習を禁止し、必要model/artifactが揃わなければfail-closeする。
- 今回の作業範囲は設計、backlog、実験scaffoldまで。実装、正規notebook採用、Kaggle package/run、提出は各別承認とする。
- 再現性: `docs/06_reproducibility.md`に従い、source/dataset/model/feature/prediction/submission SHAとkernel versionを記録する。

## 受け入れ基準

- archived source SHAとpre-override cell境界をmanifestに固定し、Gold/contact関数の実行が0であることをcontract testで確認する。
- `sp45_projection_submission`と`submission_B`をID one-to-oneで結合し、0.55/0.45式をfloat64で再現する。
- exp413との最終式は全行0.90/0.10固定で、routerや追加後処理がない。
- sample-derived row/ID/order/nonempty-well契約、NaN/Inf/重複/fallbackをfail-closedで検証する。
- stochastic PF/Beamはstable per-well seedを使い、global RNG/thread scheduling依存を禁止する。公開artifactとのbyte parityは主張しない。
- hidden-compatible出力、component差分readout、全SHA manifestが揃った場合だけ「最終提出第2枠候補」と表記する。
- CV/Private安全性は未証明なので、ML/ensemble anchor更新や第1枠との優先順位逆転に使わない。

## 2026-08-04 承認追記

- ユーザーの「exp510を実装してください」により、artifact preflight、候補source/notebook、contract
  tests、実験記録更新までを追加承認した。
- 正規notebook採用、Kaggle package/push/run、output取得、submit-check、外部提出は未承認のまま。

## 2026-08-04 実行承認追記

- ユーザーの「実行してください」により、実装済み候補をKaggle CPU/private/internet-offで
  package/push/runし、完了までlogsを監視する範囲を追加承認した。
- 正規`*_inference.ipynb`の上書き採用は行わず、候補から専用
  `*_current_test_inference.ipynb`を機械生成して実行する。
- Kaggle output archive取得、submit-check、competition submitは承認範囲に含めない。

## 2026-08-04 提出承認追記

- ユーザーの「提出してください」により、Kaggle version 2 output取得、submit-check、
  code competition submission、scoring監視、LB/提出履歴記録までを追加承認した。
- submit-checkがFAILまたはWARNの場合は提出せず停止する。PASS時だけkernel version 2の
  `submission.csv`を提出する。

## 2026-08-04 hidden rerun結果

- code submission ref `55225634`は151分後にempty scoreのhidden rerun例外で終了した。
- hidden-compatible受け入れ基準はFAIL。公開test固定のexp413 prediction sidecarをdynamic hidden
  sampleへ完全一致joinする実装が高確度原因で、version 2は再提出しない。
- 修正にはexp413 hidden-safe inferenceのexp510内動的再生成が必要。修正・再push・再提出は別承認とする。

## 2026-08-04 hidden-safe修正・再実行承認追記

- ユーザーの「まず修正・実行してください。提出はまだです。」により、exp413 dynamic sample再生成、
  static sidecar削除、同一Kaggle kernelへのpush/run、output検証までを承認した。
- version 3の生成物監査でin-memory float32と従来CSV component boundaryの最大`4.36e-4`差を検出し、
  exact契約を維持するため生成CSVの即時readback guardを追加した。
- version 4はvisible current-testでtechnical PASSし、version 2のfinal content/submission SHAと完全一致した。
- この時点ではcompetition submitと正規notebook採用は承認範囲外だった。

## 2026-08-04 version 4提出承認追記

- ユーザーの「提出に進んでください」により、version 4 outputの再submit-check、competition submit、
  scoring監視、提出履歴・LB記録までを承認した。
- FAIL/WARN 0の場合だけkernel version 4を提出する。正規notebook採用は引き続き範囲外。
- ref `55231514`として受理され、初期statusはPENDING。確定結果は下記へ追記する。

## 2026-08-05 version 4 scoring結果

- ref `55231514`は627分後にCOMPLETE、Public LB `7.201`。Kaggle UIは
  `Your latest submission scored 7.201, matching your best.`と表示した。
- exp413 ref `55080377`と公開3桁で同値。APIがfull-precision scoreを返さないため、生RMSEの完全一致は
  未確認とする。
- submitted payloadと最終blend/write経路の再監査はPASS。ただしPublic hedgeの測定可能な改善はなく、
  anchor更新・weight変更・追加提出は行わない。
