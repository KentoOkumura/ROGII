# 要件

## 依頼

PF の有限粒子 support が失われる区間に対し、元 proposal に加えて `-datum` と
`+datum` の対称枝を一度だけ混ぜる defensive particle reinjection を設計する。
2026-07-28の初回依頼ではdesign-onlyとし、同日のユーザー指示
`exp432を実装してください`でStage 0のcompact self-contained実装とcontract test
まで承認された。2026-07-29のユーザー指示`実行してください`で、正規train
Notebook採用とfixed32 Kaggle Stage 0 package/push/runが承認された。
full、inference、submissionは含まない。

## 検証する問い

exp410 では finite particle support不足が排他的 SSEの36.4701%を占め、roughening 10倍は固定 sentinel で改善したが不均一だった。exp412 の persistent beta-filter rate-gap は方向推定を treatment に使うと失敗したが、誤りが生じやすい時刻の選択性は高かった。方向を一切使わず、その最初の event でのみ対称な ±datum proposal を加え、importance correction で元 target を保てば、global noise 増加より安全に support を回復できるかを検証する。

## 制約

- Route は `pf_beam` とする。HMM は truth-free trigger cache の補助で、最終 prediction は PF だけから生成する。
- trigger は exp412 と同じ unchanged exp209 first-pass HMM の persistent beta-filter rate-gap scheduleを使う。
- exp412 の beta sign/direction は一切使わず、最初の inactive→active event時刻だけを使う。
- treatment は各 well 最大一回。proposal mass は base 0.80、`-datum` 0.10、`+datum` 0.10 に固定する。
- datum は event時点の first-pass filtered HMM position std と 0.35 ft の大きい方。
- importance ratio `p0/q` を必須とし、clipしない。元の PF target、rate transition、emission、resampling、particle数、seed数は変更しない。
- Huber、affine/AR(1)、self-GR、directional branch、global roughening と組み合わせない。
- trigger、proposal、predictionを freeze する前に suffix truth/error/cause を読まない。
- Stage 0 は fixed32 の機構 preflight。CV/promotion evidence として扱わない。
- compact self-contained Stage 0実装とcontract testを作る。
- 正規Notebook採用とKaggle Stage 0実行は承認済み。full実行は別承認とする。

## 受け入れ基準

- trigger、event、datum、三成分 proposal、importance correction、RNG、実行量、gate が文書と config で一致する。
- no-event well は parent と bitwise parity、event は各 well 最大一回。
- Stage 0 の HMM/PF run数と full の HMM/PF run数を分けて明記する。
- exp410/exp412 の negative result を再分類せず、別の単一因子仮説として扱う。
- 実行前は実装済み・未実行と明記し、Stage 0後はgate結果と非CVであることを
  全記録へ反映する。

## 非目標

- beta方向を再利用・救済しない。
- proposal mass、datum floor、trigger threshold/window/persistenceを探索しない。
- original target densityを±datum mixtureへ変更しない。
- Stage 0 fixed32 の結果だけで full CV や提出へ昇格しない。
