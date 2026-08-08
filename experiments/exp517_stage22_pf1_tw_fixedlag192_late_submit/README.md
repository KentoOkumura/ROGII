# exp517 stage 2-2 5 PF + fixed-lag 192 + tabular 再現

6位writeupのstage 2-2で説明された`5-input + fixed-lag smoother + tabular model`を再現した実験。v1の契約不一致を同一実験の失敗履歴として残し、v2で実装を修正した。

## 状態

v2の学習、推論、提出、公式scoringは終了した。CVとPublicは掲載値に近いが、Privateは未再現であり、stage 2-2全体の完全再現とは判定しない。ユーザー判断により実験statusは`completed`。one-shot提出済みのため再調整・再提出は行わない。

## 仮説

公開された5本のPF trajectoryを公開tabular stackへ接続すれば、stage 2-2の評価水準を再現できる。

## 変更内容

契約不一致だった1 PF direct出力をv1履歴として保持し、v2では5 PFのfixed-lag trajectoryを公開tabular stackへ入力した。

## 検証方針

手法契約を固定した1 variantだけをKaggleで学習・推論し、CVをgateに一度だけLATE SUBMITする。LBを見た再調整は行わない。

## 所見

CVとPublicは近似再現したが、Privateを含む完全再現は未達だった。数値と判定根拠は正の記録を参照する。

## 次のアクション

再調整や再提出は行わない。Private差を未解決事項として保持する。

- 数値・最終判定: [metrics.json](metrics.json)、[result.md](result.md)
- 実行履歴: [SESSION_NOTES.md](SESSION_NOTES.md)
- 実装契約: [config.yaml](config.yaml)
