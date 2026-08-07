# exp245 結果

## 状態

Kaggle CPU selector train v1は完了した。train-side context parityは合格したが、
selector safety guardはworst-well回帰により不合格。saved-selector inference、direct採用、
submissionには進まない。

## NaN修正

- rows / wells: 3,783,989 / 773
- context: exp238の184列からtrain-only `copcf_*` 41列を除いた143列
- 残存 `copcf_*`: 0列
- exp226診断: 必須4列すべて存在
- missing context: 0列
- nonfinite context: 0件
- saved selector: outer 5 × inner 4 = 20 models、全model SHA一致

したがって143列へ削減した学習側schemaはfiniteになったが、これは元の184特徴を保つ
raw-test parity修正ではない。41 `copcf_*`除外ablationとして解釈する。current testでの
143列再生成も、guardによりinferenceを停止しているため未検証である。

## selector top1直接採用のOOF監査

`selector_safety_readout`は、予測誤差最小と判定した候補を行単位に直接採用した結果である。

| bucket | fallback likPF RMSE | selector top1 RMSE | delta |
| --- | ---: | ---: | ---: |
| global | 11.594898 | 8.558917 | -3.035981 |
| `000_050` | 1.188877 | 0.622346 | -0.566531 |
| `1000_plus` | 12.702991 | 9.393874 | -3.309117 |

平均では大きく改善するが、773 wells中215 wellsで悪化し、184 wellsが+0.25 ftを超えた。
worstは`fb03ae90`で20.515682から58.532380へ悪化し、deltaは+38.016697 ftだった。

hard top1 RMSE 8.558917はexp218 final 8.475794より+0.083123、exp238 add-only final
7.936690より+0.622227悪い。よってadd-onlyを無条件hard top1へ置き換える根拠はない。

## exp238との比較

exp245とexp238のwell別selector delta相関は0.990883で、worst wellも同じだった。
41 train-only特徴の除外によるselector top1 RMSE差は+0.046655である。NaN修正は必要だったが、
worst-well問題の原因はNaN特徴ではなくselector誤選択そのものと判断する。

## 判断

- train-side context parity: 合格
- selector model/fold/SHA contract: 合格
- safety guard: 不合格
- current-test inference: 未実行
- unconditional direct candidate adoption: 不採用
- 次段候補: outer-trainだけで閾値を決めるwell-risk / confidence gate付きdirect採用監査
