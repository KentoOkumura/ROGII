# exp383_all_tvt_stratigraphic_vector_drift_field 結果

## 状態

Kaggle CPU Stage 0 preflight version 1はcode errorとresource gateの両方でFAIL。
truth join前に停止したためCVはなく、full runへ進めない。

## 仮説

全outer-train正解TVTから得る高密度な`S=TVT+Z` absolute/vector fieldを
fold-safeな6地層面で条件付け、対象prefixで校正すると、exp226のK16 fieldを
1 ft以上改善できる。

## 設定

- 親: exp226
- 検証: exp226 outer 5-fold / well group
- control: CV `9.427109596582213`
- metric: suffix row RMSE
- Stage 0: target-free integrity/support/resource
- Stage 1: direct physical path
- 予定量: 1 candidate / 5 reporting folds / model・HMM・PF・Beam・booster各0

## 結果

| メトリック | 値 |
| --- | --- |
| CV | - |
| Public LB | - |
| Private LB | - |
| Preflight status | ERROR / Stage 0 resource FAIL |
| 失敗前実測 | 22,055.465秒（6.13時間） |
| 5-fold surface stage投影 | 109,866.787秒（30.52時間） |
| runtime gate | 30,600秒（投影は3.5904倍） |

## 再現性

- deterministic anchor: まだ扱わない
- seed policy: no RNG / stable order
- kernel: `kentookumura/exp383-tvt-vector-drift-field-train`
- kernel version: 1（id_no `128459031`）
- feature content SHA: freeze前停止のためなし
- solver manifest SHA: 未実行
- prediction SHA: 未実行
- submission SHA: 対象外
- rerun result: resource gateにより再実行しない

## 解釈

最初の例外は、3 scaleが同じcenter MDを共有するのにscaleをjoin keyへ含めなかった
`pandas.errors.MergeError`である。scale込み一意`query_id`を導入してローカル修正した。
ただしfold 0の209,467 donor windowsへの6地層surface fitだけで約6.13時間を要した。
全fold合計1,043,436 windowsへ比例投影するとsurface stageだけで30.52時間となる。
固定上限8.5時間を大きく超えるため、join修正後のversion 2を再pushしても
科学gateへ到達できない。outer-valid formation/suffix truthは読まれていない。

## 次

exp383はStage 0 resource FAILで閉じる。version 2、full run、Stage 1、inference、
submissionは実行しない。surface計算を根本変更する案は別実験として扱う。
