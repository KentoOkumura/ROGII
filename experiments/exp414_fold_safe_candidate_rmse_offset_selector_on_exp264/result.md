# exp414_fold_safe_candidate_rmse_offset_selector_on_exp264 結果

## 状態

implementation-only完了。Kaggle Stage Bは未実行であり、RMSE offset treatmentの
科学結果はまだない。

## 根本原因の事前解析

| selector | hard OOF RMSE | 親以下fold |
| --- | ---: | ---: |
| corrected exp264 parent | 8.587004 | 5/5 |
| exp407 inverse-RMSE weight | 8.668141 | 1/5 |
| candidate×fold平均shiftだけ | 8.580477 | 4/5 |
| 平均shiftを除いたrow-local変化だけ | 8.673599 | 1/5 |

候補別の定数shiftだけでは悪化せず、row-local変化だけで悪化を再現した。
またexp407のweightが低い候補ほどscore差std、score MAE、binary loglossの悪化が
大きかった。親margin 0.5--2.0の比較的確信した選択の反転がnet damageの約74%を
占めた。

したがって根本原因は、候補別RMSEをcandidate-constant sample weightとして
共有木へ与えたことで、低重み候補の局所的に有用な行までgradient / splitへの
寄与が落ち、分散したrow-local score surface driftを起こしたことである。

## 固定treatment

- 親: corrected exp264 Stage B v5
- 候補 / feature / fold / sample / LightGBM params: 親と同一
- 変更:
  fold-safe candidate RMSEを`pred_abs_error`のadditive base offsetとして使う
- 学習:
  unweighted residual L1 regressor 1条件×5 folds
- classifier / control / GPU / inference / submission: 0

## 結果

| メトリック | 値 |
| --- | --- |
| RMSE offset Stage B CV | 未実行 |
| Public / Private LB | 対象外 |

## 再現性

- deterministic anchor: false（Kaggle未実行）
- seed policy: 親seed 42とdeterministic sample key
- parent OOF SHA:
  `9a91b62599278d4e56d57074df4725d4a09391460458b8eccc02dd50af34d48a`
- exp407 OOF SHA:
  `d993b806d92c2462c1509f110669b272b27d48806c0280a2cf54e87c7f32f1e8`
- feature schema logical SHA:
  `aaef4ffdd90667893b099b76a52f1957b22197aea9cee5e5b57bc81048ddd3a4`
- model / treatment OOF / gate SHA: 実行後
- submission SHA: 対象外

## 解釈

原因は「RMSEの値が間違っていた」ことではなく、「RMSEを学習重要度に変換した
使い方」にある。新手法はRMSEを候補の初期期待誤差として利用しつつ、各行を同じ
重さで学習する。実行後に全科学gateを通った場合だけ、この使い方を確立とする。

## 次

canonical Notebook採用とKaggle private CPU 5-booster実行を別承認後に行う。
