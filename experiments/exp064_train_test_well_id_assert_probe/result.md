# exp064_train_test_well_id_assert_probe 結果

## 仮説

hidden / private test に、train と同じ horizontal-well ファイル名 prefix として公開される exact `well_id` が含まれている場合、Kaggle code submission の hidden rerun で train/test overlap assertion が失敗する。含まれていない場合は assertion が通り、placeholder submission が生成される。同じ物理 well が別 filename / anonymized id で公開されるケースまでは、この probe では検出しない。

## 設定

- 親: `KAGGLE_DIRECTION.md` の `train_test_well_id_assert_probe`
- 検証: public sample の既知 overlap は許可し、hidden / private test でのみ no-overlap assertion を適用する。
- メトリック: なし。run status の yes/no 診断。
- シード: 42

## 結果

| メトリック | 値 |
| --- | --- |
| CV | - |
| Public LB | - |
| Private LB | - |
| Kaggle inference run | `kentookumura/exp064-train-test-well-id-assert-probe-inference` v1 completed on public sample |
| Probe phase | `public_sample` |
| Probe status | `public_sample_overlap_allowed` |
| Public sample overlap wells | `000d7d20`, `00bbac68`, `00e12e8b` |
| Submit-check | PASS |
| Code submission ref | `53627058` |
| Code submission status | complete |
| Public LB | 11551.955 |
| Scoring test assertion | not triggered |

## 解釈

Kaggle inference kernel v1 は public sample で成功した。public sample は既知の train overlap 3 wells であり、assert は設計通り public sample 例外として通った。続く code submission ref `53627058` は complete し、`HIDDEN_TRAIN_TEST_WELL_ID_OVERLAP_DETECTED` assertion は発火しなかった。したがって、この scoring test では「公開された horizontal-well ファイル名 prefix としての exact `well_id` overlap」は検出されなかったと解釈する。placeholder zero submission のため Public LB 11551.955 はモデル性能として扱わない。hidden test 内部の overlap 件数、対象行数、id hash、予測差分は取得不能なので記録対象にしない。なお、Kaggle が hidden test の物理 well を別 filename / anonymized id で公開している場合は、この probe では検出できない。

## 次

train/test same exposed-`well_id` 前提の static replay / visible override は優先度を下げる。ただし同じ物理 well が別 id で出る可能性までは否定しない。見えない新規 well 用の hidden branch、public replay integrity audit、PF confidence residual clip を優先する。
