# exp122_exact_override_negative_control 結果

## 仮説

Pilkwang replay の exact-match recovery / guarded overlap override は、hidden-safe な改善根拠として採用しない。exp122 は、その判断を summary JSON と guard output inventory で機械的に残すための negative-control audit。

## 設定

- 親: `exp079_public_artifact_replay_integrity_audit`、`exp064_train_test_well_id_assert_probe`
- 検証: notebook / exp079 / exp064 / guard output の file audit
- メトリック: `diagnostic_status`
- シード: 42

## 結果

| メトリック | 値 |
| --- | --- |
| CV | 対象外 |
| Public LB | 対象外 |
| Private LB | 対象外 |
| Diagnostic status | `negative_control_passed_current_evidence` |

## 再現性

- deterministic anchor: いいえ
- seed policy: `none_deterministic_file_audit`
- kernel version: `kentookumura/exp122-exact-override-negative-control-train` v1
- feature content SHA: feature は生成しない
- model SHA / manifest SHA: model は生成しない
- prediction SHA: before/after submission が見つかった場合だけ記録
- submission SHA: submission は生成しない
- rerun result: Kaggle train v1 で `negative_control_passed_current_evidence`

## 解釈

Kaggle train v1 では Pilkwang final が archived base branch と一致し、exp064 hidden code submission も exposed filename-prefix の train/test well_id overlap assertion を発火しなかった。したがって current evidence では negative control passed とし、same-well exact / guarded override は改善根拠から除外する。

ただし、archived notebook source には same-well shortcut flags が enabled として見える一方、exp079 source spec は exact/override disabled check を期待している。この矛盾は hidden-safe 改善根拠ではなく risk として記録する。同じ物理 well が別 anonymized id で出る可能性は、この監査でも否定しない。

## 次

完了。追加提出や inference run は不要。
