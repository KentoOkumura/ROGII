# exp160_sp45_bimodal_selector_confidence_features_on_exp148

## 状態

- Route: `ml_model`
- Status: `planned`
- 親実験: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- 比較基準: exp148 `lgb_mean` CV 8.50128118189582 / Public LB 7.960
- 提出: なし

## 仮説

公開上位 notebook の SP45 / PF / Beam / bimodal selector 系 signal は強いが、ML route へそのまま hard switch すると hidden rerun、leak、Public LB 過適合のリスクが高い。target-free な candidate quality、score margin、bimodal spread、prefix trust、path shape として exp148 に add-only すれば、exp148 anchor を壊さずに外れやすい row / well の判断材料になる可能性がある。

## 検証方針

`sp45_bimodal_selector_confidence_addonly` だけを学習する。control は再学習せず、保存済み exp148 の CV / Public LB を historical baseline として参照する。実行予定は 1 variant、3 LightGBM configs、5 folds、合計 15 boosters。

## 所見

未実行。現時点では実装と静的確認のみ完了している。

## 注意

- selector 出力を direct replacement / late blend / postprocess hard gate として使わない。
- visible-prefix gold overlay、exact contact override、public output CSV copy、oracle best、true-error rank、validation/test true TVT は使わない。
- global OOF が小幅改善しても、near-row、worst-well、hidden-like stress、raw-test/current-test parity が弱ければ submit しない。
