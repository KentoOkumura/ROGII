# 全 horizontal well 軌跡のインタラクティブ 3D EDA

`all_horizontal_well_paths_3d.html` をブラウザで開くと、全 776 本（train 773 本、test 3 本）の座標軌跡を回転・ズームできます。

- 通常モードでは train は青、test は橙です。
- TVT モードでは各経路を TVT（無い箇所は `TVT_input`）で連続的に色分けし、両方無い区間は灰色です。
- 地層境界面モードでは train horizontal well の `ANCC`、`ASTNU`、`ASTNL`、`EGFDU`、`EGFDL`、`BUDA` を色分けした半透明 3D メッシュとして重ねます。well を選択した場合は全体メッシュを隠し、選択 well 上の対応する地層境界線だけを表示します。test にはこれらの列がないため、test well を選択した場合は地層境界線を表示できません。
- 左ドラッグで回転、右/中ドラッグまたは Shift+左ドラッグで画面を横・縦に移動できます。ホイールまたはピンチで最大 100 倍までズームできます。
- 軌跡にマウスを重ねると、最も近い well の split と well ID を表示します。
- well リストは複数選択でき、選択した well だけを表示します。未選択なら全 well を表示し、「well 選択解除（全表示）」で戻せます。
- typewell CSV には X/Y/Z 座標がないため、描画対象は対応する horizontal well のみです。
- 各経路は端点を保った一様間引き（最大 250 点）で描画しています。元の全点数は `path_summary.json` にあります。

## 実行結果

- 入力行数: 5,111,476
- 描画頂点数: 194,000
- X span: 183680.81 m
- Y span: 134682.03 m
- Z span: 3853.43 m
- TVT 色範囲（全描画頂点の 2–98 percentile）: 10357.97 – 12770.67
- TVT 色に使える頂点数: 193,452
- 地層境界メッシュ: ANCC: 905 quads, ASTNU: 907 quads, ASTNL: 907 quads, EGFDU: 907 quads, EGFDL: 907 quads, BUDA: 907 quads

生成: `studies/plot_all_well_paths_3d.py`
