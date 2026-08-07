# Treatment run 1 — top five

Evidence cutoff は 2026-07-12。許可sourceは `rogii_source_hidden_packet_v1.md` のみ。候補集合の truth-aware **coverage** と、hiddenで真値なしに選ぶ **selectability** は全案で別評価にした。

| 順位/slot | Idea | 中心仮説 | cheap proxy → kill |
|---|---|---|---|
| 1 / safe | I03: safety-constrained residual stacking | ML anchorを固定fallbackにし、候補disagreementが長期誤差を示す時だけ縮小補正する | 既存OOFだけで4-fold fit/1-fold評価。合算RMSEが9.5264未満でなければkill |
| 2 / exploration | I01: union lattice decoder | whole-path top-1ではなく、既存＋heatmap候補をrow nodeにした連続性付きjoint decodeで誤選択区間を短くする | 1 outer foldを完全holdout。anchor非改善、追加heatmapで悪化、hidden runtime 9h以上ならkill |
| 3 / orthogonal | I02: candidate-conditioned heatmap compatibility | direct pathとして弱いgenuine GR signalを候補rerankingの観測へrole-changeする | 学習済みlogit集約だけでpairwise ranking。ranking lossとoracle regretが共に改善しなければkill |
| 4 / compute_enabler | I11: one-pass cache / streaming scorer | candidate共通計算を一度にまとめ、nested検証とhidden実行をwall内にする | 50 wellsでparity・wall・RSS測定。差1e-6超、speedup 2倍未満、projected 7.2h超ならkill |
| 5 / exploration | I07: complementary candidate generator | top heatmap peakでなく、既存5候補へのmarginal coverageと多様性を目的化する | 既存logitのcross-fit再選択。union oracle/worst-well coverageが改善しなければkill |

主なrejectは、heatmap probability-weighted point pathの再試行、truth-aware oracle選択、prefix-GR likelihoodのraw採用、seed交絡を残したlag拡大、public 3-well固有分岐、train-only formation列利用。いずれも既存の閉じた実装、leakage、runtime、またはhidden availability gateに抵触する。

未解決入力は、候補artifactの正確なschemaとhidden再生成可能性、heatmap logitとcandidate pathのrow/depth対応、current MLに含まれるuncertainty feature、runtime profile・peak memory、hiddenの正確な行数/長さ分布。これらは仮定としてJSONに明記し、満たせない案はconfidenceを下げた。
