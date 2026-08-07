# Source-hidden idea portfolio: top five

Allowed evidenceは `rogii_source_hidden_packet_v1.md` のみ、cutoffは2026-07-12。実装・学習・提出は行っていない。

| 優先 | 枠 | 案 | 最初の判定 |
|---:|---|---|---|
| 1 | safe / information | I02 Coverage-aware same-well calibration reference | E06のgainをprefix coverage別に再集計し、cross-fitted reliabilityがharmful wellsを識別できなければ終了。raw likelihood追加やhard gateは再試行しない。 |
| 2 | orthogonal / fusion | I04 Abstaining soft candidate posterior | 既存OOFだけでnested soft fusionを行い、anchor比0.05未満、fold符号反転、worst-20悪化なら終了。oracle coverageとtruth-free selectabilityを別表にする。 |
| 3 | orthogonal / data generation | I03 OOF-shaped corrupted-candidate refiner | OOF candidate誤差の振幅・自己相関・bias・missing・rank inversionを再現できるか先に検査。corruption mismatchまたはcopy-throughなら終了。 |
| 4 | exploration / representation | I01 Whole-well increment posterior | candidate/current pointを使わずsuffix全体をjoint decode。1 held foldでtailまたはworst-20が悪化すればfull OOFへ進めない。 |
| 5 | compute enabler | I05 Paired-parity batched state-space engine | 旧engineとの数値parity、同一seed、200-well 7.5時間以内を要求し、解禁するI02がend-to-end OOFで改善しなければ採用しない。 |

主なrejectは、truth-aware oracle selection、heatmap/learned pathのdirect point化、E06のweight tuningだけの再試行、seed非paired smoothing拡大、公開三well固有分岐、train-only formation列の利用。未解決入力はhiddenでのheatmap再生成runtime、candidate availability、prefix/reference coverage分布、真に成立するTVT-coordinate invariantであり、各案はavailability maskとglobal ML fallbackを前提にする。
