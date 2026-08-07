# exp226 offset root-cause audit

保存済み group-safe exp226 OOF 3,783,989行 / 773 wellsを対象にした、
read-onlyの根本原因監査です。新規学習、推論、提出は行いません。

主要結論:

- global biasが説明するMSEは0.10%だけ。
- K16 segment mean offsetを診断的に除くとRMSEは`9.427110 -> 1.130603`。
- persistent episodeは全行の18.99%だが、SSEの82.01%を占める。
- episode onsetは一行jumpではなく、約`0.02 ft/row`の緩い累積drift。
- donor距離上位quartileは下位quartileよりwell RMSE中央値とepisode率が大きい。
- GRとU projectionはpooledでは改善するため単独の根本原因ではない。
- 公開deterministic v6 coreとexp226 portは9数値核で最大絶対差`0.0`。

根本機構は、最後の既知TVTを一度だけanchorにし、空間donor由来の相対増分を
長いsuffixへ累積し、その後absolute re-anchorを持たないことです。局所的な
signed rate mismatchが後続K16区間へvertical offsetとして継承されます。

詳しい解釈:
`docs/analysis/exp226_offset_root_cause_audit_20260727.md`

再実行:

```bash
.venv/bin/python studies/exp226_offset_root_cause_audit.py
```
