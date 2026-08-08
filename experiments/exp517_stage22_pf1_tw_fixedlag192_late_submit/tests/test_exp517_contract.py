from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml


HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
SOURCE = HERE / "exp517_stage22_pf1_tw_fixedlag192_late_submit_compact_selfcontained_inference.py"
PUBLIC_CONFIG = HERE / "pf_banks_config_v96_public.json"
CONFIG = HERE / "config.yaml"
METRICS = HERE / "metrics.json"
GENERATOR = ROOT / "scripts" / "prepare_exp517_stage22_pf1_tw_fixedlag.py"

PUBLIC_CONFIG_SHA256 = "80e973d5f5e0e39be758a03f399cdd3d81d9e79320da8db6fbddbc25c2a202f3"


def read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def top_level_literals(source: str) -> dict[str, object]:
    values: dict[str, object] = {}
    for node in ast.parse(source).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            values[target.id] = ast.literal_eval(node.value)
        except (TypeError, ValueError):
            pass
    return values


def test_public_final_pf1_config_is_sha_pinned() -> None:
    raw = PUBLIC_CONFIG.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == PUBLIC_CONFIG_SHA256
    public = json.loads(raw)
    pf1 = public["params"]["pf_1"]

    assert public["bank_order"][0] == "pf_1"
    assert public["n_seed"] == 32
    assert pf1["n_particles"] == 600
    assert pf1["use_anchor"] is False
    assert pf1["use_phys"] is False
    assert pf1["smooth_lag"] == 192


def test_raw_config_and_normalized_embedded_text_have_separate_hashes() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    literals = top_level_literals(source)
    text_sha = hashlib.sha256(PUBLIC_CONFIG.read_text(encoding="utf-8").encode()).hexdigest()

    assert literals["PUBLIC_CONFIG_SHA256"] == PUBLIC_CONFIG_SHA256
    assert literals["PUBLIC_CONFIG_TEXT_SHA256"] == text_sha
    assert PUBLIC_CONFIG_SHA256 != text_sha
    assert (
        'hashlib.sha256(PUBLIC_CONFIG_JSON.encode("utf-8")).hexdigest() '
        '!= PUBLIC_CONFIG_TEXT_SHA256'
    ) in source


def test_v1_proxy_contract_is_preserved_as_failed_history() -> None:
    config = read_yaml(CONFIG)
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    v1 = metrics["history"]["v1"]

    assert config["experiment"]["route"] == "ensemble"
    assert config["method_contract"]["fidelity"] == "historical_contract_reconstruction"
    assert v1["status"] == "contract_mismatch_failed"
    assert v1["route"] == "pf_beam"
    assert v1["contract"] == "pf_1_twgr_fixedlag192_direct_mean"
    assert v1["technical_generation"] == "pass"
    assert v1["stage22_method_contract"] == "fail"
    assert v1["source_sha256"] == hashlib.sha256(SOURCE.read_bytes()).hexdigest()


def test_generated_orchestration_runs_only_pf1_tw_fixedlag192() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    literals = top_level_literals(source)
    assert literals["FIDELITY"] == "proxy"
    assert literals["PF_BANK"] == "pf_1"
    assert literals["PF_REPRESENTATION"] == "tw"
    assert literals["PF_GENERATION_SEED"] == 4423098
    assert literals["PF_N_PARTICLES"] == 600
    assert literals["PF_N_SEEDS"] == 32
    assert literals["PF_SMOOTH_MODE"] == "fixedlag"
    assert literals["PF_SMOOTH_LAG"] == 192
    assert literals["PF_PHYSICS_ENABLED"] is False
    assert literals["PF_EMISSION_WEIGHT"] == 0.0

    run_section = source.split(
        "# ## 4. Run pf_1 × twGR × fixed-lag 192 only", maxsplit=1
    )[1].split("# ## 5. Sample-ID alignment", maxsplit=1)[0]
    assert run_section.count("P = bank_param(PF_BANK)") == 1
    assert run_section.count("outs = run_smoother_ext(") == 1
    assert 'P["smooth_mode"] = PF_SMOOTH_MODE' in run_section
    assert 'P["smooth_lag"] = PF_SMOOTH_LAG' in run_section
    assert 'P["_physics"] = PF_PHYSICS_ENABLED' in run_section
    assert 'P["_w_nn"] = PF_EMISSION_WEIGHT' in run_section
    assert "x = attach_anchor(x, wid, physics=False)" in run_section
    assert "build_sims" not in run_section
    assert "for bank in" not in run_section
    assert "for representation in" not in run_section


def test_gr_only_state_and_fixed_lag_decode_are_preserved() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    required_fragments = [
        "pos = ls[:, None] + IS * rn(); rate = ir[:, None] + IR * rn()",
        "v = pos - zi",
        "def _grid(tw_tvt, tw_gr, step=0.2):",
        "L = int(P.get(\"smooth_lag\", 32))",
        "buf = torch.full((B, L, N)",
        "if i >= L:",
        "pts_s[:, i - L] = (w * old).sum(1) - z[:, i - L]",
        "for j in range(max(0, T - L), T):",
        "ww = np.exp(lk / max(ls_scale, 1e-6))",
        "w_nn=PF_EMISSION_WEIGHT",
    ]
    for fragment in required_fragments:
        assert fragment in source
    assert "PUBLIC_ANCHOR_SOURCE" not in source
    assert "EXPECTED_CHECKPOINT_SHA256" not in source
    assert "build_sims(" not in source


def test_runtime_uses_sample_schema_and_has_no_visible_test_branch() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    assert "TARGET_COLUMN = str(sample.columns[1])" in source
    assert 'sample_keys.str.rsplit("_", n=1, expand=True)' in source
    assert "expected_ids = set(sample_keys)" in source
    assert "actual_ids = set(prediction_lookup)" in source
    assert "submission[TARGET_COLUMN]" in source
    assert not re.search(r"if\s+sample_(?:rows|keys)\s*==", source)
    assert not re.search(r"if\s+len\(sample\)\s*==", source)
    assert not re.search(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        source,
        re.IGNORECASE,
    )


def test_late_submit_proxy_label_and_generator_are_reproducible() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    assert "# # LATE SUBMIT — exp517 stage 2-2" in source
    assert '"LATE_SUBMIT": True' in source
    assert (
        '"submission_message": "LATE SUBMIT | exp517 | stage2-2 pf1 x twGR '
        'fixedlag192 proxy | fixed v1"'
    ) in source
    subprocess.run([sys.executable, str(GENERATOR), "--check"], cwd=ROOT, check=True)
