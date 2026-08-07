from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

SCHEMA_VERSION = "1.0.0"
ANCHOR_CANDIDATE_ID = "last_anchor"
ANCHOR_RMSE = 15.909866082
N_FOLDS = 5
ROLLING_WINDOWS = (32, 128, 512)

RAWTEST_READY_STATUSES = {
    "rawtest_inference_exists",
    "rawtest_regeneration_exists",
    "available_in_existing_rawtest_cache",
}
FORBIDDEN_CANDIDATE_IDS = {
    "hmm_lgb_exp148",
    "hmm_exp218_residual_scale",
    "hmm_exp218_shrink_a050",
}

COMMON_CONFIDENCE_SLOTS = (
    "sigma_tvt",
    "loglik_per_row",
    "entropy",
    "score_margin",
    "support_count",
    "ess_fraction",
    "fallback_rate",
)

# Stage 1 exports these source-native fields beside the six regenerated primitive
# values.  ``confidence_valid`` is exported for every primitive, including
# ``likpf_mean`` where no source-native scalar exists and the value is therefore
# deterministically false.  Formula candidates inherit their parent fields in
# the downstream selector; cross-family confidence is never averaged here.
STAGE1_NATIVE_CONFIDENCE_FIELDS: dict[str, tuple[str, ...]] = {
    "exp226_k16": ("geometry_gr_delta",),
    "selfgr_hmm_a070": (
        "sigma_tvt",
        "source_loglik",
        "loglik_per_row",
        "candidate_finite_source",
        "selfgr_quality",
        "selfgr_peak_tvt",
        "score_margin",
        "selfgr_typewell_agreement",
        "selfgr_valid",
    ),
    "likpf_mean": (),
    "exact_hmm": ("sigma_tvt", "source_loglik", "loglik_per_row"),
    "pf_ancc": ("sigma_tvt",),
    "beam_mean": ("beam_family_std",),
}


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    candidate_name: str
    family: str
    source_exp: str
    rawtest_status: str
    cache_role: str
    global_rmse: float
    formula: str
    source_key: str | None = None
    source_artifact: str | None = None
    value_column: str | None = None
    transform: str = "direct"
    well_column: str = "well"
    row_idx_column: str | None = None
    id_column: str | None = "id"
    anchor_column: str | None = None
    confidence_columns: tuple[tuple[str, str], ...] = ()
    confidence_expected: tuple[str, ...] = ()
    selection_reason: str = ""

    @property
    def is_core(self) -> bool:
        return self.cache_role == "core"

    @property
    def is_rawtest_ready(self) -> bool:
        return self.rawtest_status in RAWTEST_READY_STATUSES

    def as_catalog_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["is_core"] = self.is_core
        row["is_rawtest_ready"] = self.is_rawtest_ready
        row["confidence_columns"] = dict(self.confidence_columns)
        available = set(dict(self.confidence_columns))
        derived = {"loglik_per_row"} if "source_loglik" in available else set()
        row["confidence_derived"] = sorted(derived)
        row["confidence_available"] = sorted(available | derived)
        row["confidence_unavailable"] = sorted(
            set(self.confidence_expected) - available - derived
        )
        return row


@dataclass(frozen=True)
class PairSpec:
    pair_id: str
    left: str
    right: str
    tier: str
    fixed_50_rmse: float
    crossfit_rmse: float
    residual_cosine: float
    crossfit_folds_beating_better_parent: int
    crossfit_fold_weights: tuple[tuple[float, float], ...]
    selection_reason: str

    def as_manifest_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["weights"] = {self.left: 0.5, self.right: 0.5}
        row["components"] = [self.left, self.right]
        row["deployability"] = self.tier
        row["formula"] = f"0.5*{self.left} + 0.5*{self.right}"
        row["crossfit_fold_weights"] = [
            {self.left: left, self.right: right}
            for left, right in self.crossfit_fold_weights
        ]
        return row


def _c(
    candidate_id: str,
    source_exp: str,
    family: str,
    rawtest_status: str,
    role: str,
    rmse: float,
    **kwargs: Any,
) -> CandidateSpec:
    return CandidateSpec(
        candidate_id=candidate_id,
        candidate_name=candidate_id,
        family=family,
        source_exp=source_exp,
        rawtest_status=rawtest_status,
        cache_role=role,
        global_rmse=rmse,
        formula=candidate_id,
        **kwargs,
    )


REFERENCE_CANDIDATES: tuple[CandidateSpec, ...] = (
    _c(
        "exp226_k16",
        "exp226",
        "geometry",
        "rawtest_inference_exists",
        "core",
        9.42710967407494,
        source_key="exp226_oof",
        source_artifact="exp226 train OOF predictions",
        value_column="tvt_pred",
        well_column="well_id",
        row_idx_column="row_idx",
        id_column=None,
        confidence_columns=(("geometry_gr_delta", "gr_delta"),),
        confidence_expected=(
            "donor_count",
            "neighbor_distance",
            "weighted_std",
            "u_projection_gap",
            "geometry_gr_delta",
            "condition",
            "fallback_rate",
            "coverage_valid",
        ),
        selection_reason="scope-best primitive and raw-test inference exists",
    ),
    _c(
        "selfgr_hmm_a070",
        "exp223",
        "hmm_selfgr",
        "rawtest_regeneration_exists",
        "core",
        11.349942882774709,
        source_key="exp223_oof",
        source_artifact="exp223 self-GR HMM train features",
        value_column="hmm_selfgr_boost_only_a070_c100_mean_tvt",
        confidence_columns=(
            ("sigma_tvt", "hmm_selfgr_boost_only_a070_c100_std"),
            ("source_loglik", "hmm_selfgr_boost_only_a070_c100_loglik"),
            ("candidate_finite_source", "hmm_selfgr_boost_only_a070_c100_finite"),
            ("selfgr_quality", "self_gr_quality"),
            ("selfgr_peak_tvt", "self_gr_peak_tvt"),
            ("score_margin", "self_gr_peak_gap"),
            ("selfgr_typewell_agreement", "self_gr_typewell_agreement"),
            ("selfgr_valid", "self_gr_valid"),
        ),
        confidence_expected=(
            *COMMON_CONFIDENCE_SLOTS,
            "posterior_entropy",
            "top1_top2_mass_gap",
            "grid_edge_mass",
            "prefix_sigma",
            "initial_rate_spread",
            "selfgr_quality",
            "selfgr_peak_tvt",
            "selfgr_typewell_agreement",
            "selfgr_valid",
        ),
        selection_reason="best self-GR HMM and best fixed pair with exp226",
    ),
    _c(
        "exp192_likpf",
        "exp192",
        "pf_hard_window",
        "train_cache_only",
        "core",
        11.544811567971532,
        source_key="exp192_oof",
        source_artifact="exp192 hard-window replay train features",
        value_column="likpf_mean_d",
        transform="anchor_plus",
        anchor_column="last_known_tvt",
        confidence_expected=(
            *COMMON_CONFIDENCE_SLOTS,
            "particle_tvt_std",
            "resampling_rate",
            "collapse_rate",
            "seed_prediction_weighted_std",
            "seed_weight_entropy",
            "seed_weight_max",
            "effective_seed_count",
            "multi_observation_score",
        ),
        selection_reason="best hard-window PF representative",
    ),
    _c(
        "selfgr_hmm_a150",
        "exp223",
        "hmm_selfgr",
        "diagnostic_only",
        "core",
        11.559584,
        source_key="exp223_oof",
        source_artifact="exp223 self-GR HMM train features",
        value_column="hmm_selfgr_boost_only_a150_c100_mean_tvt",
        confidence_columns=(
            ("sigma_tvt", "hmm_selfgr_boost_only_a150_c100_std"),
            ("source_loglik", "hmm_selfgr_boost_only_a150_c100_loglik"),
            ("candidate_finite_source", "hmm_selfgr_boost_only_a150_c100_finite"),
            ("selfgr_quality", "self_gr_quality"),
            ("selfgr_peak_tvt", "self_gr_peak_tvt"),
            ("score_margin", "self_gr_peak_gap"),
            ("selfgr_typewell_agreement", "self_gr_typewell_agreement"),
            ("selfgr_valid", "self_gr_valid"),
        ),
        confidence_expected=(
            *COMMON_CONFIDENCE_SLOTS,
            "posterior_entropy",
            "top1_top2_mass_gap",
            "grid_edge_mass",
            "prefix_sigma",
            "initial_rate_spread",
            "selfgr_quality",
            "selfgr_peak_tvt",
            "selfgr_typewell_agreement",
            "selfgr_valid",
        ),
        selection_reason="self-GR family diagnostic parameter retained in core",
    ),
    _c(
        "hmm_peer_atlas",
        "exp231",
        "hmm_atlas",
        "rejected_no_rawtest",
        "core",
        11.569941914757122,
        source_key="exp231_oof",
        source_artifact="exp231 peer-atlas HMM train features",
        value_column="hmm_peer_atlas_a025_mean_tvt",
        confidence_columns=(
            ("sigma_tvt", "hmm_peer_atlas_a025_std"),
            ("source_loglik", "hmm_peer_atlas_a025_loglik"),
            ("candidate_finite_source", "hmm_peer_atlas_a025_finite"),
            ("prefix_sigma", "hmm_prefix_sigma"),
            ("support_count", "peer_atlas_support"),
            ("atlas_match_confidence", "peer_atlas_match_confidence"),
            ("atlas_novelty", "peer_atlas_novelty"),
            ("atlas_uniqueness", "peer_atlas_uniqueness"),
            ("atlas_base_uncertainty", "peer_atlas_base_uncertainty"),
            ("atlas_innovation", "peer_atlas_innovation"),
            ("atlas_change_point", "peer_atlas_change_point"),
            ("atlas_confidence", "peer_atlas_confidence"),
            ("coverage_valid", "peer_atlas_available"),
        ),
        confidence_expected=(
            *COMMON_CONFIDENCE_SLOTS,
            "posterior_entropy",
            "top1_top2_mass_gap",
            "grid_edge_mass",
            "prefix_sigma",
            "initial_rate_spread",
            "atlas_match_confidence",
            "atlas_novelty",
            "atlas_uniqueness",
            "atlas_base_uncertainty",
            "atlas_innovation",
            "atlas_change_point",
            "atlas_confidence",
            "coverage_valid",
        ),
        selection_reason="atlas family representative with strong exp226 complementarity",
    ),
    _c(
        "likpf_mean",
        "exp072",
        "pf",
        "available_in_existing_rawtest_cache",
        "core",
        11.594897672217703,
        source_key="exp072_oof",
        source_artifact="exp072 canonical full replay train features",
        value_column="likpf_mean_d",
        transform="anchor_plus",
        anchor_column="last_known_tvt",
        confidence_expected=(
            *COMMON_CONFIDENCE_SLOTS,
            "particle_tvt_std",
            "resampling_rate",
            "collapse_rate",
            "seed_prediction_weighted_std",
            "seed_weight_entropy",
            "seed_weight_max",
            "effective_seed_count",
            "multi_observation_score",
        ),
        selection_reason="raw-test-ready PF reference",
    ),
    _c(
        "exact_hmm",
        "exp209",
        "hmm",
        "rawtest_regeneration_exists",
        "core",
        11.938287416925935,
        source_key="exp209_oof",
        source_artifact="exp209 enriched exact-HMM train features",
        value_column="hmm_mean_tvt",
        confidence_columns=(
            ("sigma_tvt", "hmm_std"),
            ("source_loglik", "hmm_loglik"),
        ),
        confidence_expected=(
            *COMMON_CONFIDENCE_SLOTS,
            "posterior_entropy",
            "top1_top2_mass_gap",
            "grid_edge_mass",
            "prefix_sigma",
            "initial_rate_spread",
        ),
        selection_reason="raw-test-ready exact HMM and w500 primitive",
    ),
    _c(
        "pf_medoid_k8_m0",
        "exp243",
        "pf_medoid",
        "candidate_only_no_rawtest",
        "core",
        12.499353298613801,
        source_key="exp243_oof",
        source_artifact="exp243 K8 medoid row candidates, four shards",
        value_column="pf_seed_medoid_k8_m0",
        row_idx_column="row_idx",
        id_column=None,
        confidence_columns=(("seed_prediction_std", "pf_seed_std_diag"),),
        confidence_expected=(
            *COMMON_CONFIDENCE_SLOTS,
            "cluster_likelihood_mass",
            "cluster_rank",
            "cluster_gap",
            "assignment_distance",
            "cluster_entropy",
            "seed_prediction_std",
        ),
        selection_reason="best K8 medoid; remaining bank stays external diagnostic",
    ),
    _c(
        "pf_ancc",
        "exp072",
        "pf",
        "available_in_existing_rawtest_cache",
        "core",
        14.493051,
        source_key="exp072_oof",
        source_artifact="exp072 canonical full replay train features",
        value_column="pf_ancc",
        confidence_columns=(("sigma_tvt", "pf_ancc_std"),),
        confidence_expected=(
            *COMMON_CONFIDENCE_SLOTS,
            "particle_tvt_std",
            "resampling_rate",
            "collapse_rate",
        ),
        selection_reason="raw-test-ready alternative PF observation reserve",
    ),
    _c(
        "exp103_xy_likpf_scale_12",
        "exp103",
        "xy_likpf",
        "train_candidate_only",
        "core",
        13.916271,
        source_key="exp103_oof",
        source_artifact="exp103 xy-likPF candidate wide",
        value_column="xy_likpf_scale_12",
        row_idx_column="row_idx",
        id_column=None,
        confidence_columns=(("seed_prediction_std", "xy_likpf_seed_std"),),
        confidence_expected=(
            *COMMON_CONFIDENCE_SLOTS,
            "particle_tvt_std",
            "xy_slope_fit_residual",
            "xy_condition",
            "xy_scale",
            "core_pf_disagreement",
            "seed_prediction_std",
        ),
        selection_reason="best xy-likPF scale representative",
    ),
    _c(
        "hmm_state_selfgr",
        "exp225",
        "hmm_selfgr",
        "rejected_no_rawtest",
        "core",
        14.212951,
        source_key="exp225_oof",
        source_artifact="exp225 state-known self-GR HMM train features",
        value_column="hmm_selfgr_state_known_tvt_curve_boost_only_a070_c100_mean_tvt",
        confidence_columns=(
            ("sigma_tvt", "hmm_selfgr_state_known_tvt_curve_boost_only_a070_c100_std"),
            (
                "source_loglik",
                "hmm_selfgr_state_known_tvt_curve_boost_only_a070_c100_loglik",
            ),
            (
                "candidate_finite_source",
                "hmm_selfgr_state_known_tvt_curve_boost_only_a070_c100_finite",
            ),
            ("selfgr_quality", "self_gr_quality"),
            ("score_margin", "self_gr_peak_gap"),
            ("selfgr_valid", "self_gr_valid"),
            ("state_valid_rate", "self_gr_state_valid_rate"),
        ),
        confidence_expected=(
            *COMMON_CONFIDENCE_SLOTS,
            "posterior_entropy",
            "top1_top2_mass_gap",
            "grid_edge_mass",
            "prefix_sigma",
            "initial_rate_spread",
            "selfgr_quality",
            "selfgr_valid",
            "state_valid_rate",
        ),
        selection_reason="state-known self-GR family representative",
    ),
    _c(
        "beam_mean",
        "exp072",
        "beam",
        "available_in_existing_rawtest_cache",
        "core",
        15.774327,
        source_key="exp072_oof",
        source_artifact="exp072 canonical full replay train features",
        value_column="beam_mean_d",
        transform="anchor_plus",
        anchor_column="last_known_tvt",
        confidence_columns=(("beam_family_std", "beam_std_d"),),
        confidence_expected=(
            *COMMON_CONFIDENCE_SLOTS,
            "beam_family_std",
            "path_straightness",
            "local_slope",
            "local_curvature",
            "boundary_rate",
            "clip_rate",
            "gr_cost",
            "gr_gap",
        ),
        selection_reason="raw-test-ready Beam selector reserve",
    ),
    _c(
        "pf_mix_e02",
        "exp233",
        "mixture_pf",
        "rejected_no_rawtest",
        "external_reference",
        13.519963,
        selection_reason="near-duplicate rejected PF family member",
    ),
    _c(
        "pf_medoid_k8_m1",
        "exp243",
        "pf_medoid",
        "candidate_only_no_rawtest",
        "external_reference",
        12.852916,
        selection_reason="external K8 diagnostic bank",
    ),
    _c(
        "pf_medoid_k8_m2",
        "exp243",
        "pf_medoid",
        "candidate_only_no_rawtest",
        "external_reference",
        13.527396,
        selection_reason="external K8 diagnostic bank",
    ),
    _c(
        "pf_temp_t2",
        "exp232",
        "robust_pf",
        "rejected_no_rawtest",
        "external_reference",
        13.529887,
        selection_reason="near-duplicate rejected PF family member",
    ),
    _c(
        "pf_temp_t4",
        "exp232",
        "robust_pf",
        "rejected_no_rawtest",
        "external_reference",
        13.532730,
        selection_reason="near-duplicate rejected PF family member",
    ),
    _c(
        "pf_mix_e05",
        "exp233",
        "mixture_pf",
        "rejected_no_rawtest",
        "external_reference",
        13.550173,
        selection_reason="near-duplicate rejected PF family member",
    ),
    _c(
        "exp192_pf_ancc",
        "exp192",
        "pf_hard_window",
        "train_cache_only",
        "external_reference",
        13.821165,
        selection_reason="represented by exp192_likpf in hard-window family",
    ),
    _c(
        "pf_medoid_k8_m4",
        "exp243",
        "pf_medoid",
        "candidate_only_no_rawtest",
        "external_reference",
        13.922659,
        selection_reason="external K8 diagnostic bank",
    ),
    _c(
        "pf_medoid_k8_m3",
        "exp243",
        "pf_medoid",
        "candidate_only_no_rawtest",
        "external_reference",
        13.936275,
        selection_reason="external K8 diagnostic bank",
    ),
    _c(
        "exp103_xy_likpf_scale_8",
        "exp103",
        "xy_likpf",
        "train_candidate_only",
        "external_reference",
        13.961015,
        selection_reason="represented by xy-likPF scale 12",
    ),
    _c(
        "exp103_xy_likpf_scale_5",
        "exp103",
        "xy_likpf",
        "train_candidate_only",
        "external_reference",
        14.030092,
        selection_reason="represented by xy-likPF scale 12",
    ),
    _c(
        "exp103_xy_likpf_scale_3",
        "exp103",
        "xy_likpf",
        "train_candidate_only",
        "external_reference",
        14.092584,
        selection_reason="represented by xy-likPF scale 12",
    ),
    _c(
        "exp104_pf_z_seedbag_scale_12",
        "exp104",
        "pf_z_seedbag",
        "train_cache_only",
        "superseded_reference",
        14.145856,
        selection_reason="superseded by exp103 xy-likPF scale 12; no raw-test port",
    ),
    _c(
        "exp104_pf_z_seedbag_scale_8",
        "exp104",
        "pf_z_seedbag",
        "train_cache_only",
        "superseded_reference",
        14.171680,
        selection_reason="superseded by exp103 xy-likPF scale 12; no raw-test port",
    ),
    _c(
        "exp104_pf_z_seedbag_scale_5",
        "exp104",
        "pf_z_seedbag",
        "train_cache_only",
        "superseded_reference",
        14.178127,
        selection_reason="superseded by exp103 xy-likPF scale 12; no raw-test port",
    ),
    _c(
        "exp104_pf_z_seedbag_scale_3",
        "exp104",
        "pf_z_seedbag",
        "train_cache_only",
        "superseded_reference",
        14.215698,
        selection_reason="superseded by exp103 xy-likPF scale 12; no raw-test port",
    ),
    _c(
        "pf_medoid_k8_m5",
        "exp243",
        "pf_medoid",
        "candidate_only_no_rawtest",
        "external_reference",
        14.482404,
        selection_reason="external K8 diagnostic bank",
    ),
    _c(
        "exp103_xy_likpf_mean",
        "exp103",
        "xy_likpf",
        "train_candidate_only",
        "external_reference",
        14.580554,
        selection_reason="represented by xy-likPF scale 12",
    ),
    _c(
        "exp104_pf_z_seedbag_mean",
        "exp104",
        "pf_z_seedbag",
        "train_cache_only",
        "superseded_reference",
        14.587060,
        selection_reason="superseded by exp103 xy-likPF scale 12; no raw-test port",
    ),
    _c(
        "exp192_beam_mean",
        "exp192",
        "beam_hard_window",
        "train_cache_only",
        "external_reference",
        15.677016,
        selection_reason="represented by exp192_likpf inside hard-window family",
    ),
    _c(
        "pf_medoid_k8_m6",
        "exp243",
        "pf_medoid",
        "candidate_only_no_rawtest",
        "external_reference",
        15.721456,
        selection_reason="external K8 diagnostic bank",
    ),
)


PAIR_SHORTLIST: tuple[PairSpec, ...] = (
    PairSpec(
        "exp226_k16__selfgr_hmm_a070",
        "exp226_k16",
        "selfgr_hmm_a070",
        "raw-test",
        8.53271508582674,
        8.401770080850183,
        0.3436414069429523,
        5,
        (
            (0.6340150366528021, 0.3659849633471979),
            (0.6455624537020387, 0.3544375462979613),
            (0.6109138808116157, 0.3890861191883843),
            (0.6733653293669218, 0.3266346706330782),
            (0.6289790910341397, 0.3710209089658603),
        ),
        "best primitive fixed pair; 5/5 folds",
    ),
    PairSpec(
        "exp226_k16__exact_hmm",
        "exp226_k16",
        "exact_hmm",
        "raw-test",
        8.635074349079765,
        8.419750627213405,
        0.2970624966505328,
        5,
        (
            (0.6645283683824653, 0.3354716316175347),
            (0.6807880917744590, 0.3192119082255410),
            (0.6401752123854650, 0.3598247876145350),
            (0.6921670136061129, 0.3078329863938871),
            (0.6368094838882818, 0.3631905161117182),
        ),
        "low residual correlation; 5/5 folds",
    ),
    PairSpec(
        "exp226_k16__likpf_mean",
        "exp226_k16",
        "likpf_mean",
        "raw-test",
        8.813822138574935,
        8.610566968561942,
        0.3998934019736887,
        5,
        (
            (0.6582154718638368, 0.3417845281361632),
            (0.6883229743502485, 0.3116770256497515),
            (0.6621652484515846, 0.3378347515484154),
            (0.6816361376819075, 0.3183638623180925),
            (0.6475243952206489, 0.3524756047793511),
        ),
        "PF reference pair; 5/5 folds",
    ),
    PairSpec(
        "selfgr_hmm_a070__likpf_mean",
        "selfgr_hmm_a070",
        "likpf_mean",
        "raw-test",
        10.123456412256017,
        10.14915065740638,
        0.5572683654088965,
        5,
        (
            (0.5117285263758404, 0.4882714736241596),
            (0.5408660685320188, 0.4591339314679812),
            (0.5615019516151335, 0.4384980483848665),
            (0.4999241639066622, 0.5000758360933378),
            (0.5046262918159380, 0.4953737081840620),
        ),
        "component pair for raw-test-ready outer-convex diagnostic",
    ),
    PairSpec(
        "likpf_mean__exact_hmm",
        "likpf_mean",
        "exact_hmm",
        "raw-test",
        10.269696262188775,
        10.285675660797963,
        0.523403103053945,
        5,
        (
            (0.5492333961112691, 0.4507666038887309),
            (0.5259529437266802, 0.4740470562733198),
            (0.4995170024210371, 0.5004829975789629),
            (0.5528900442084341, 0.4471099557915659),
            (0.5238687145210947, 0.4761312854789053),
        ),
        "primitive definition of blend_likpf_hmm_w500",
    ),
    PairSpec(
        "exp226_k16__hmm_peer_atlas",
        "exp226_k16",
        "hmm_peer_atlas",
        "train-only",
        8.60748413546012,
        8.448157316705423,
        0.33749165538695425,
        5,
        (
            (0.6494937630038083, 0.3505062369961917),
            (0.6644224268263081, 0.3355775731736919),
            (0.6273445945111445, 0.3726554054888555),
            (0.6853314089718912, 0.3146685910281089),
            (0.6263929066133134, 0.3736070933866866),
        ),
        "atlas family representative with large fixed gain",
    ),
    PairSpec(
        "exp226_k16__exp192_likpf",
        "exp226_k16",
        "exp192_likpf",
        "train-only",
        8.72740600954707,
        8.530865141230338,
        0.37909677299086225,
        5,
        (
            (0.6503926234634996, 0.3496073765365004),
            (0.6713083128596033, 0.3286916871403967),
            (0.6452881281096942, 0.3547118718903058),
            (0.6708157844039870, 0.3291842155960130),
            (0.6578433635132761, 0.3421566364867239),
        ),
        "hard-window PF representative",
    ),
    PairSpec(
        "exp226_k16__pf_medoid_k8_m0",
        "exp226_k16",
        "pf_medoid_k8_m0",
        "train-only",
        8.989906707165222,
        8.61220034702476,
        0.33169659766086357,
        5,
        (
            (0.6943834372439473, 0.3056165627560527),
            (0.7101665548784330, 0.2898334451215670),
            (0.6923387752147157, 0.3076612247852844),
            (0.7172271471856742, 0.2827728528143258),
            (0.6955347548490929, 0.3044652451509071),
        ),
        "best medoid family representative",
    ),
)


NAMED_COMBINATIONS: dict[str, dict[str, Any]] = {
    "blend_likpf_hmm_w500": {
        "kind": "alias",
        "deployability": "raw-test",
        "components": ["likpf_mean", "exact_hmm"],
        "weights": {"likpf_mean": 0.5, "exact_hmm": 0.5},
        "formula": "0.5*likpf_mean + 0.5*exact_hmm",
        "fixed_oof_rmse": 10.269696262188775,
    },
    "exp226_w500_50_50": {
        "kind": "fixed",
        "deployability": "raw-test",
        "components": ["exp226_k16", "likpf_mean", "exact_hmm"],
        "weights": {"exp226_k16": 0.5, "likpf_mean": 0.25, "exact_hmm": 0.25},
        "formula": "0.5*exp226_k16 + 0.25*likpf_mean + 0.25*exact_hmm",
        "fixed_oof_rmse": 8.238331,
        "folds_beating_exp226": 5,
    },
    "exp226_selfgr_a070_likpf_outer_convex": {
        "kind": "outer_crossfit",
        "deployability": "raw-test-components_diagnostic-only",
        "components": ["exp226_k16", "selfgr_hmm_a070", "likpf_mean"],
        "fold_weights": [
            {
                "exp226_k16": 0.5555850057818239,
                "selfgr_hmm_a070": 0.26710808615626924,
                "likpf_mean": 0.1773069080619069,
            },
            {
                "exp226_k16": 0.5771565544388395,
                "selfgr_hmm_a070": 0.2678915207655287,
                "likpf_mean": 0.15495192479563175,
            },
            {
                "exp226_k16": 0.5388301131722688,
                "selfgr_hmm_a070": 0.2901831159535959,
                "likpf_mean": 0.17098677087413536,
            },
            {
                "exp226_k16": 0.5895718220947345,
                "selfgr_hmm_a070": 0.22124220256610086,
                "likpf_mean": 0.18918597533916465,
            },
            {
                "exp226_k16": 0.5543177079614755,
                "selfgr_hmm_a070": 0.2664754263619408,
                "likpf_mean": 0.1792068656765838,
            },
        ],
        "crossfit_oof_rmse": 8.240049681137599,
        "folds_beating_exp226": 5,
    },
    "exp226_exp192_likpf_exact_hmm_outer_convex": {
        "kind": "outer_crossfit",
        "deployability": "train-only",
        "components": ["exp226_k16", "exp192_likpf", "exact_hmm"],
        "fold_weights": [
            {
                "exp226_k16": 0.5667581389558393,
                "exp192_likpf": 0.20449180034350534,
                "exact_hmm": 0.2287500607006554,
            },
            {
                "exp226_k16": 0.591809726127703,
                "exp192_likpf": 0.1942945994450357,
                "exact_hmm": 0.21389567442726135,
            },
            {
                "exp226_k16": 0.5480450971859819,
                "exp192_likpf": 0.19894598671870867,
                "exact_hmm": 0.25300891609530946,
            },
            {
                "exp226_k16": 0.5922960430013566,
                "exp192_likpf": 0.20390507096940994,
                "exact_hmm": 0.20379888602923352,
            },
            {
                "exp226_k16": 0.560094329520119,
                "exp192_likpf": 0.16966551859848492,
                "exact_hmm": 0.27024015188139616,
            },
        ],
        "crossfit_oof_rmse": 8.209224972738143,
        "folds_beating_exp226": 5,
    },
}


CORE_CANDIDATE_IDS = tuple(c.candidate_id for c in REFERENCE_CANDIDATES if c.is_core)
RAWTEST_CORE_CANDIDATE_IDS = tuple(
    c.candidate_id for c in REFERENCE_CANDIDATES if c.is_core and c.is_rawtest_ready
)


def candidate_by_id(candidate_id: str) -> CandidateSpec:
    matches = [item for item in REFERENCE_CANDIDATES if item.candidate_id == candidate_id]
    if len(matches) != 1:
        raise KeyError(candidate_id)
    return matches[0]


def formula_graph() -> dict[str, list[str]]:
    graph = {candidate_id: [] for candidate_id in CORE_CANDIDATE_IDS}
    graph.update(
        {
            pair.pair_id: [pair.left, pair.right]
            for pair in PAIR_SHORTLIST
        }
    )
    graph.update(
        {
            name: list(spec["components"])
            for name, spec in NAMED_COMBINATIONS.items()
        }
    )
    return graph


def topological_formula_order(graph: dict[str, list[str]] | None = None) -> list[str]:
    graph = formula_graph() if graph is None else graph
    state: dict[str, int] = {}
    order: list[str] = []

    def visit(node: str) -> None:
        status = state.get(node, 0)
        if status == 1:
            raise ValueError(f"formula cycle detected at {node}")
        if status == 2:
            return
        if node not in graph:
            raise ValueError(f"formula dependency not registered: {node}")
        state[node] = 1
        for child in graph[node]:
            visit(child)
        state[node] = 2
        order.append(node)

    for name in graph:
        visit(name)
    return order


def validate_selectable_names(names: Iterable[str]) -> None:
    selected = set(names)
    w500 = NAMED_COMBINATIONS["blend_likpf_hmm_w500"]
    if "blend_likpf_hmm_w500" in selected and selected.intersection(w500["components"]):
        raise ValueError("w500 alias and its primitive parents cannot be selectable together")
    pair_ids = {pair.pair_id for pair in PAIR_SHORTLIST}
    if selected.intersection(pair_ids) and selected.intersection(NAMED_COMBINATIONS):
        raise ValueError("pair/named recursive selectable closure is forbidden")


def validate_contract() -> dict[str, int]:
    ids = [item.candidate_id for item in REFERENCE_CANDIDATES]
    if len(ids) != len(set(ids)):
        raise ValueError("candidate ids must be unique")
    if len(ids) != 33:
        raise ValueError(f"reference inventory must contain 33 candidates, got {len(ids)}")
    if len(CORE_CANDIDATE_IDS) != 12:
        raise ValueError("core inventory must contain 12 candidates")
    if len(RAWTEST_CORE_CANDIDATE_IDS) != 6:
        raise ValueError("raw-test-ready core must contain 6 candidates")
    if any(item.global_rmse >= ANCHOR_RMSE for item in REFERENCE_CANDIDATES):
        raise ValueError("reference candidate does not beat last_anchor")
    if set(ids).intersection(FORBIDDEN_CANDIDATE_IDS):
        raise ValueError("HMM+LGB candidates are forbidden from this contract")
    exp104 = [item for item in REFERENCE_CANDIDATES if item.source_exp == "exp104"]
    if len(exp104) != 5 or any(item.cache_role != "superseded_reference" for item in exp104):
        raise ValueError("exp104 must remain a five-candidate superseded reference")
    if len(PAIR_SHORTLIST) != 8:
        raise ValueError("pair shortlist must contain exactly eight pairs")
    if sum(pair.tier == "raw-test" for pair in PAIR_SHORTLIST) != 5:
        raise ValueError("pair shortlist must contain exactly five raw-test pairs")
    if len(NAMED_COMBINATIONS) != 4:
        raise ValueError("one w500 alias and three named combinations are required")
    topological_formula_order()
    return {
        "reference_candidates": len(ids),
        "core_candidates": len(CORE_CANDIDATE_IDS),
        "rawtest_core_candidates": len(RAWTEST_CORE_CANDIDATE_IDS),
        "shortlisted_pairs": len(PAIR_SHORTLIST),
        "rawtest_pairs": sum(pair.tier == "raw-test" for pair in PAIR_SHORTLIST),
        "named_triples": len(NAMED_COMBINATIONS) - 1,
    }
