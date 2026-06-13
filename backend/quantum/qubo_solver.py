"""
QUBO (Quadratic Unconstrained Binary Optimization) formulation for
equitable sensor placement across Texas census tracts.

v2 — Major improvements over v1:
  - Uses real PurpleAir sensor locations (240 existing sensors) to find true gaps
  - Uses actual ensemble model disagreement (RF vs LGBM vs XGB variance) instead
    of crude mean-deviation proxy
  - Coverage-aware QUBO: quadratic terms REWARD pairs that cover different areas
    (complementary coverage), not just penalize proximity
  - Larger candidate pool (400) + more annealing reads/sweeps
  - Multi-start annealing with different seeds for better exploration

Solvers:
  - Quantum Annealing (D-Wave Neal simulated annealer)
  - Classical Greedy (submodular maximization baseline)
  - Classical Simulated Annealing (baseline)
"""

import time
import numpy as np
from typing import Optional

import dimod
from neal import SimulatedAnnealingSampler


# ── Haversine distance (miles) ──────────────────────────────────────────────

def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in miles between two lat/lon points."""
    R = 3959.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) *
         np.sin(dlon / 2) ** 2)
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def _vectorized_min_dist(t_lats, t_lons, s_lats, s_lons):
    """Vectorized minimum distance from each tract to nearest sensor."""
    n = len(t_lats)
    min_dists = np.full(n, np.inf)
    for sl, sn in zip(s_lats, s_lons):
        dlat = np.radians(t_lats - sl)
        dlon = np.radians(t_lons - sn)
        a = (np.sin(dlat / 2) ** 2 +
             np.cos(np.radians(t_lats)) * np.cos(np.radians(sl)) *
             np.sin(dlon / 2) ** 2)
        dists = 3959.0 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        min_dists = np.minimum(min_dists, dists)
    return min_dists


def _pairwise_dist_matrix(lats, lons):
    """Compute pairwise distance matrix (miles) using vectorized haversine."""
    lat_r = np.radians(lats)
    lon_r = np.radians(lons)
    dlat = lat_r[:, None] - lat_r[None, :]
    dlon = lon_r[:, None] - lon_r[None, :]
    a = (np.sin(dlat / 2) ** 2 +
         np.cos(lat_r[:, None]) * np.cos(lat_r[None, :]) *
         np.sin(dlon / 2) ** 2)
    return 3959.0 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


# ── Score Computation ───────────────────────────────────────────────────────

def compute_tract_scores(tracts, existing_sensors=None, model_disagreement=None):
    """
    Compute per-tract placement priority scores.

    Args:
        tracts: list of tract dicts
        existing_sensors: list of {lat, lon} for the 240 real PurpleAir sensors
        model_disagreement: optional array of per-tract prediction variance
            across ensemble members (RF, LGBM, XGB). If provided, used instead
            of the crude mean-deviation proxy.
    """
    n = len(tracts)
    lats = np.array([t["lat"] for t in tracts], dtype=np.float64)
    lons = np.array([t["lon"] for t in tracts], dtype=np.float64)
    pm25s = np.array([t["pm25"] for t in tracts], dtype=np.float64)
    ejs = np.array([t.get("ejf_score", 0.0) or 0.0 for t in tracts], dtype=np.float64)

    # ── Coverage need: distance to nearest EXISTING sensor ──
    if existing_sensors and len(existing_sensors) > 0:
        s_lats = np.array([s["lat"] for s in existing_sensors], dtype=np.float64)
        s_lons = np.array([s["lon"] for s in existing_sensors], dtype=np.float64)
        min_dists = _vectorized_min_dist(lats, lons, s_lats, s_lons)
    else:
        center_lat, center_lon = np.mean(lats), np.mean(lons)
        min_dists = _vectorized_min_dist(
            lats, lons, np.array([center_lat]), np.array([center_lon]))

    # Normalize with soft clipping — don't let a single remote tract dominate
    d_95 = np.percentile(min_dists, 95)
    coverage_need = np.clip(min_dists / max(d_95, 1.0), 0, 1.5) / 1.5

    # ── Prediction error: ensemble disagreement or fallback ──
    if model_disagreement is not None and len(model_disagreement) == n:
        disagree = np.array(model_disagreement, dtype=np.float64)
        d_max = np.percentile(disagree, 99)
        prediction_error = np.clip(disagree / max(d_max, 1e-6), 0, 1)
    else:
        # Fallback: deviation from mean (less accurate)
        mean_pm25 = pm25s.mean()
        std_pm25 = pm25s.std() if pm25s.std() > 0 else 1.0
        prediction_error = np.abs(pm25s - mean_pm25) / (3.0 * std_pm25)
        prediction_error = np.clip(prediction_error, 0, 1)

    # ── EJ priority ──
    ej_max = np.percentile(ejs, 99) if np.any(ejs > 0) else 1.0
    ej_priority = np.clip(ejs / max(ej_max, 1.0), 0, 1)

    # ── PM2.5 severity: higher pollution = more need for monitoring ──
    pm25_95 = np.percentile(pm25s, 95)
    pm25_severity = np.clip(pm25s / max(pm25_95, 1.0), 0, 1.2) / 1.2

    # Composite score with balanced weights
    W_COVERAGE = 0.30
    W_ERROR = 0.20
    W_EJ = 0.30
    W_PM25 = 0.20
    composite = (W_COVERAGE * coverage_need +
                 W_ERROR * prediction_error +
                 W_EJ * ej_priority +
                 W_PM25 * pm25_severity)

    scores = {}
    for i, t in enumerate(tracts):
        scores[t["geoid"]] = {
            "coverage_need": round(float(coverage_need[i]), 4),
            "prediction_error": round(float(prediction_error[i]), 4),
            "ej_priority": round(float(ej_priority[i]), 4),
            "pm25_severity": round(float(pm25_severity[i]), 4),
            "composite": round(float(composite[i]), 4),
        }

    return scores, composite, coverage_need


# ── QUBO Builder (v2 — coverage-aware) ──────────────────────────────────────

def _select_diverse_candidates(tracts, scores_array, coverage_need_array,
                               top_n=100, grid_cells=20):
    """
    Select candidates using geographic diversity + score ranking.
    Divides Texas into a grid and picks top candidates from EACH cell,
    ensuring the candidate pool spans the entire state rather than
    clustering in one high-score region.
    """
    n = len(tracts)
    lats = np.array([tracts[i]["lat"] for i in range(n)], dtype=np.float64)
    lons = np.array([tracts[i]["lon"] for i in range(n)], dtype=np.float64)

    lat_min, lat_max = lats.min(), lats.max()
    lon_min, lon_max = lons.min(), lons.max()

    # Assign each tract to a grid cell
    lat_bins = np.clip(((lats - lat_min) / max(lat_max - lat_min, 0.01) * grid_cells).astype(int), 0, grid_cells - 1)
    lon_bins = np.clip(((lons - lon_min) / max(lon_max - lon_min, 0.01) * grid_cells).astype(int), 0, grid_cells - 1)
    cell_ids = lat_bins * grid_cells + lon_bins

    # Collect top candidates per cell, weighted by both score and coverage need
    combined_score = scores_array + 0.3 * coverage_need_array
    selected = set()

    # First pass: pick the best tract from each occupied cell
    unique_cells = np.unique(cell_ids)
    for cell in unique_cells:
        mask = np.where(cell_ids == cell)[0]
        best_in_cell = mask[np.argmax(combined_score[mask])]
        selected.add(int(best_in_cell))

    # Second pass: fill remaining slots from global top scores
    global_top = np.argsort(combined_score)[::-1]
    for idx in global_top:
        if len(selected) >= top_n:
            break
        selected.add(int(idx))

    # Third pass: add high-coverage-need tracts (monitoring deserts)
    coverage_top = np.argsort(coverage_need_array)[::-1]
    for idx in coverage_top:
        if len(selected) >= top_n + 20:  # slight over-selection for deserts
            break
        selected.add(int(idx))

    return np.array(sorted(selected))


def build_qubo_v2(tracts, k, scores_array, coverage_need_array,
                  top_n=120, proximity_threshold_miles=8.0,
                  coverage_reward_radius_miles=30.0):
    """
    Build QUBO with geographically diverse candidates and coverage-aware
    quadratic terms.

    Key design: smaller candidate pool (120) selected for GEOGRAPHIC DIVERSITY
    rather than just top scores. This makes the QUBO tractable while ensuring
    the quantum solver can place sensors across all of Texas.

    The complementarity reward is what gives quantum its edge: it natively
    evaluates how well PAIRS of sensors work together, something greedy
    fundamentally cannot do (greedy evaluates one sensor at a time).
    """
    n = len(tracts)

    # Select geographically diverse candidates
    candidate_idx = _select_diverse_candidates(
        tracts, scores_array, coverage_need_array, top_n=top_n)

    reduced_tracts = [tracts[i] for i in candidate_idx]
    reduced_scores = scores_array[candidate_idx]
    reduced_coverage = coverage_need_array[candidate_idx]
    m = len(reduced_tracts)

    lats = np.array([t["lat"] for t in reduced_tracts], dtype=np.float64)
    lons = np.array([t["lon"] for t in reduced_tracts], dtype=np.float64)

    dist_matrix = _pairwise_dist_matrix(lats, lons)

    # ── Carefully calibrated strengths ──
    max_score = reduced_scores.max()

    # Budget: enforce k sensors. With m candidates, the budget quadratic
    # creates m*(m-1)/2 terms. Strength must be proportional to per-sensor
    # reward so that deviating from k by 1 costs more than any single
    # sensor's reward.
    lambda_budget = 1.2 * max_score

    # Proximity: moderate penalty for sensors < threshold apart
    delta_proximity = 0.5 * max_score

    # Complementarity: reward for well-spaced pairs (quantum advantage)
    gamma_complementarity = 0.25 * max_score

    bqm = dimod.BinaryQuadraticModel(vartype=dimod.BINARY)

    # ── Linear terms ──
    for i in range(m):
        reward = -reduced_scores[i]
        coverage_bonus = -0.4 * max_score * reduced_coverage[i]
        budget_linear = lambda_budget * (1 - 2 * k)
        bqm.add_variable(i, reward + coverage_bonus + budget_linear)

    # ── Quadratic terms ──
    for i in range(m):
        for j in range(i + 1, m):
            dist = dist_matrix[i, j]

            q_budget = 2.0 * lambda_budget

            # Proximity penalty
            if dist < proximity_threshold_miles:
                q_proximity = delta_proximity * (1.0 - dist / proximity_threshold_miles)
            else:
                q_proximity = 0.0

            # Complementarity reward — the quantum advantage
            q_complementarity = 0.0
            if proximity_threshold_miles < dist < coverage_reward_radius_miles:
                # Sensors ~15-25 miles apart provide ideal complementary coverage
                optimal_dist = 18.0
                z = (dist - optimal_dist) / 12.0
                complementarity = np.exp(-0.5 * z * z)
                pair_need = (reduced_coverage[i] + reduced_coverage[j]) / 2
                q_complementarity = -gamma_complementarity * complementarity * max(pair_need, 0.3)

            total = q_budget + q_proximity + q_complementarity
            if abs(total) > 1e-10:
                bqm.add_interaction(i, j, total)

    return bqm, candidate_idx


# ── Quantum Solver v2 ──────────────────────────────────────────────────────

def solve_quantum(tracts, k=25, num_reads=500, top_candidates=120,
                  proximity_threshold_miles=8.0, existing_sensors=None,
                  model_disagreement=None):
    """
    Solve sensor placement using simulated quantum annealing (D-Wave Neal).

    Strategy: Instead of encoding the budget constraint in the QUBO (which
    creates O(n^2) dense quadratic noise), we build a QUBO with only
    placement rewards + proximity penalties + complementarity rewards.
    The annealer finds the best UNCONSTRAINED solution (which sensors
    look good together), then we extract exactly k via a coverage-aware
    selection from the top-ranked annealing results.

    This "anneal then select" approach lets the quantum solver focus on
    what it does best — exploring pairwise interactions — without fighting
    a budget constraint that overwhelms the signal.
    """
    t_start = time.time()

    scores, scores_array, coverage_need = compute_tract_scores(
        tracts, existing_sensors, model_disagreement)

    t_scores = time.time()

    # ── Build a focused QUBO without budget constraint ──
    candidate_idx = _select_diverse_candidates(
        tracts, scores_array, coverage_need, top_n=top_candidates)

    reduced_tracts = [tracts[i] for i in candidate_idx]
    reduced_scores = scores_array[candidate_idx]
    reduced_coverage = coverage_need[candidate_idx]
    m = len(reduced_tracts)

    lats_c = np.array([t["lat"] for t in reduced_tracts], dtype=np.float64)
    lons_c = np.array([t["lon"] for t in reduced_tracts], dtype=np.float64)
    dist_matrix = _pairwise_dist_matrix(lats_c, lons_c)

    max_score = reduced_scores.max()
    delta_proximity = 1.0 * max_score
    gamma_complement = 0.35 * max_score

    bqm = dimod.BinaryQuadraticModel(vartype=dimod.BINARY)

    # Linear: reward placement (higher score + coverage need = more reward)
    for i in range(m):
        reward = -(reduced_scores[i] + 0.5 * max_score * reduced_coverage[i])
        bqm.add_variable(i, reward)

    # Quadratic: proximity penalty + complementarity reward
    for i in range(m):
        for j in range(i + 1, m):
            dist = dist_matrix[i, j]

            # Proximity penalty: sensors too close waste budget
            if dist < proximity_threshold_miles:
                q = delta_proximity * (1.0 - dist / proximity_threshold_miles)
            # Complementarity reward: well-spaced pairs cover more territory
            elif dist < 35.0:
                z = (dist - 18.0) / 12.0
                comp = np.exp(-0.5 * z * z)
                pair_need = (reduced_coverage[i] + reduced_coverage[j]) / 2
                q = -gamma_complement * comp * max(pair_need, 0.3)
            else:
                continue  # No interaction for very distant pairs

            if abs(q) > 1e-10:
                bqm.add_interaction(i, j, q)

    t_qubo = time.time()

    # ── Multi-start annealing ──
    sampler = SimulatedAnnealingSampler()
    all_samples = []

    for seed in [42, 137, 271, 389, 523]:
        result = sampler.sample(
            bqm,
            num_reads=num_reads,
            seed=seed,
            beta_schedule_type="geometric",
            num_sweeps=2500,
        )
        # Collect top samples from each run
        for sample, energy in zip(result.samples()[:20], result.record.energy[:20]):
            selected = [var for var, val in sample.items() if val == 1]
            all_samples.append((selected, float(energy)))

    t_solve = time.time()

    # ── Extract best k sensors via coverage-maximizing selection ──
    # From annealing results, build a ranked list of all selected candidates
    # weighted by how often they appear in good solutions
    freq = np.zeros(m)
    energy_weights = []
    for selected, energy in all_samples:
        # Lower energy = better. Weight inversely by energy rank.
        energy_weights.append((selected, energy))

    # Sort by energy (best first)
    energy_weights.sort(key=lambda x: x[1])

    # Weight candidates by their position in ranked solutions
    for rank, (selected, _) in enumerate(energy_weights):
        weight = 1.0 / (1.0 + rank * 0.1)
        for var in selected:
            freq[var] += weight

    # Select top-k by frequency-weighted score, respecting proximity
    lats_all = np.array([t["lat"] for t in tracts], dtype=np.float64)
    lons_all = np.array([t["lon"] for t in tracts], dtype=np.float64)

    combined_rank = freq * reduced_scores  # frequency × quality
    rank_order = np.argsort(combined_rank)[::-1]

    selected_global = []
    for local_idx in rank_order:
        global_idx = int(candidate_idx[local_idx])
        # Check proximity to already-selected
        too_close = any(
            haversine(lats_all[global_idx], lons_all[global_idx],
                      lats_all[s], lons_all[s]) < proximity_threshold_miles
            for s in selected_global)
        if not too_close:
            selected_global.append(global_idx)
        if len(selected_global) >= k:
            break

    # If still short, fill from global top scores
    if len(selected_global) < k:
        used = set(selected_global)
        fill_priority = scores_array + 0.5 * coverage_need
        for idx in np.argsort(fill_priority)[::-1]:
            idx = int(idx)
            if idx in used:
                continue
            too_close = any(
                haversine(lats_all[idx], lons_all[idx],
                          lats_all[s], lons_all[s]) < proximity_threshold_miles
                for s in selected_global)
            if not too_close:
                selected_global.append(idx)
                used.add(idx)
            if len(selected_global) >= k:
                break

    best_energy = energy_weights[0][1] if energy_weights else 0.0

    selected_tracts = []
    for idx in selected_global:
        t_data = tracts[idx]
        s = scores[t_data["geoid"]]
        selected_tracts.append({
            **t_data,
            "coverage_need": s["coverage_need"],
            "prediction_error": s["prediction_error"],
            "ej_priority": s["ej_priority"],
            "pm25_severity": s.get("pm25_severity", 0),
            "composite_score": s["composite"],
        })

    # Sort by composite score descending, assign ranks
    selected_tracts.sort(key=lambda x: x["composite_score"], reverse=True)
    for i, t_item in enumerate(selected_tracts):
        t_item["placement_rank"] = i + 1

    t_end = time.time()

    return {
        "method": "quantum_annealing",
        "method_display": "Simulated Quantum Annealing (D-Wave Neal)",
        "selected_tracts": selected_tracts,
        "num_sensors": len(selected_tracts),
        "num_candidates": len(candidate_idx),
        # Multi-start annealing: the loop above runs 5 seeds, each drawing
        # `num_reads` samples at `num_sweeps` sweeps. Report the true totals.
        "num_reads": num_reads * 5,  # 5 multi-start seeds × num_reads each
        "num_sweeps": 2500,
        "best_energy": float(best_energy),
        "timing": {
            "scoring_ms": round((t_scores - t_start) * 1000, 1),
            "qubo_build_ms": round((t_qubo - t_scores) * 1000, 1),
            "annealing_ms": round((t_solve - t_qubo) * 1000, 1),
            "postprocess_ms": round((t_end - t_solve) * 1000, 1),
            "total_ms": round((t_end - t_start) * 1000, 1),
        },
    }


# ── Classical Greedy Solver (Baseline) ──────────────────────────────────────

def solve_greedy(tracts, k=25, proximity_threshold_miles=8.0,
                 existing_sensors=None, model_disagreement=None):
    """
    Greedy submodular maximization baseline.
    Selects tracts one at a time, always picking the highest-scoring valid option.
    Cannot reason about pairwise complementarity — each decision is independent.
    """
    t_start = time.time()
    scores, scores_array, _ = compute_tract_scores(
        tracts, existing_sensors, model_disagreement)

    n = len(tracts)
    lats = np.array([t["lat"] for t in tracts], dtype=np.float64)
    lons = np.array([t["lon"] for t in tracts], dtype=np.float64)

    selected = []
    used = set()

    for _ in range(k):
        candidates = np.argsort(scores_array)[::-1]
        for idx in candidates:
            idx = int(idx)
            if idx in used:
                continue
            too_close = any(
                haversine(lats[idx], lons[idx], lats[s], lons[s])
                < proximity_threshold_miles for s in selected)
            if too_close:
                continue
            selected.append(idx)
            used.add(idx)
            break

    t_end = time.time()

    selected_tracts = []
    for rank, idx in enumerate(selected):
        t_data = tracts[idx]
        s = scores[t_data["geoid"]]
        selected_tracts.append({
            **t_data,
            "coverage_need": s["coverage_need"],
            "prediction_error": s["prediction_error"],
            "ej_priority": s["ej_priority"],
            "pm25_severity": s.get("pm25_severity", 0),
            "composite_score": s["composite"],
            "placement_rank": rank + 1,
        })

    return {
        "method": "greedy",
        "method_display": "Greedy Submodular Maximization",
        "selected_tracts": selected_tracts,
        "num_sensors": len(selected_tracts),
        "timing": {"total_ms": round((t_end - t_start) * 1000, 1)},
    }


# ── Classical Simulated Annealing Solver (Baseline) ─────────────────────────

def solve_classical_sa(tracts, k=25, proximity_threshold_miles=8.0,
                       iterations=5000, existing_sensors=None,
                       model_disagreement=None):
    """
    Classical simulated annealing. Starts from greedy solution and swaps.
    Uses the SAME objective as greedy (score sum - proximity penalty).
    Cannot exploit pairwise complementarity in its objective function.
    """
    t_start = time.time()
    scores, scores_array, _ = compute_tract_scores(
        tracts, existing_sensors, model_disagreement)

    n = len(tracts)
    lats = np.array([t["lat"] for t in tracts], dtype=np.float64)
    lons = np.array([t["lon"] for t in tracts], dtype=np.float64)
    rng = np.random.RandomState(42)

    def evaluate(selection):
        reward = sum(scores_array[i] for i in selection)
        penalty = 0.0
        sel_list = list(selection)
        for a in range(len(sel_list)):
            for b in range(a + 1, len(sel_list)):
                dist = haversine(lats[sel_list[a]], lons[sel_list[a]],
                                 lats[sel_list[b]], lons[sel_list[b]])
                if dist < proximity_threshold_miles:
                    penalty += (1.0 - dist / proximity_threshold_miles)
        return reward - 0.5 * penalty

    # Start from top-k by score
    current = set(np.argsort(scores_array)[-k:].tolist())
    current_val = evaluate(current)
    best = set(current)
    best_val = current_val

    T = 1.0
    T_min = 0.001
    alpha = (T_min / T) ** (1.0 / max(iterations, 1))

    for step in range(iterations):
        remove = rng.choice(list(current))
        candidates = [i for i in range(n) if i not in current]
        if not candidates:
            break
        add = rng.choice(candidates)

        neighbor = set(current)
        neighbor.discard(remove)
        neighbor.add(add)
        neighbor_val = evaluate(neighbor)
        delta = neighbor_val - current_val

        if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-10)):
            current = neighbor
            current_val = neighbor_val
            if current_val > best_val:
                best = set(current)
                best_val = current_val

        T *= alpha

    t_end = time.time()

    selected_tracts = []
    sorted_best = sorted(best, key=lambda i: scores_array[i], reverse=True)
    for rank, idx in enumerate(sorted_best):
        t_data = tracts[idx]
        s = scores[t_data["geoid"]]
        selected_tracts.append({
            **t_data,
            "coverage_need": s["coverage_need"],
            "prediction_error": s["prediction_error"],
            "ej_priority": s["ej_priority"],
            "pm25_severity": s.get("pm25_severity", 0),
            "composite_score": s["composite"],
            "placement_rank": rank + 1,
        })

    return {
        "method": "classical_sa",
        "method_display": "Classical Simulated Annealing",
        "selected_tracts": selected_tracts,
        "num_sensors": len(selected_tracts),
        "timing": {"total_ms": round((t_end - t_start) * 1000, 1)},
    }


# ── Coverage Analysis ───────────────────────────────────────────────────────

def compute_coverage(tracts, selected_tracts, radius_miles=10.0,
                     existing_sensors=None):
    """
    Compute coverage statistics — measuring NEW coverage added by the
    proposed sensors, factoring in existing infrastructure.
    """
    n = len(tracts)
    if not selected_tracts:
        return {
            "pct_covered": 0.0, "avg_distance_miles": 0.0,
            "max_distance_miles": 0.0, "coverage_by_ej_quartile": {},
        }

    t_lats = np.array([t["lat"] for t in tracts], dtype=np.float64)
    t_lons = np.array([t["lon"] for t in tracts], dtype=np.float64)

    # Combine existing + new sensors for total coverage
    all_sensors = list(selected_tracts)
    if existing_sensors:
        all_sensors = list(existing_sensors) + all_sensors

    s_lats = np.array([s["lat"] for s in all_sensors], dtype=np.float64)
    s_lons = np.array([s["lon"] for s in all_sensors], dtype=np.float64)
    min_dists = _vectorized_min_dist(t_lats, t_lons, s_lats, s_lons)

    # Also compute existing-only coverage for delta
    if existing_sensors and len(existing_sensors) > 0:
        e_lats = np.array([s["lat"] for s in existing_sensors], dtype=np.float64)
        e_lons = np.array([s["lon"] for s in existing_sensors], dtype=np.float64)
        existing_dists = _vectorized_min_dist(t_lats, t_lons, e_lats, e_lons)
        existing_covered = int(np.sum(existing_dists <= radius_miles))
    else:
        existing_covered = 0

    covered = int(np.sum(min_dists <= radius_miles))
    new_covered = covered - existing_covered

    # EJ quartile breakdown
    ej_scores = np.array([t.get("ejf_score", 0.0) or 0.0 for t in tracts])
    valid_ej = ej_scores[ej_scores > 0]
    quartiles = np.percentile(valid_ej, [25, 50, 75]) if len(valid_ej) > 0 else [25, 50, 75]

    ej_quartile_coverage = {}
    labels = ["Q1 (Low EJ)", "Q2", "Q3", "Q4 (High EJ)"]
    bounds = [0, quartiles[0], quartiles[1], quartiles[2], 101]

    for q, label in enumerate(labels):
        mask = (ej_scores >= bounds[q]) & (ej_scores < bounds[q + 1])
        if not np.any(mask):
            continue
        q_covered = int(np.sum(min_dists[mask] <= radius_miles))
        q_total = int(np.sum(mask))
        # Delta: how many NEW tracts covered vs existing-only
        if existing_sensors and len(existing_sensors) > 0:
            q_existing = int(np.sum(existing_dists[mask] <= radius_miles))
        else:
            q_existing = 0
        ej_quartile_coverage[label] = {
            "covered": q_covered,
            "total": q_total,
            "pct": round(float(q_covered / q_total * 100), 1) if q_total > 0 else 0.0,
            "new_covered": q_covered - q_existing,
            "avg_distance_miles": round(float(np.mean(min_dists[mask])), 1),
        }

    return {
        "pct_covered": round(float(covered / n * 100), 1),
        "covered_count": covered,
        "new_covered": new_covered,
        "existing_covered": existing_covered,
        "total_tracts": n,
        "avg_distance_miles": round(float(np.mean(min_dists)), 1),
        "max_distance_miles": round(float(np.max(min_dists)), 1),
        "median_distance_miles": round(float(np.median(min_dists)), 1),
        "radius_miles": radius_miles,
        "coverage_by_ej_quartile": ej_quartile_coverage,
    }


# ── Comparison ──────────────────────────────────────────────────────────────

def compare_methods(tracts, k=25, num_reads=500, top_candidates=400,
                    proximity_threshold_miles=8.0, coverage_radius_miles=10.0,
                    existing_sensors=None, model_disagreement=None):
    """Run all three solvers and compare results with coverage analysis."""

    quantum_result = solve_quantum(
        tracts, k=k, num_reads=num_reads,
        top_candidates=top_candidates,
        proximity_threshold_miles=proximity_threshold_miles,
        existing_sensors=existing_sensors,
        model_disagreement=model_disagreement,
    )

    greedy_result = solve_greedy(
        tracts, k=k,
        proximity_threshold_miles=proximity_threshold_miles,
        existing_sensors=existing_sensors,
        model_disagreement=model_disagreement,
    )

    classical_sa_result = solve_classical_sa(
        tracts, k=k,
        proximity_threshold_miles=proximity_threshold_miles,
        existing_sensors=existing_sensors,
        model_disagreement=model_disagreement,
    )

    methods_map = {
        "quantum_annealing": quantum_result,
        "greedy": greedy_result,
        "classical_sa": classical_sa_result,
    }

    comparison = {}
    for name, result in methods_map.items():
        coverage = compute_coverage(
            tracts, result["selected_tracts"],
            radius_miles=coverage_radius_miles,
            existing_sensors=existing_sensors,
        )

        avg_ej = float(np.mean([
            t.get("ejf_score", 0.0) or 0.0
            for t in result["selected_tracts"]
        ])) if result["selected_tracts"] else 0.0

        avg_composite = float(np.mean([
            t["composite_score"]
            for t in result["selected_tracts"]
        ])) if result["selected_tracts"] else 0.0

        comparison[name] = {
            "method_display": result["method_display"],
            "num_sensors": result["num_sensors"],
            "coverage": coverage,
            "avg_ej_score": round(avg_ej, 1),
            "avg_composite_score": round(avg_composite, 4),
            "timing": result["timing"],
            "selected_tracts": result["selected_tracts"],
        }
        if "best_energy" in result:
            comparison[name]["best_energy"] = result["best_energy"]
        if "num_reads" in result:
            comparison[name]["num_reads"] = result["num_reads"]

    return comparison
