import numpy as np
import pandas as pd
import time
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Tuple

# ==========================================
# 1. CONFIGURATION & DESIGN SPACE
# ==========================================

@dataclass
class DesignSpace:
    # 77 Variables total based on the paper
    # Ranges from Table 2
    TA_range: Tuple[float, float] = (90, 150)       # 36 variables
    AA_range: Tuple[float, float] = (270, 330)      # 36 variables
    SWD_range: Tuple[float, float] = (0, 0.5)       # 1 variable
    WW_Left_range: Tuple[float, float] = (6.5, 9.9) # 1 variable
    WW_Right_range: Tuple[float, float] = (0.1, 3.5)# 1 variable
    HW_Down_range: Tuple[float, float] = (2, 2.95)  # 1 variable
    HW_Up_range: Tuple[float, float] = (0.05, 1)    # 1 variable

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns Lower Bound (LB) and Upper Bound (UB) arrays for all 77 variables."""
        lbs = []
        ubs = []

        # 36 Tilt Angles
        lbs.extend([self.TA_range[0]] * 36)
        ubs.extend([self.TA_range[1]] * 36)

        # 36 Azimuth Angles
        lbs.extend([self.AA_range[0]] * 36)
        ubs.extend([self.AA_range[1]] * 36)

        # 1 SWD
        lbs.append(self.SWD_range[0])
        ubs.append(self.SWD_range[1])

        # 4 Window Dimensions
        lbs.append(self.WW_Left_range[0]); ubs.append(self.WW_Left_range[1])
        lbs.append(self.WW_Right_range[0]); ubs.append(self.WW_Right_range[1])
        lbs.append(self.HW_Down_range[0]); ubs.append(self.HW_Down_range[1])
        lbs.append(self.HW_Up_range[0]); ubs.append(self.HW_Up_range[1])

        return np.array(lbs), np.array(ubs)

# ==========================================
# 2. SIMULATION INTERFACE (Placeholder)
# ==========================================

class SimulationEngine:
    """
    Connects Python to EnergyPlus and Radiance.
    Currently uses mathematical functions to simulate output for demonstration.
    """

    @staticmethod
    def evaluate(x: np.ndarray) -> List[float]:
        """
        Input: x (vector of 77 design variables)
        Output: [EP, EC, sDA, DGP]
        """
        # --- PLACEHOLDER LOGIC (Replace with actual EnergyPlus calls) ---

        # Extract mean values for simple calculation proxy
        mean_TA = np.mean(x[0:36])
        mean_AA = np.mean(x[36:72])
        swd = x[72]
        w_area = (x[73] + x[74]) * (x[75] + x[76])

        # 1. EP (Electricity Production) - Maximize
        # Dependent on Tilt Angle (TA) and Azimuth (AA)
        ep_base = 5000
        # Optimal TA approx 120-130, Optimal AA approx 300
        ep = ep_base + (150 - abs(mean_TA - 130) * 10) + (100 - abs(mean_AA - 300) * 5)

        # 2. EC (Electricity Consumption) - Minimize
        # Larger windows -> more heat gain -> more cooling load
        # Shading (SWD) reduces load
        ec_base = 1500
        ec = ec_base + (w_area * 15) - (swd * 80)

        # 3. sDA (Spatial Daylight Autonomy) - Maximize
        # Larger windows and larger SWD (less blocking) -> Higher sDA
        sda = 40 + (w_area * 1.2) + (swd * 30)
        sda = min(95, sda) # Cap at 95%

        # 4. DGP (Daylight Glare Probability) - Minimize
        # Larger windows -> Higher Glare
        # Proper shading angles reduce glare
        dgp = 0.2 + (w_area * 0.008) - (swd * 0.02)
        dgp = max(0.15, dgp)

        return [ep, ec, sda, dgp]

# ==========================================
# 3. MULTI-OBJECTIVE OPTIMIZER BASE
# ==========================================

class MOO_Base:
    def __init__(self, design_space: DesignSpace, pop_size=30, max_iter=20, archive_size=100):
        self.lb, self.ub = design_space.get_bounds()
        self.dim = len(self.lb)
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.archive_size = archive_size
        self.archive_pos = []
        self.archive_objs = []

        # Objectives: [Maximize EP, Minimize EC, Maximize sDA, Minimize DGP]
        # Internal Conversion: [Min(-EP), Min(EC), Min(-sDA), Min(DGP)]
        self.obj_signs = np.array([-1, 1, -1, 1])

    def initialize_population(self):
        population = np.zeros((self.pop_size, self.dim))
        for i in range(self.dim):
            population[:, i] = np.random.uniform(self.lb[i], self.ub[i], self.pop_size)
        return population

    def evaluate_population(self, population):
        objs = []
        for ind in population:
            raw_res = SimulationEngine.evaluate(ind)
            internal_res = [raw_res[i] * self.obj_signs[i] for i in range(4)]
            objs.append(internal_res)
        return np.array(objs)

    def dominates(self, obj1, obj2):
        return np.all(obj1 <= obj2) and np.any(obj1 < obj2)

    def update_archive(self, pop, pop_objs):
        if len(self.archive_pos) == 0:
            combined_pos = pop
            combined_objs = pop_objs
        else:
            combined_pos = np.vstack((self.archive_pos, pop))
            combined_objs = np.vstack((self.archive_objs, pop_objs))

        non_dominated_indices = []
        for i in range(len(combined_objs)):
            is_dominated = False
            for j in range(len(combined_objs)):
                if i != j and self.dominates(combined_objs[j], combined_objs[i]):
                    is_dominated = True
                    break
            if not is_dominated:
                non_dominated_indices.append(i)

        if len(non_dominated_indices) > 0:
            self.archive_pos = combined_pos[non_dominated_indices]
            self.archive_objs = combined_objs[non_dominated_indices]

            # Pruning if archive is too large
            if len(self.archive_pos) > self.archive_size:
                # Remove duplicates first
                _, unique_indices = np.unique(self.archive_objs, axis=0, return_index=True)
                if len(unique_indices) < len(self.archive_pos):
                     self.archive_pos = self.archive_pos[unique_indices]
                     self.archive_objs = self.archive_objs[unique_indices]

                # Random pruning if still too large
                if len(self.archive_pos) > self.archive_size:
                    keep_idx = np.random.choice(len(self.archive_pos), self.archive_size, replace=False)
                    self.archive_pos = self.archive_pos[keep_idx]
                    self.archive_objs = self.archive_objs[keep_idx]

    def get_best_solution_roulette(self):
        if len(self.archive_pos) == 0:
            return np.random.uniform(self.lb, self.ub)
        return self.archive_pos[np.random.randint(0, len(self.archive_pos))]

    def run(self):
        raise NotImplementedError

# ==========================================
# 4. ALGORITHMS IMPLEMENTATION
# ==========================================

class MOGWO(MOO_Base):
    """Multi-Objective Grey Wolf Optimizer"""
    def run(self):
        wolves = self.initialize_population()
        start_time = time.time()

        for it in range(self.max_iter):
            objs = self.evaluate_population(wolves)
            self.update_archive(wolves, objs)

            # Select Leaders
            if len(self.archive_pos) >= 3:
                # Ideally use grid mechanism, simplified here to random top
                indices = np.random.choice(len(self.archive_pos), 3, replace=False if len(self.archive_pos)>=3 else True)
                alpha, beta, delta = self.archive_pos[indices[0]], self.archive_pos[indices[1]], self.archive_pos[indices[2]]
            elif len(self.archive_pos) > 0:
                alpha = beta = delta = self.archive_pos[0]
            else:
                alpha = beta = delta = wolves[0]

            a = 2 - it * (2 / self.max_iter) # Eq 12a

            for i in range(self.pop_size):
                for j in range(self.dim):
                    r1, r2 = np.random.random(), np.random.random()
                    A1, C1 = 2*a*r1 - a, 2*r2
                    D_alpha = abs(C1 * alpha[j] - wolves[i, j])
                    X1 = alpha[j] - A1 * D_alpha

                    r1, r2 = np.random.random(), np.random.random()
                    A2, C2 = 2*a*r1 - a, 2*r2
                    D_beta = abs(C2 * beta[j] - wolves[i, j])
                    X2 = beta[j] - A2 * D_beta

                    r1, r2 = np.random.random(), np.random.random()
                    A3, C3 = 2*a*r1 - a, 2*r2
                    D_delta = abs(C3 * delta[j] - wolves[i, j])
                    X3 = delta[j] - A3 * D_delta

                    wolves[i, j] = (X1 + X2 + X3) / 3

                wolves[i] = np.clip(wolves[i], self.lb, self.ub)

        return self.archive_pos, self.archive_objs, time.time() - start_time

class MOWOA(MOO_Base):
    """Multi-Objective Whale Optimization Algorithm"""
    def run(self):
        whales = self.initialize_population()
        start_time = time.time()

        for it in range(self.max_iter):
            objs = self.evaluate_population(whales)
            self.update_archive(whales, objs)

            a = 2 - it * (2 / self.max_iter)

            for i in range(self.pop_size):
                r = np.random.random()
                A = 2 * a * r - a
                C = 2 * np.random.random()
                l = np.random.uniform(-1, 1)
                p = np.random.random()

                target = self.get_best_solution_roulette()

                if p < 0.5:
                    if abs(A) < 1:
                        D = abs(C * target - whales[i])
                        whales[i] = target - A * D
                    else:
                        rand_idx = np.random.randint(0, self.pop_size)
                        X_rand = whales[rand_idx]
                        D = abs(C * X_rand - whales[i])
                        whales[i] = X_rand - A * D
                else:
                    dist = abs(target - whales[i])
                    whales[i] = dist * np.exp(1 * l) * np.cos(2 * np.pi * l) + target

                whales[i] = np.clip(whales[i], self.lb, self.ub)

        return self.archive_pos, self.archive_objs, time.time() - start_time

class MOMFO(MOO_Base):
    """Multi-Objective Moth-Flame Optimization"""
    def run(self):
        moths = self.initialize_population()
        start_time = time.time()

        for it in range(self.max_iter):
            objs = self.evaluate_population(moths)
            self.update_archive(moths, objs)

            flame_no = int(np.ceil(self.archive_size - (it + 1) * ((self.archive_size - 1) / self.max_iter)))
            flames = self.archive_pos if len(self.archive_pos) > 0 else moths
            if len(flames) > flame_no: flames = flames[:flame_no]

            r = -1 + it * ((-1) / self.max_iter)

            for i in range(self.pop_size):
                for j in range(self.dim):
                    flame_idx = i if i < len(flames) else len(flames) - 1
                    dist = abs(flames[flame_idx][j] - moths[i, j])
                    t = (r - 1) * np.random.random() + 1
                    moths[i, j] = dist * np.exp(1 * t) * np.cos(2 * np.pi * t) + flames[flame_idx][j]

            moths = np.clip(moths, self.lb, self.ub)

        return self.archive_pos, self.archive_objs, time.time() - start_time

class MOACO(MOO_Base):
    """Multi-Objective Ant Colony Optimization"""
    def run(self):
        ants = self.initialize_population()
        start_time = time.time()

        for it in range(self.max_iter):
            objs = self.evaluate_population(ants)
            self.update_archive(ants, objs)

            for i in range(self.pop_size):
                guide = self.get_best_solution_roulette()
                for j in range(self.dim):
                    # Gaussian kernel sampling around guide
                    sigma = (self.ub[j] - self.lb[j]) * (0.5 / (it + 1))
                    ants[i, j] = np.random.normal(guide[j], sigma)

            ants = np.clip(ants, self.lb, self.ub)

        return self.archive_pos, self.archive_objs, time.time() - start_time

# ==========================================
# 5. METRICS
# ==========================================

class Metrics:
    @staticmethod
    def calculate_MID(front, ideal_point):
        """Mean Ideal Distance"""
        if len(front) == 0: return float('inf')
        dists = np.sqrt(np.sum((front - ideal_point)**2, axis=1))
        return np.mean(dists)

    @staticmethod
    def calculate_SP(front):
        """Spacing"""
        if len(front) < 2: return 0.0
        dists = []
        for i in range(len(front)):
            # Manhattan distance to nearest neighbor
            diffs = np.abs(front - front[i])
            manhattan = np.sum(diffs, axis=1)
            manhattan[i] = float('inf') # Ignore self
            dists.append(np.min(manhattan))

        d_mean = np.mean(dists)
        return np.sqrt(np.mean((np.array(dists) - d_mean)**2))

    @staticmethod
    def calculate_MS(front):
        """Maximum Spread"""
        if len(front) == 0: return 0.0
        mins = np.min(front, axis=0)
        maxs = np.max(front, axis=0)
        return np.sqrt(np.sum((maxs - mins)**2))

    @staticmethod
    def calculate_GD(front, ref_front):
        """Generational Distance"""
        if len(front) == 0: return float('inf')
        sum_sq = 0
        for sol in front:
            dists = np.sqrt(np.sum((ref_front - sol)**2, axis=1))
            sum_sq += np.min(dists)**2
        return np.sqrt(sum_sq) / len(front)

    @staticmethod
    def calculate_IGD(front, ref_front):
        """Inverted Generational Distance"""
        if len(ref_front) == 0: return 0.0
        sum_sq = 0
        for ref in ref_front:
            dists = np.sqrt(np.sum((front - ref)**2, axis=1))
            sum_sq += np.min(dists)**2
        return np.sqrt(sum_sq) / len(ref_front)

# ==========================================
# 6. MCDM (CORRECTED - ROBUST)
# ==========================================

class MCDM:
    """Handles Shannon Entropy and TOPSIS Ranking"""

    @staticmethod
    def shannon_entropy_weights(decision_matrix):
        """Robust Shannon Entropy Calculation."""
        rows, cols = decision_matrix.shape
        if rows <= 1: return np.ones(cols) / cols

        # 1. Normalize (P_ij)
        col_sums = np.sum(decision_matrix, axis=0)

        # FIX: Handle zero sum columns
        col_sums[col_sums == 0] = 1

        p_matrix = decision_matrix / col_sums

        k = 1 / np.log(rows)
        E = []

        for j in range(cols):
            sum_pln = 0
            for i in range(rows):
                p = p_matrix[i, j]
                # Avoid log(0)
                if p > 0:
                    sum_pln += p * np.log(p)
            E.append(-k * sum_pln)

        E = np.array(E)
        d = 1 - E # Divergence

        # FIX: Handle case where all criteria have 0 divergence (all values identical)
        total_d = np.sum(d)
        if total_d == 0:
            return np.ones(cols) / cols

        return d / total_d

    @staticmethod
    def topsis_rank(decision_matrix, weights, criteria_signs):
        """Robust TOPSIS Ranking."""
        # 1. Vector Normalization
        denom = np.sqrt(np.sum(decision_matrix**2, axis=0))

        # FIX: Avoid division by zero if a column is all zeros
        denom[denom == 0] = 1

        norm_matrix = decision_matrix / denom

        # 2. Weighted Matrix
        weighted_matrix = norm_matrix * weights

        # 3. Ideal (V+) and Negative Ideal (V-)
        v_pos = []
        v_neg = []

        for j in range(len(criteria_signs)):
            col = weighted_matrix[:, j]
            if criteria_signs[j] == 1: # Benefit
                v_pos.append(np.max(col))
                v_neg.append(np.min(col))
            else: # Cost
                v_pos.append(np.min(col))
                v_neg.append(np.max(col))

        v_pos = np.array(v_pos)
        v_neg = np.array(v_neg)

        # 4. Distances
        s_pos = np.sqrt(np.sum((weighted_matrix - v_pos)**2, axis=1))
        s_neg = np.sqrt(np.sum((weighted_matrix - v_neg)**2, axis=1))

        # 5. Closeness Coefficient
        # FIX: Avoid division by zero if s_pos + s_neg == 0
        total_dist = s_pos + s_neg
        total_dist[total_dist == 0] = 1

        cc = s_neg / total_dist
        return cc

# ==========================================
# 7. MAIN
# ==========================================

def main():
    print("-- - Adaptive BIPV Optimization (Simulation) ---")

    ds = DesignSpace()
    algos = {
        "MOGWO": MOGWO(ds),
        "MOWOA": MOWOA(ds),
        "MOACO": MOACO(ds),
        "MOMFO": MOMFO(ds)
    }

    results = {}
    all_fronts = []

    # 1. Run Optimization
    for name, algo in algos.items():
        print(f"Running {name}...")
        front_pos, front_objs, runtime = algo.run()

        # Convert internal min(-EP) back to EP for reporting
        report_objs = deepcopy(front_objs)
        if len(report_objs) > 0:
            report_objs[:, 0] *= -1 # EP
            report_objs[:, 2] *= -1 # sDA

        results[name] = {"front": report_objs, "time": runtime}
        if len(report_objs) > 0:
            all_fronts.extend(report_objs)

    # 2. Create Reference Front
    # Filter for non-dominated from all algorithms
    all_fronts = np.array(all_fronts)
    if len(all_fronts) > 0:
        # Simple non-dominated filter for ref set
        is_dominated = np.zeros(len(all_fronts), dtype=bool)
        # Note: direction is [Max, Min, Max, Min].
        # For simplicity in metrics, we assume user converts or handles logic.
        # Here we just use the set as is for GD/IGD to avoid zero-error if identical.
        ref_set = all_fronts
    else:
        ref_set = np.zeros((1, 4))

    # 3. Calculate Metrics
    # Metrics: GD(min), IGD(min), NPS(max), MID(min), SP(min), MS(max), Time(min), Q(max)
    metric_data = []

    # Approx Ideal Point for MID: [Max EP, Min EC, Max sDA, Min DGP]
    ideal_point = np.array([7000, 1400, 100, 0.15])

    for name in algos.keys():
        front = results[name]["front"]
        if len(front) == 0:
            metric_data.append([1e9]*8)
            continue

        gd = Metrics.calculate_GD(front, ref_set)
        igd = Metrics.calculate_IGD(front, ref_set)
        nps = len(front)
        mid = Metrics.calculate_MID(front, ideal_point)
        sp = Metrics.calculate_SP(front)
        ms = Metrics.calculate_MS(front)
        t = results[name]["time"]
        q = 2.0 # Placeholder constant from paper example

        metric_data.append([gd, igd, nps, mid, sp, ms, t, q])

    metrics_df = pd.DataFrame(metric_data, index=algos.keys(),
                              columns=["GD", "IGD", "NPS", "MID", "SP", "MS", "Time", "Q"])

    print("\n-- - Performance Metrics ---")
    print(metrics_df)

    # 4. Shannon Entropy & TOPSIS
    matrix = metrics_df.values

    # Signs: -1 for cost (Min), 1 for benefit (Max)
    # GD(-), IGD(-), NPS(+), MID(-), SP(-), MS(+), Time(-), Q(+) (Assumption on Q being positive)
    signs = [-1, -1, 1, -1, -1, 1, -1, 1]

    weights = MCDM.shannon_entropy_weights(matrix)
    print("\n-- - Entropy Weights ---")
    print(weights)

    scores = MCDM.topsis_rank(matrix, weights, signs)

    rank_df = pd.DataFrame({"TOPSIS Score": scores}, index=algos.keys())
    rank_df = rank_df.sort_values("TOPSIS Score", ascending=False)

    print("\n-- - Final Ranking ---")
    print(rank_df)

if __name__ == "__main__":
    main()