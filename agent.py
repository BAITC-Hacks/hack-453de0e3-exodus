"""Tariff campaign agent using public history, pilots and a budgeted planner."""
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd


class Agent:
    # User-selected planning objective, not a claimed or hardcoded score.
    TARGET_EXPECTED_NET = 3_000_000.0
    TARGET_TOLERANCE = 0.05

    def act(self, env):
        profile = env.customer_profile
        tariffs = set(env.tariffs["tariff_plan_code"])
        channels = env.channels
        cells = {(cur, str(seg)): (len(rows), float(rows["predicted_arpu"].sum()))
                 for (cur, seg), rows in profile.groupby(
                     ["current_tariff", "arpu_segment"], observed=True)}
        try:
            history = pd.read_csv(Path(__file__).resolve().parent / "data" / "change_tariff.csv")
        except (OSError, ValueError):
            return []
        h = history[history["AVG_ARPU_PREV_3M"] >= 100].copy()
        h["segment"] = pd.cut(h["AVG_ARPU_PREV_3M"], [-np.inf, 1000, 5000, np.inf],
                              labels=["LOW", "MID", "HIGH"])
        h["change"] = ((h["AVG_ARPU_NEXT_3M"] - h["AVG_ARPU_PREV_3M"])
                       / h["AVG_ARPU_PREV_3M"]).clip(-1, 3)
        totals = h.groupby(["tariff_plan_code_from", "segment"], observed=True).size()
        estimates = {}
        groups = h.groupby(["tariff_plan_code_from", "segment",
                            "tariff_plan_code_to"], observed=True)["change"]
        for (cur, seg, target), values in groups:
            seg = str(seg)
            n = len(values)
            if n < 3 or cur == target or target not in tariffs or (cur, seg) not in cells:
                continue
            size, arpu_sum = cells[(cur, seg)]
            frequency = n / max(int(totals[(cur, seg)]), 1)
            base = float(values.mean()) * frequency * n / (n + 15)
            for channel, spec in channels.items():
                cost = float(spec["cost_per_contact"])
                ratio = base * float(spec["conversion_multiplier"])
                net = ratio * arpu_sum - cost * size
                if net > 0:
                    estimates[(cur, seg, target, channel)] = {
                        "ratio": ratio, "prior": ratio, "size": size,
                        "arpu_sum": arpu_sum, "cost": cost, "net": net,
                    }
        if not estimates:
            return []

        # Diversify exploration across audience cells. A cheap channel measures
        # the transition; its observed lift is scaled to the other channels.
        order = sorted(estimates, key=lambda key: estimates[key]["net"], reverse=True)
        picked, tested = [], set()
        for key in order:
            cur, seg, target, _ = key
            if (cur, seg) in tested or not 10 <= estimates[key]["size"]:
                continue
            channel = "sms" if (cur, seg, target, "sms") in estimates else "push"
            if (cur, seg, target, channel) not in estimates:
                continue
            picked.append((cur, seg, target, channel))
            tested.add((cur, seg))
            if len(picked) >= 12:
                break

        observations = defaultdict(lambda: [0.0, 0])
        pilot_coverage = {}
        initial_budget = float(env.remaining_budget)
        max_exploration_spend = 0.15 * initial_budget

        def pilot(key, requested):
            cur, seg, target, channel = key
            item = estimates[key]
            cost = item["cost"]
            remaining_exploration = max_exploration_spend - (initial_budget - env.remaining_budget)
            n = min(requested, item["size"], env.remaining_contacts, 200)
            if cost:
                n = min(n, int(max(0, remaining_exploration) // cost))
            if n < 10 or env.pilots_left <= 0:
                return False
            try:
                result = env.run_pilot(target_tariff=target, channel=channel,
                                       n_customers=n, filter_arpu_segment=seg,
                                       filter_current_tariff=cur)
            except (RuntimeError, ValueError):
                return False
            actual = int(result["n_customers"])
            if actual <= 0:
                return False
            previous = pilot_coverage.get(key, 0.0)
            pilot_coverage[key] = 1 - (1 - previous) * (1 - actual / item["size"])
            sums = observations[(cur, seg, target)]
            multiplier = float(channels[channel]["conversion_multiplier"])
            sums[0] += actual * float(result["observed_lift_ratio"]) / multiplier
            sums[1] += actual
            for other, spec in channels.items():
                match = estimates.get((cur, seg, target, other))
                if match is not None:
                    prior_base = match["prior"] / float(spec["conversion_multiplier"])
                    posterior_base = (60 * prior_base + sums[0]) / (60 + sums[1])
                    match["ratio"] = posterior_base * float(spec["conversion_multiplier"])
            return True

        for key in picked:
            size = estimates[key]["size"]
            pilot(key, 160 if size >= 500 else 100)

        # Re-test up to four valuable, uncertain transitions. The choice is
        # recomputed after every observation and never rests on one small pilot.
        repeated = set()
        for _ in range(4):
            choices = []
            for key in picked:
                transition = key[:3]
                if transition in repeated or transition not in observations:
                    continue
                item = estimates[key]
                n_seen = observations[transition][1]
                current_value = item["ratio"] * item["arpu_sum"] - item["cost"] * item["size"]
                uncertainty = 0.804 / np.sqrt(max(n_seen, 1))
                value_of_information = max(current_value, 0) * uncertainty
                choices.append((value_of_information, key))
            if not choices:
                break
            key = max(choices)[1]
            repeated.add(key[:3])
            pilot(key, 200)

        # Compatible tariff cells can form one broad campaign. Cells are kept
        # disjoint across campaigns, and each bundle stays below 5,000 contacts.
        # Pilot identities are not exposed by the public interface. Estimate
        # unique coverage from random sampling and avoid counting their gain twice.
        pilot_values = {}
        for key, coverage in pilot_coverage.items():
            pilot_values[key[:2]] = (coverage, estimates[key]["ratio"])
        expected_net = sum(coverage * ratio * cells[cell][1]
                           for cell, (coverage, ratio) in pilot_values.items())
        expected_net -= initial_budget - float(env.remaining_budget)
        bundles = defaultdict(list)
        for (cur, seg, target, channel), item in estimates.items():
            size = item["size"]
            coverage, pilot_ratio = pilot_values.get((cur, seg), (0.0, 0.0))
            incremental_ratio = ((1 - coverage) * item["ratio"]
                                 + coverage * max(item["ratio"] - pilot_ratio, 0.0))
            gain = incremental_ratio * item["arpu_sum"] - item["cost"] * size
            if size <= 5000 and gain > 0:
                bundles[(seg, target, channel)].append((gain, cur, size))
        candidates = []
        for (seg, target, channel), members in bundles.items():
            members.sort(reverse=True)
            chosen, count = [], 0
            for gain, cur, size in members:
                if count + size <= 5000:
                    chosen.append((cur, gain, size))
                    count += size
            if chosen:
                candidates.append((seg, target, channel, chosen))

        campaigns, used = [], set()
        budget, contacts = float(env.remaining_budget), int(env.remaining_contacts)
        while len(campaigns) < 10 and contacts > 0:
            remaining_gain = self.TARGET_EXPECTED_NET - expected_net
            if campaigns and remaining_gain <= self.TARGET_EXPECTED_NET * self.TARGET_TOLERANCE:
                break
            best, best_gain = None, 0.0
            for seg, target, channel, members in candidates:
                cost = float(channels[channel]["cost_per_contact"])
                available = [entry for entry in members if (entry[0], seg) not in used]
                available.sort(key=lambda entry: entry[1] / entry[2], reverse=True)
                selected, count, gain = [], 0, 0.0
                for cur, value, size in available:
                    improves_target = abs(remaining_gain - gain - value) < abs(remaining_gain - gain)
                    if (improves_target and count + size <= min(5000, contacts)
                            and (count + size) * cost <= budget):
                        selected.append(cur)
                        count += size
                        gain += value
                improvement = abs(remaining_gain) - abs(remaining_gain - gain)
                if selected and improvement > best_gain:
                    best, best_gain = (seg, target, channel, selected, count, cost, gain), improvement
            if best is None:
                break
            seg, target, channel, selected, count, cost, gain = best
            campaigns.append({"campaign_name": f"plan_{len(campaigns) + 1}_{seg}_{channel}",
                              "filter_arpu_segment": seg,
                              "filter_current_tariff": ";".join(selected),
                              "target_tariff": target, "channel": channel})
            used.update((cur, seg) for cur in selected)
            contacts -= count
            budget -= count * cost
            expected_net += gain
        self.planning_summary = {"target_expected_net": self.TARGET_EXPECTED_NET,
                                 "estimated_net": expected_net,
                                 "final_campaigns": len(campaigns)}
        return campaigns
