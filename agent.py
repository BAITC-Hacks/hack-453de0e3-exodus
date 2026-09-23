"""Adaptive tariff campaign agent for the public Beeline case package.

The history is used only to choose hypotheses. Campaign decisions are made from
pilot observations on the current audience. The agent does not inspect the
environment's hidden impact model.
"""

from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd


PILOT_STD = 0.804
MAX_CAMPAIGNS = 10
MAX_CAMPAIGN_SIZE = 5000
FIRST_PILOTS = 12
FIRST_SAMPLE = 150
CONFIRM_PILOTS = 4
CONFIRM_SAMPLE = 200
PILOT_BUDGET_FRACTION = 0.25
MAX_SINGLE_CAMPAIGN_BUDGET_FRACTION = 0.30
UNCONFIRMED_PAID_LIMIT_FRACTION = 0.10


class Agent:
    @staticmethod
    def _history():
        # The archive may be launched from any current working directory.
        public_paths = (
            Path(__file__).resolve().parent / "data" / "change_tariff.csv",
            Path.cwd() / "data" / "change_tariff.csv",
        )
        history_path = next((path for path in public_paths if path.is_file()), None)
        if history_path is None:
            return {}
        try:
            history = pd.read_csv(history_path)
            history = history.loc[history["AVG_ARPU_PREV_3M"] >= 100].copy()
            history["segment"] = pd.cut(
                history["AVG_ARPU_PREV_3M"],
                [-np.inf, 1000, 5000, np.inf],
                labels=["LOW", "MID", "HIGH"],
            )
            history["lift"] = (
                (history["AVG_ARPU_NEXT_3M"] - history["AVG_ARPU_PREV_3M"])
                / history["AVG_ARPU_PREV_3M"]
            ).clip(-1, 3)
            summary = history.groupby(
                ["tariff_plan_code_from", "tariff_plan_code_to", "segment"],
                observed=True,
            )["lift"].agg(["mean", "count"])
            return {
                (str(source), str(target), str(segment)):
                (float(row["mean"]), int(row["count"]))
                for (source, target, segment), row in summary.iterrows()
            }
        except (OSError, KeyError, TypeError, ValueError, pd.errors.ParserError):
            return {}

    @staticmethod
    def _cohorts(profile):
        """Build disjoint cohorts that fit the campaign size cap."""
        result = []
        levels = [
            (["current_tariff", "arpu_segment"], []),
            (["current_tariff", "arpu_segment", "data_segment"],
             ["filter_data_segment"]),
            (["current_tariff", "arpu_segment", "data_segment", "call_segment"],
             ["filter_data_segment", "filter_call_segment"]),
        ]

        def add_group(group, values, depth):
            count = len(group)
            if count < 10:
                return
            if count > MAX_CAMPAIGN_SIZE:
                if depth + 1 >= len(levels):
                    return
                columns, _ = levels[depth + 1]
                extra_column = columns[-1]
                for extra, subgroup in group.groupby(extra_column, observed=True):
                    add_group(subgroup, {**values, extra_column: str(extra)}, depth + 1)
                return
            if count < FIRST_SAMPLE:
                return
            filters = {
                "filter_current_tariff": values["current_tariff"],
                "filter_arpu_segment": values["arpu_segment"],
            }
            if "data_segment" in values:
                filters["filter_data_segment"] = values["data_segment"]
            if "call_segment" in values:
                filters["filter_call_segment"] = values["call_segment"]
            result.append({
                "filters": filters,
                "ids": set(group["ID_NUMBER"].tolist()),
                "n": count,
                "arpu_sum": float(group["predicted_arpu"].sum()),
                "arpu_mean": float(group["predicted_arpu"].mean()),
            })

        for (source, segment), group in profile.groupby(
            ["current_tariff", "arpu_segment"], observed=True
        ):
            add_group(group, {"current_tariff": str(source),
                              "arpu_segment": str(segment)}, 0)
        return result

    @staticmethod
    def _candidates(env, cohorts, history):
        tariff_table = env.tariffs.set_index("tariff_plan_code")
        prices = pd.to_numeric(tariff_table["price_tariff"], errors="coerce").to_dict()
        candidates = []
        for cohort in cohorts:
            source = cohort["filters"]["filter_current_tariff"]
            segment = cohort["filters"]["filter_arpu_segment"]
            source_price = prices.get(source, float("nan"))
            options = []
            for target, target_price in prices.items():
                target = str(target)
                if target == source or not math.isfinite(target_price):
                    continue
                if math.isfinite(source_price) and source_price > 0:
                    price_gain = (target_price - source_price) / source_price
                    if price_gain < -0.10 or price_gain > 2.5:
                        continue
                else:
                    price_gain = 0.25 if target_price > 0 else 0.0
                historic_lift, historic_n = history.get(
                    (source, target, segment), (0.0, 0)
                )
                # Sparse history is weak evidence. Price is only a fallback
                # hypothesis, not a predicted effect for the audience.
                shrink = historic_n / (historic_n + 30)
                prior_score = (0.7 * max(historic_lift, -0.2) * shrink
                               + 0.3 * min(max(price_gain, 0.0), 0.8))
                options.append((prior_score, target))
            if options:
                score, target = max(options)
                candidates.append({**cohort, "target": target,
                                   "priority": score * cohort["arpu_sum"]})
        candidates.sort(key=lambda c: c["priority"], reverse=True)
        return candidates

    @staticmethod
    def _pilot(env, candidate, channel, n, pilot_budget_left):
        if env.pilots_left <= 0 or env.remaining_contacts < 10:
            return None
        n = min(int(n), int(env.remaining_contacts), candidate["n"], 200)
        cost_per_contact = float(env.channels[channel]["cost_per_contact"])
        if cost_per_contact > 0:
            n = min(n, int(pilot_budget_left // cost_per_contact),
                    int(env.remaining_budget // cost_per_contact))
        if n < 10:
            return None
        try:
            result = env.run_pilot(
                target_tariff=candidate["target"],
                channel=channel,
                n_customers=n,
                **candidate["filters"],
            )
        except (RuntimeError, ValueError, KeyError, TypeError):
            return None
        actual_n = int(result.get("n_customers", 0))
        if actual_n < 10:
            return None
        ratio = float(result["observed_lift_ratio"])
        if not math.isfinite(ratio):
            return None
        return actual_n, ratio, channel, float(result["cost"])

    @staticmethod
    def _estimate(observations, channels):
        # Each observed lift already includes its channel's conversion factor.
        # Weight by inverse variance after putting channels on one base scale.
        information = 0.0
        weighted_lift = 0.0
        for n, observed, channel, _ in observations:
            multiplier = float(channels[channel]["conversion_multiplier"])
            information += n * multiplier * multiplier
            weighted_lift += n * multiplier * observed
        base_mean = weighted_lift / information
        base_se = PILOT_STD / math.sqrt(information)
        # Require the measured effect to exceed one standard error before
        # treating it as a positive launch signal.
        return base_mean - base_se

    @staticmethod
    def _channel_options(candidate, safe_ratio, channels):
        options = []
        if safe_ratio <= 0:
            return options
        for name, details in channels.items():
            multiplier = float(details["conversion_multiplier"])
            cost = candidate["n"] * float(details["cost_per_contact"])
            gross = candidate["arpu_sum"] * safe_ratio * multiplier
            net = gross - cost
            if net > 0:
                options.append({"channel": name, "cost": cost,
                                "gross": gross, "net": net})
        return options

    def act(self, env):
        profile = env.customer_profile
        cohorts = self._cohorts(profile)
        if not cohorts:
            return []
        history = self._history()
        candidates = self._candidates(env, cohorts, history)
        if not candidates:
            return []

        observations = {}
        initial_channels = ("push", "sms", "digital_ads")
        pilot_budget_left = min(
            float(env.remaining_budget),
            float(env.total_budget) * PILOT_BUDGET_FRACTION,
        )
        for index, candidate in enumerate(candidates[:FIRST_PILOTS]):
            channel = initial_channels[index % len(initial_channels)]
            sample = 100 if channel == "digital_ads" else FIRST_SAMPLE
            result = self._pilot(
                env, candidate, channel, sample, pilot_budget_left
            )
            if result is not None:
                observations[index] = [result]
                pilot_budget_left -= result[3]
        if not observations:
            return []

        # Confirm candidates with the largest estimated whole-cohort value.
        first_rank = []
        for index, items in observations.items():
            estimate = self._estimate(items, env.channels)
            options = self._channel_options(candidates[index], estimate, env.channels)
            if options:
                first_rank.append((max(o["net"] for o in options), index))
        first_rank.sort(reverse=True)
        call_piloted = False
        for _, index in first_rank[:min(CONFIRM_PILOTS, env.pilots_left)]:
            candidate = candidates[index]
            estimate = self._estimate(observations[index], env.channels)
            options = self._channel_options(candidate, estimate, env.channels)
            previous_channels = {item[2] for item in observations[index]}
            push_net = next((o["net"] for o in options if o["channel"] == "push"), 0.0)
            call_option = next((o for o in options if o["channel"] == "call"), None)

            if (not call_piloted and call_option is not None
                    and call_option["net"] > push_net
                    and pilot_budget_left >= 80 * float(env.channels["call"]["cost_per_contact"])):
                channel, sample = "call", 80
            else:
                other_options = [o for o in options
                                 if o["channel"] in {"sms", "digital_ads"}
                                 and o["channel"] not in previous_channels]
                other_options.sort(key=lambda o: o["net"], reverse=True)
                channel = other_options[0]["channel"] if other_options else "push"
                sample = (100 if channel == "digital_ads" else CONFIRM_SAMPLE)

            result = self._pilot(
                env, candidate, channel, sample, pilot_budget_left
            )
            if result is None and channel != "push":
                result = self._pilot(
                    env, candidate, "push", CONFIRM_SAMPLE, pilot_budget_left
                )
            if result is not None:
                observations[index].append(result)
                pilot_budget_left -= result[3]
                if result[2] == "call":
                    call_piloted = True

        measured = []
        for index, items in observations.items():
            candidate = candidates[index]
            estimate = self._estimate(items, env.channels)
            options = self._channel_options(candidate, estimate, env.channels)
            if options:
                candidate["pilot_count"] = len(items)
                measured.append((candidate, options))

        # Start with the free baseline, then spend only where a paid channel
        # adds more expected value than it costs. Cohorts are disjoint.
        measured.sort(key=lambda pair: max(o["net"] for o in pair[1]), reverse=True)
        selected = []
        contacts_left = int(env.remaining_contacts)
        used_ids = set()
        for candidate, options in measured:
            if len(selected) >= MAX_CAMPAIGNS:
                break
            if candidate["n"] > contacts_left or used_ids.intersection(candidate["ids"]):
                continue
            push = next((o for o in options if o["channel"] == "push"), None)
            if push is None:
                continue
            selected.append({"candidate": candidate, "choice": push,
                             "options": options})
            contacts_left -= candidate["n"]
            used_ids.update(candidate["ids"])

        budget_left = float(env.remaining_budget)
        while True:
            improvements = []
            for item in selected:
                current = item["choice"]
                for option in item["options"]:
                    if option["cost"] > (float(env.total_budget)
                                         * MAX_SINGLE_CAMPAIGN_BUDGET_FRACTION):
                        continue
                    if (item["candidate"]["pilot_count"] < 2
                            and option["cost"] > (float(env.total_budget)
                                                  * UNCONFIRMED_PAID_LIMIT_FRACTION)):
                        continue
                    extra_cost = option["cost"] - current["cost"]
                    extra_net = option["net"] - current["net"]
                    if extra_cost > 0 and extra_cost <= budget_left and extra_net > 0:
                        improvements.append((extra_net / extra_cost, extra_net,
                                             item, option, extra_cost))
            if not improvements:
                break
            _, _, item, option, extra_cost = max(
                improvements, key=lambda row: (row[0], row[1])
            )
            item["choice"] = option
            budget_left -= extra_cost

        campaigns = []
        for item in selected:
            candidate = item["candidate"]
            campaigns.append({
                "campaign_name": (
                    f"adaptive_{candidate['filters']['filter_current_tariff']}_"
                    f"{candidate['target']}_"
                    f"{candidate['filters']['filter_arpu_segment']}"
                ),
                **candidate["filters"],
                "target_tariff": candidate["target"],
                "channel": item["choice"]["channel"],
            })
        return campaigns
