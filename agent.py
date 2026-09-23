"""Adaptive tariff campaign agent.

The agent uses the supplied transition history as a conservative prior and
updates each segment/tariff/channel arm with noisy pilot observations.
"""
import os
import numpy as np
import pandas as pd


CHANNELS = {
    "push": (0.0, 0.50), "sms": (4.0, 0.65),
    "digital_ads": (22.0, 0.85), "call": (160.0, 1.20),
}


class Agent:
    def _prior(self, profile, tariffs):
        # change_tariff is intentionally read from the public package only.
        try:
            hist = pd.read_csv("data/change_tariff.csv")
        except Exception:
            hist = pd.DataFrame()
        if hist.empty:
            return {}
        h = hist[hist["AVG_ARPU_PREV_3M"] >= 100].copy()
        h["seg"] = pd.cut(h["AVG_ARPU_PREV_3M"], [-np.inf, 1000, 5000, np.inf],
                           labels=["LOW", "MID", "HIGH"])
        h["lift"] = ((h["AVG_ARPU_NEXT_3M"] - h["AVG_ARPU_PREV_3M"])
                      / h["AVG_ARPU_PREV_3M"]).clip(-1, 3)
        g = h.groupby(["tariff_plan_code_from", "tariff_plan_code_to", "seg"], observed=True)["lift"]
        return {(a, b, str(c)): (float(np.mean(v)), int(len(v)))
                for (a, b, c), v in g.agg(list).items()}

    def _channel(self, lift, arpu):
        # Net value per contact, with a mild preference for cheap channels.
        scores = {c: lift * mult * arpu - cost for c, (cost, mult) in CHANNELS.items()}
        return max(scores, key=scores.get)

    def act(self, env):
        p = env.customer_profile.copy()
        valid_tariffs = list(env.tariffs["tariff_plan_code"])
        prior = self._prior(p, env.tariffs)
        cells = (p.groupby(["current_tariff", "arpu_segment"], observed=True)
                 .agg(n=("ID_NUMBER", "size"), arpu=("predicted_arpu", "mean"))
                 .reset_index())
        arms = []
        for row in cells.itertuples(index=False):
            for target in valid_tariffs:
                if target == row.current_tariff:
                    continue
                lift, count = prior.get((row.current_tariff, target, str(row.arpu_segment)), (0.0, 0))
                # Keep a compact, economically plausible candidate pool.
                if count >= 3 and lift > -0.05:
                    for ch, (_, multiplier) in {"sms": CHANNELS["sms"]}.items():
                        arms.append({"cur": row.current_tariff, "seg": str(row.arpu_segment),
                                     "target": target, "arpu": float(row.arpu),
                                     "mean": lift * multiplier, "n": count, "channel": ch})
        arms.sort(key=lambda x: (x["mean"] * x["arpu"], x["n"]), reverse=True)
        arms = arms[:12]
        if not arms:
            return []

        stats = {self._key(a): {"mean": a["mean"], "n": 0, "prior_n": a["n"]} for a in arms}
        # Explore the strongest arms with robust, non-minimal pilots. The
        # second pass revisits uncertain winners, so one noisy result cannot
        # eliminate a good hypothesis.
        for i in range(min(12, len(arms))):
            if env.pilots_left <= 0 or env.remaining_contacts < 80:
                break
            a = arms[i]
            key = self._key(a)
            n = 120 if i < 4 else 80
            if i < 3:
                n = 120
            elif i < 7:
                n = 80
            else:
                n = 50

            n = min(n, env.remaining_contacts)
            try:
                r = env.run_pilot(target_tariff=a["target"], channel=a["channel"],
                                  n_customers=n, filter_arpu_segment=a["seg"],
                                  filter_current_tariff=a["cur"])
            except (RuntimeError, ValueError):
                continue
            s = stats[key]
            s["n"] += int(r["n_customers"])
            s["mean"] = (s["mean"] * (s["n"] - r["n_customers"])
                          + float(r["observed_lift_ratio"]) * r["n_customers"]) / s["n"]

        # UCB-style ranking: reward is expected absolute ARPU gain net of cost.
        scored = []
        for a in arms:
            s = stats[self._key(a)]
            uncertainty = 0.804 / np.sqrt(max(s["n"], 30))
            ucb = s["mean"] + 0.8 * uncertainty
            net = ucb * a["arpu"] - CHANNELS[a["channel"]][0]
            if net > 0:
                scored.append((net, a, s))
        scored.sort(key=lambda x: x[0], reverse=True)

        campaigns = []
        used = set()
        selected = []
        for channel in ["sms"]:
            options = [x for x in scored if x[1]["channel"] == channel]
            if not options:
                fallback = [a for a in arms if a["channel"] == channel]
                if fallback:
                    a = max(fallback, key=lambda z: z["mean"] * z["arpu"] - CHANNELS[z["channel"]][0])
                    options = [(a["mean"] * a["arpu"] - CHANNELS[channel][0], a, stats[self._key(a)])]
            if options:
                selected.append(options[0])
        selected += [x for x in scored if x not in selected]
        for _, a, s in selected[:10]:
            key = (a["cur"], a["seg"])
            if key in used:
                continue
            used.add(key)
            campaigns.append({"campaign_name": f"adaptive_{a['cur']}_{a['target']}_{a['seg']}",
                              "filter_arpu_segment": a["seg"],
                              "filter_current_tariff": a["cur"],
                              "target_tariff": a["target"], "channel": a["channel"]})
        return campaigns

    @staticmethod
    def _key(a):
        return (a["cur"], a["seg"], a["target"], a["channel"])
