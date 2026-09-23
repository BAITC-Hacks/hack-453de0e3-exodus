from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
import math
import os

import numpy as np
import pandas as pd

from agent import Agent

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
PROFILE = pd.read_csv(DATA / "customer_profile_optimized.csv")
TARIFFS = pd.read_csv(DATA / "dict_tariff.csv")
BASELINE = pd.read_csv(DATA / "segment_baseline.csv")
CHANNELS = {
    "push": {"cost_per_contact": 0.0, "conversion_multiplier": 0.50},
    "sms": {"cost_per_contact": 4.0, "conversion_multiplier": 0.65},
    "digital_ads": {"cost_per_contact": 22.0, "conversion_multiplier": 0.85},
    "call": {"cost_per_contact": 160.0, "conversion_multiplier": 1.20},
}


class DemoEnvironment:
    """Small case-compatible simulator for interactive local demonstration."""
    def __init__(self):
        self.customer_profile = PROFILE.copy()
        self.tariffs = TARIFFS.copy()
        self.channels = CHANNELS
        self.total_budget = 100000.0
        self.remaining_budget = self.total_budget
        self.remaining_contacts = 15000
        self.pilots_left = 20
        self.pilot_history = []

    def run_pilot(self, target_tariff, channel, n_customers=100,
                  filter_arpu_segment=None, filter_current_tariff=None,
                  filter_data_segment=None, filter_call_segment=None):
        mask = pd.Series(True, index=self.customer_profile.index)
        for col, val in (("arpu_segment", filter_arpu_segment),
                         ("current_tariff", filter_current_tariff),
                         ("data_segment", filter_data_segment),
                         ("call_segment", filter_call_segment)):
            if val:
                mask &= self.customer_profile[col].astype(str).eq(str(val))
        audience = self.customer_profile.loc[mask]
        if audience.empty or self.pilots_left <= 0:
            raise ValueError("Нет доступной аудитории или пилотов")
        n = min(int(n_customers), len(audience), self.remaining_contacts, 200)
        rate = CHANNELS[channel]["cost_per_contact"]
        if rate:
            n = min(n, int(self.remaining_budget // rate))
        if n < 10:
            raise ValueError("Недостаточно контактов для пилота")
        seed = 20 - self.pilots_left
        rng = np.random.default_rng(seed + 104729 * len(self.pilot_history))
        sample = audience.sample(n=n, random_state=seed + len(self.pilot_history))
        base = rng.normal(0.075, 0.08)
        tariff_price = float(self.tariffs.loc[self.tariffs.tariff_plan_code.eq(target_tariff), "price_tariff"].iloc[0])
        current_price = sample["current_tariff_price"].fillna(sample["ARPU_3m_avg"]).mean()
        tariff_signal = np.clip((tariff_price - current_price) / max(current_price, 1000), -0.20, 0.20)
        segment_signal = {"LOW": .02, "MID": .04, "HIGH": .06}.get(str(filter_arpu_segment), .03)
        true_lift = base + segment_signal + tariff_signal
        multiplier = CHANNELS[channel]["conversion_multiplier"]
        observed = float(np.clip((true_lift + rng.normal(0, .804 / math.sqrt(n))) * multiplier, -.6, 1.2))
        cost = n * rate
        result = {"campaign_name": f"pilot_{len(self.pilot_history)+1}",
                  "target_tariff": target_tariff, "channel": channel,
                  "n_customers": n, "cost": cost,
                  "observed_lift_ratio": observed,
                  "filter_arpu_segment": filter_arpu_segment,
                  "filter_current_tariff": filter_current_tariff,
                  "filter_data_segment": filter_data_segment,
                  "filter_call_segment": filter_call_segment}
        self.pilot_history.append(result)
        self.remaining_budget -= cost
        self.remaining_contacts -= n
        self.pilots_left -= 1
        return result


def json_safe(value):
    if isinstance(value, dict): return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [json_safe(x) for x in value]
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return float(value)
    if pd.isna(value): return None
    return value


def friendly(result, env):
    campaigns = []
    for raw in result:
        mask = pd.Series(True, index=env.customer_profile.index)
        for col, key in (("arpu_segment", "filter_arpu_segment"),
                         ("current_tariff", "filter_current_tariff"),
                         ("data_segment", "filter_data_segment"),
                         ("call_segment", "filter_call_segment")):
            val = raw.get(key)
            if val: mask &= env.customer_profile[col].astype(str).eq(str(val))
        size = min(5000, int(mask.sum()))
        channel_rate = CHANNELS[raw["channel"]]["cost_per_contact"]
        max_affordable = int(max(0, (env.total_budget - sum(p["cost"] for p in env.pilot_history)) // max(channel_rate, 0.000001))) if channel_rate else 15000
        size = min(size, max_affordable, max(0, 15000 - sum(p["n_customers"] for p in env.pilot_history)))
        campaign = dict(raw)
        campaign["audience_size"] = size
        campaigns.append(campaign)
    tariff_names = {r.tariff_plan_code: f"{int(r.price_tariff):,} у.е. · {int(r.Data_in_PKG):,} МБ".replace(",", " ") for r in TARIFFS.itertuples()}
    channel_names = {"push": "Push", "sms": "SMS", "digital_ads": "Реклама", "call": "Звонок"}
    piloted = {p["target_tariff"] for p in env.pilot_history}
    for c in campaigns:
        c["tariff_label"] = tariff_names.get(c["target_tariff"], c["target_tariff"])
        c["channel_label"] = channel_names.get(c["channel"], c["channel"])
        matched = [p for p in env.pilot_history if p["target_tariff"] == c["target_tariff"] and p.get("filter_arpu_segment") == c.get("filter_arpu_segment")]
        c["pilot_count"] = len(matched)
        c["pilot_mean"] = sum(p["observed_lift_ratio"] for p in matched) / len(matched) if matched else None
    return campaigns


def recommend():
    env = DemoEnvironment()
    plans = Agent().act(env)
    campaigns = friendly(plans, env)
    # The model may suggest overlapping or oversized cohorts. Apply case limits
    # before showing them: one contact per subscriber, max 5,000 per campaign,
    # 15,000 total contacts and 100,000 total spend including pilots.
    used = set()
    normalized = []
    spent_so_far = sum(p["cost"] for p in env.pilot_history)
    contacts_so_far = sum(p["n_customers"] for p in env.pilot_history)
    for c in campaigns:
        mask = pd.Series(True, index=PROFILE.index)
        for col, key in (("arpu_segment", "filter_arpu_segment"),
                         ("current_tariff", "filter_current_tariff"),
                         ("data_segment", "filter_data_segment"),
                         ("call_segment", "filter_call_segment")):
            if c.get(key): mask &= PROFILE[col].astype(str).eq(str(c[key]))
        ids = PROFILE.loc[mask & ~PROFILE.ID_NUMBER.isin(used), "ID_NUMBER"].tolist()
        size = min(5000, len(ids), 15000 - contacts_so_far)
        rate = CHANNELS[c["channel"]]["cost_per_contact"]
        if rate: size = min(size, int(max(0, 100000 - spent_so_far) // rate))
        if size < 1: continue
        ids = ids[:size]
        used.update(ids); c["audience_size"] = size
        normalized.append(c)
        charge = size * rate
        spent_so_far += charge; contacts_so_far += size
    campaigns = normalized
    total_cost = sum(c["audience_size"] * CHANNELS[c["channel"]]["cost_per_contact"] for c in campaigns)
    total_contacts = sum(c["audience_size"] for c in campaigns)
    summary = {
        "customers": len(PROFILE), "budget": 100000,
        "spent": round(total_cost + sum(p["cost"] for p in env.pilot_history), 2),
        "campaign_spend": round(total_cost, 2),
        "pilot_spend": round(sum(p["cost"] for p in env.pilot_history), 2),
        "contacts": total_contacts + sum(p["n_customers"] for p in env.pilot_history),
        "pilot_contacts": sum(p["n_customers"] for p in env.pilot_history),
        "pilots": len(env.pilot_history), "campaigns": len(campaigns),
    }
    if campaigns:
        top = max(campaigns, key=lambda c: c["audience_size"])
        summary["recommendation"] = f"Сначала охватить сегмент {top.get('filter_arpu_segment','всей базы')} предложением перейти на {top['tariff_label']} через канал «{top['channel_label']}». Это рекомендация агента после {len(env.pilot_history)} пилотов; потенциальный прирост нельзя считать гарантированным."
    else:
        summary["recommendation"] = "Агент не нашёл кампаний с положительной ожидаемой ценностью. Повторите запуск после проверки входных данных."
    return {"summary": summary, "campaigns": campaigns,
            "pilots": env.pilot_history,
            "channels": [{"name": "Push", "cost": 0, "effect": "×0,50"},
                         {"name": "SMS", "cost": 4, "effect": "×0,65"},
                         {"name": "Реклама", "cost": 22, "effect": "×0,85"},
                         {"name": "Звонок", "cost": 160, "effect": "×1,20"}],
            "baseline": BASELINE.groupby("arpu_segment", as_index=False).agg(customers=("customers", "sum"), mean_arpu=("mean_arpu_3m", "mean")).to_dict("records") if "mean_arpu_3m" in BASELINE else []}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/recommendations":
            try:
                payload = json.dumps(recommend(), ensure_ascii=False, default=json_safe).encode()
                self.send_response(200); self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(payload))); self.end_headers(); self.wfile.write(payload)
            except Exception as e:
                payload = json.dumps({"error": str(e)}, ensure_ascii=False).encode()
                self.send_response(500); self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(payload))); self.end_headers(); self.wfile.write(payload)
            return
        path = BASE.parent / "index.html"
        payload = path.read_bytes()
        self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload))); self.end_headers(); self.wfile.write(payload)
    def log_message(self, *_): pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8001"))
    host = os.environ.get("HOST", "0.0.0.0")
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Beeline Campaign Copilot listening on {host}:{port}", flush=True)
    server.serve_forever()
