from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.api.recommender import PortfolioRecommender
from src.rl.contract import CAUSAL_TRAINING_BACKEND
import src.api.recommender as recommender_module


class FeatureStoreStub:
    def __init__(self, snapshots: dict[str, pd.DataFrame]):
        self.snapshots = snapshots
        self.calls: list[str] = []

    def snapshot(self, ft: str, as_of: pd.Timestamp) -> pd.DataFrame:
        self.calls.append(ft)
        value = self.snapshots[ft]
        return value.copy() if isinstance(value, pd.DataFrame) else value


class UniverseManagerStub:
    def __init__(self, sectors: list[str], tickers: list[str]):
        self.snapshot = SimpleNamespace(
            sectors=sectors,
            tickers=tickers,
            stocks=[SimpleNamespace(ticker=t, cap="large") for t in tickers],
        )

    def get_universe(self, as_of, price_matrix=None):
        return self.snapshot

    @staticmethod
    def get_sector_map(snapshot) -> dict[str, str]:
        return {stock.ticker: snapshot.sectors[0] for stock in snapshot.stocks}


class UniverseManagerMappingStub:
    def __init__(self, sectors: list[str], ticker_sector_pairs: list[tuple[str, str]]):
        self.snapshot = SimpleNamespace(
            sectors=sectors,
            tickers=[ticker for ticker, _ in ticker_sector_pairs],
            stocks=[
                SimpleNamespace(ticker=ticker, cap="large")
                for ticker, _ in ticker_sector_pairs
            ],
        )
        self._sector_map = {ticker: sector for ticker, sector in ticker_sector_pairs}

    def get_universe(self, as_of, price_matrix=None):
        return self.snapshot

    def get_sector_map(self, snapshot) -> dict[str, str]:
        return dict(self._sector_map)


class OptimizerStub:
    def __init__(self, target_weights: dict[str, float]):
        self.target_weights = target_weights
        self.calls: list[dict] = []

    def optimize(self, **kwargs) -> dict[str, float]:
        self.calls.append(kwargs)
        return self.target_weights


def _make_recommender(monkeypatch, tmp_path) -> PortfolioRecommender:
    cfg = {
        "paths": {
            "model_dir": str(tmp_path / "models"),
            "artifact_dir": str(tmp_path / "artifacts"),
            "feature_data": str(tmp_path / "features"),
        },
        "stock_model": {"top_k_per_sector": 2},
        "sector_model": {},
        "rl": {},
    }
    monkeypatch.setattr(PortfolioRecommender, "_try_load", lambda self: None)
    rec = PortfolioRecommender(cfg)
    rec._loaded = True
    rec._price_matrix = pd.DataFrame(
        {"AAA": [100.0, 101.0], "BBB": [50.0, 51.0]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )
    return rec


def _make_recommender_with_cfg(monkeypatch, tmp_path, cfg_overrides: dict | None = None) -> PortfolioRecommender:
    cfg = {
        "paths": {
            "model_dir": str(tmp_path / "models"),
            "artifact_dir": str(tmp_path / "artifacts"),
            "feature_data": str(tmp_path / "features"),
        },
        "stock_model": {"top_k_per_sector": 2},
        "sector_model": {},
        "rl": {},
    }
    if cfg_overrides:
        for key, value in cfg_overrides.items():
            if isinstance(value, dict) and isinstance(cfg.get(key), dict):
                cfg[key] = {**cfg[key], **value}
            else:
                cfg[key] = value
    monkeypatch.setattr(PortfolioRecommender, "_try_load", lambda self: None)
    rec = PortfolioRecommender(cfg)
    rec._loaded = True
    rec._price_matrix = pd.DataFrame(
        {"AAA": [100.0, 101.0], "BBB": [50.0, 51.0]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )
    return rec


def _patch_covariance(monkeypatch):
    monkeypatch.setattr(
        recommender_module.PortfolioOptimizer,
        "estimate_covariance",
        staticmethod(
            lambda prices, tickers: pd.DataFrame(
                np.eye(len(tickers)),
                index=tickers,
                columns=tickers,
            )
        ),
    )


def test_recommender_uses_macro_feature_snapshot_for_sector_and_rl(monkeypatch, tmp_path):
    rec = _make_recommender_with_cfg(
        monkeypatch,
        tmp_path,
        {"rl": {"use_rl": True, "training_backend": CAUSAL_TRAINING_BACKEND}},
    )
    _patch_covariance(monkeypatch)

    class SectorScorerStub:
        is_fitted = True

        def __init__(self):
            self.macro_features_row = None

        def predict(self, sector_features_row, macro_features_row=None):
            self.macro_features_row = macro_features_row.copy()
            return {"IT": 0.8}

    class StockRankerStub:
        def rank_stocks(self, stock_features_snap, sector, top_k=None):
            return pd.DataFrame({"ticker": ["AAA"], "score": [0.7], "rank": [1]})

    class RLAgentStub:
        is_trained = True

        def __init__(self):
            self.macro_state = None

        def decide(self, macro_state, sector_state, portfolio_state):
            self.macro_state = dict(macro_state)
            return {
                "sector_tilts": {"IT": 1.1},
                "cash_target": 0.05,
                "aggressiveness": 1.0,
                "should_rebalance": True,
            }

    rec._feature_store = FeatureStoreStub(
        {
            "sector": pd.DataFrame(
                {"sector": ["IT"], "mom_1m": [0.1], "mom_3m": [0.2], "rel_str_1m": [0.05], "breadth_3m": [0.6]}
            ),
            "stock": pd.DataFrame(
                {"ticker": ["AAA"], "sector": ["IT"], "ret_3m": [0.12], "mom_3m": [0.12]}
            ),
            "macro": pd.DataFrame(
                {"macro_stress_score": [0.25], "vix_level": [14.0]},
                index=pd.to_datetime(["2024-01-02"]),
            ),
        }
    )
    rec._macro_mgr = SimpleNamespace(load=lambda: (_ for _ in ()).throw(AssertionError("raw macro load should not be used")))
    rec._universe_mgr = UniverseManagerStub(["IT"], ["AAA"])
    rec._sector_scorer = SectorScorerStub()
    rec._stock_ranker = StockRankerStub()
    rec._rl_agent = RLAgentStub()
    rec._optimizer = OptimizerStub({"AAA": 0.9, "CASH": 0.1})

    result = rec.recommend(capital_inr=100_000, risk_profile="moderate")

    assert result["model_mode"] == "RL"
    assert "macro" in rec._feature_store.calls
    assert rec._sector_scorer.macro_features_row["macro_stress_score"] == 0.25
    assert rec._rl_agent.macro_state["macro_stress_score"] == 0.25
    assert result["allocation"]["AAA"] == 0.9


def test_recommender_falls_back_when_models_are_missing(monkeypatch, tmp_path):
    rec = _make_recommender(monkeypatch, tmp_path)
    _patch_covariance(monkeypatch)

    rec._feature_store = FeatureStoreStub(
        {
            "sector": pd.DataFrame({"sector": ["IT"], "mom_3m": [0.15]}),
            "stock": pd.DataFrame(
                {
                    "ticker": ["AAA", "BBB"],
                    "sector": ["IT", "IT"],
                    "ret_3m": [0.12, 0.05],
                    "mom_3m": [0.12, 0.05],
                }
            ),
            "macro": pd.DataFrame(
                {"macro_stress_score": [0.2], "vix_level": [15.0]},
                index=pd.to_datetime(["2024-01-02"]),
            ),
        }
    )
    rec._macro_mgr = SimpleNamespace(load=lambda: pd.DataFrame())
    rec._universe_mgr = UniverseManagerStub(["IT"], ["AAA", "BBB"])
    rec._sector_scorer = None
    rec._stock_ranker = None
    rec._rl_agent = None
    rec._optimizer = OptimizerStub({"AAA": 0.6, "BBB": 0.3, "CASH": 0.1})

    result = rec.recommend(capital_inr=100_000, risk_profile="moderate")

    assert result["model_mode"] == "Baseline"
    assert result["allocation"]["AAA"] == 0.6
    assert result["allocation"]["BBB"] == 0.3
    assert result["sector_tilts"]["IT"] == 1.0


def test_recommender_bypasses_trained_rl_when_rl_is_disabled(monkeypatch, tmp_path):
    rec = _make_recommender_with_cfg(
        monkeypatch,
        tmp_path,
        {"rl": {"use_rl": True, "training_backend": "disabled"}},
    )
    _patch_covariance(monkeypatch)

    class SectorScorerStub:
        is_fitted = True

        def predict(self, sector_features_row, macro_features_row=None):
            return {"IT": 0.8}

    class StockRankerStub:
        def rank_stocks(self, stock_features_snap, sector, top_k=None):
            return pd.DataFrame({"ticker": ["AAA"], "score": [0.7], "rank": [1]})

    class RLAgentStub:
        is_trained = True

        def __init__(self):
            self.calls = 0

        def decide(self, macro_state, sector_state, portfolio_state):
            self.calls += 1
            return {
                "sector_tilts": {"IT": 1.7},
                "cash_target": 0.20,
                "aggressiveness": 0.7,
                "should_rebalance": True,
            }

    rl_agent = RLAgentStub()
    optimizer = OptimizerStub({"AAA": 0.9, "CASH": 0.1})

    rec._feature_store = FeatureStoreStub(
        {
            "sector": pd.DataFrame({"sector": ["IT"], "mom_3m": [0.15]}),
            "stock": pd.DataFrame(
                {"ticker": ["AAA"], "sector": ["IT"], "ret_3m": [0.12], "mom_3m": [0.12]}
            ),
            "macro": pd.DataFrame(
                {"macro_stress_score": [0.2], "vix_level": [15.0]},
                index=pd.to_datetime(["2024-01-02"]),
            ),
        }
    )
    rec._macro_mgr = SimpleNamespace(load=lambda: pd.DataFrame())
    rec._universe_mgr = UniverseManagerStub(["IT"], ["AAA"])
    rec._sector_scorer = SectorScorerStub()
    rec._stock_ranker = StockRankerStub()
    rec._rl_agent = rl_agent
    rec._optimizer = optimizer

    result = rec.recommend(capital_inr=100_000, risk_profile="moderate")

    assert rl_agent.calls == 0
    assert result["model_mode"] == "Baseline"
    assert optimizer.calls[-1]["sector_tilts"]["IT"] == 1.0
    assert optimizer.calls[-1]["posture"] == "neutral"


def test_recommender_baseline_uses_full_stack_sector_first_selection(monkeypatch, tmp_path):
    rec = _make_recommender_with_cfg(
        monkeypatch,
        tmp_path,
        {
            "rl": {
                "use_rl": True,
                "training_backend": "disabled",
                "posture_profiles": {
                    "neutral": {
                        "cash_target": 0.05,
                        "aggressiveness": 1.0,
                        "turnover_cap": 0.35,
                        "sector_top_n": 1,
                        "stock_top_k_per_sector": 1,
                    }
                },
            }
        },
    )
    _patch_covariance(monkeypatch)

    class SectorScorerStub:
        is_fitted = True

        def predict(self, sector_features_row, macro_features_row=None):
            return {"IT": 0.9, "Banking": 0.6, "FMCG": 0.3}

    class StockRankerStub:
        def rank_stocks(self, stock_features_snap, sector, top_k=None):
            rows = stock_features_snap[stock_features_snap["sector"] == sector].copy()
            rows = rows.sort_values("ret_3m", ascending=False).reset_index(drop=True)
            rows["score"] = rows["ret_3m"].astype(float)
            rows["rank"] = range(1, len(rows) + 1)
            return rows[["ticker", "score", "rank"]].head(top_k)

    optimizer = OptimizerStub({"AAA": 0.9, "CASH": 0.1})

    rec._feature_store = FeatureStoreStub(
        {
            "sector": pd.DataFrame(
                {
                    "sector": ["IT", "Banking", "FMCG"],
                    "mom_3m": [0.2, 0.1, 0.05],
                    "breadth_3m": [0.6, 0.6, 0.6],
                }
            ),
            "stock": pd.DataFrame(
                {
                    "ticker": ["AAA", "AAB", "BBB", "BBC", "CCC", "CCD"],
                    "sector": ["IT", "IT", "Banking", "Banking", "FMCG", "FMCG"],
                    "ret_3m": [0.20, 0.15, 0.12, 0.08, 0.09, 0.07],
                    "mom_3m": [0.20, 0.15, 0.12, 0.08, 0.09, 0.07],
                }
            ),
            "macro": pd.DataFrame(
                {"macro_stress_score": [0.2], "vix_level": [15.0]},
                index=pd.to_datetime(["2024-01-02"]),
            ),
        }
    )
    rec._macro_mgr = SimpleNamespace(load=lambda: pd.DataFrame())
    rec._universe_mgr = UniverseManagerMappingStub(
        ["IT", "Banking", "FMCG"],
        [("AAA", "IT"), ("AAB", "IT"), ("BBB", "Banking"), ("BBC", "Banking"), ("CCC", "FMCG"), ("CCD", "FMCG")],
    )
    rec._sector_scorer = SectorScorerStub()
    rec._stock_ranker = StockRankerStub()
    rec._rl_agent = None
    rec._optimizer = optimizer

    result = rec.recommend(capital_inr=100_000, risk_profile="moderate")

    assert result["model_mode"] == "Baseline"
    assert set(optimizer.calls[-1]["alpha_scores"].keys()) == {"AAA"}
    assert optimizer.calls[-1]["posture"] == "neutral"


def test_recommender_matches_backtest_rl_application_semantics(monkeypatch, tmp_path):
    rec = _make_recommender_with_cfg(
        monkeypatch,
        tmp_path,
        {"rl": {"use_rl": True, "training_backend": CAUSAL_TRAINING_BACKEND}},
    )
    _patch_covariance(monkeypatch)

    class SectorScorerStub:
        is_fitted = True

        def predict(self, sector_features_row, macro_features_row=None):
            return {"IT": 0.9}

    class StockRankerStub:
        def rank_stocks(self, stock_features_snap, sector, top_k=None):
            return pd.DataFrame({"ticker": ["AAA"], "score": [0.8], "rank": [1]})

    class RLAgentStub:
        is_trained = True

        def decide(self, macro_state, sector_state, portfolio_state):
            return {
                "sector_tilts": {"IT": 1.5},
                "cash_target": 0.08,
                "aggressiveness": 0.7,
                "should_rebalance": True,
            }

    optimizer = OptimizerStub({"AAA": 0.88, "CASH": 0.12})

    rec._feature_store = FeatureStoreStub(
        {
            "sector": pd.DataFrame(
                {"sector": ["IT"], "mom_1m": [0.1], "mom_3m": [0.2], "rel_str_1m": [0.05], "breadth_3m": [0.6]}
            ),
            "stock": pd.DataFrame(
                {"ticker": ["AAA"], "sector": ["IT"], "ret_3m": [0.12], "mom_3m": [0.12]}
            ),
            "macro": pd.DataFrame(
                {"macro_stress_score": [0.25], "vix_level": [14.0]},
                index=pd.to_datetime(["2024-01-02"]),
            ),
        }
    )
    rec._macro_mgr = SimpleNamespace(load=lambda: pd.DataFrame())
    rec._universe_mgr = UniverseManagerStub(["IT"], ["AAA"])
    rec._sector_scorer = SectorScorerStub()
    rec._stock_ranker = StockRankerStub()
    rec._rl_agent = RLAgentStub()
    rec._optimizer = optimizer

    result = rec.recommend(capital_inr=100_000, risk_profile="moderate")
    optimize_call = optimizer.calls[-1]

    assert result["model_mode"] == "RL"
    assert optimize_call["alpha_scores"]["AAA"] == pytest.approx(0.8)
    assert optimize_call["sector_tilts"]["IT"] == pytest.approx(1.5)
    assert optimize_call["aggressiveness"] == pytest.approx(0.7)


def test_api_returns_503_when_recommender_is_unavailable(monkeypatch, tmp_path):
    pytest.importorskip("httpx")
    testclient_module = pytest.importorskip("fastapi.testclient")
    TestClient = testclient_module.TestClient

    import src.config as config_module
    import src.api.portfolio_store as portfolio_store_module
    import src.api.recommender as api_recommender_module

    class FakeRecommender:
        def __init__(self, cfg):
            self.cfg = cfg

        def is_ready(self):
            return False

        def recommend(self, **kwargs):
            raise RuntimeError("backend unavailable")

    class FakePortfolioStore:
        def __init__(self, path):
            self.path = path

        def save(self, portfolio_id, payload):
            return None

        def load(self, portfolio_id):
            return None

        def list_all(self):
            return []

    monkeypatch.setattr(
        config_module,
        "load_config",
        lambda: {
            "paths": {
                "artifact_dir": str(tmp_path / "artifacts"),
                "model_dir": str(tmp_path / "models"),
            }
        },
    )
    monkeypatch.setattr(api_recommender_module, "PortfolioRecommender", FakeRecommender)
    monkeypatch.setattr(portfolio_store_module, "PortfolioStore", FakePortfolioStore)

    sys.modules.pop("api.main", None)
    api_main = importlib.import_module("api.main")

    client = TestClient(api_main.app)
    response = client.get("/suggest")

    assert response.status_code == 503
    assert response.json()["detail"] == "backend unavailable"
