"""Portfolio optimization functions for multi-strategy allocation and risk management."""

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from bot.fp.analysis.performance import (
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
)
from bot.fp.types.result import Failure, Success
from bot.fp.types.result import Result as Either

logger = logging.getLogger(__name__)


# Types
StrategyWeights = dict[str, float]
StrategyReturns = dict[str, pd.Series]
CovarianceMatrix = pd.DataFrame
CorrelationMatrix = pd.DataFrame


def calculate_portfolio_variance(
    weights: np.ndarray, covariance_matrix: np.ndarray
) -> float:
    """Calculate portfolio variance given weights and covariance matrix."""
    return weights.T @ covariance_matrix @ weights


def calculate_portfolio_return(
    weights: np.ndarray, expected_returns: np.ndarray
) -> float:
    """Calculate expected portfolio return given weights and expected returns."""
    return weights.T @ expected_returns


def calculate_risk_contribution(
    weights: np.ndarray, covariance_matrix: np.ndarray
) -> np.ndarray:
    """Calculate risk contribution of each asset to portfolio risk."""
    portfolio_variance = calculate_portfolio_variance(weights, covariance_matrix)
    marginal_contributions = covariance_matrix @ weights
    contributions = weights * marginal_contributions / np.sqrt(portfolio_variance)
    return contributions / np.sum(contributions)


def equal_risk_contribution_objective(
    weights: np.ndarray, covariance_matrix: np.ndarray
) -> float:
    """Objective function for equal risk contribution optimization."""
    risk_contributions = calculate_risk_contribution(weights, covariance_matrix)
    target_risk = 1.0 / len(weights)
    return np.sum((risk_contributions - target_risk) ** 2)


def mean_variance_objective(
    weights: np.ndarray,
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    risk_aversion: float = 1.0,
) -> float:
    """Objective function for mean-variance optimization (maximize utility)."""
    portfolio_return = calculate_portfolio_return(weights, expected_returns)
    portfolio_variance = calculate_portfolio_variance(weights, covariance_matrix)
    return -(portfolio_return - 0.5 * risk_aversion * portfolio_variance)


def sharpe_ratio_objective(
    weights: np.ndarray,
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    risk_free_rate: float = 0.0,
) -> float:
    """Objective function for Sharpe ratio maximization (negative for minimization)."""
    portfolio_return = calculate_portfolio_return(weights, expected_returns)
    portfolio_std = np.sqrt(calculate_portfolio_variance(weights, covariance_matrix))
    if portfolio_std == 0:
        return 0
    return -((portfolio_return - risk_free_rate) / portfolio_std)


def calculate_strategy_correlations(
    returns: StrategyReturns,
) -> Either[str, CorrelationMatrix]:
    """Calculate correlation matrix between strategy returns."""
    try:
        # Align all series to common index
        aligned_returns = pd.DataFrame(returns)

        # Calculate correlation matrix
        correlation_matrix = aligned_returns.corr()

        return Success(correlation_matrix)
    except Exception as e:
        return Failure(f"Failed to calculate correlations: {e!s}")


def calculate_strategy_covariance(
    returns: StrategyReturns,
) -> Either[str, CovarianceMatrix]:
    """Calculate covariance matrix between strategy returns."""
    try:
        # Align all series to common index
        aligned_returns = pd.DataFrame(returns)

        # Calculate covariance matrix
        covariance_matrix = aligned_returns.cov()

        return Success(covariance_matrix)
    except Exception as e:
        return Failure(f"Failed to calculate covariance: {e!s}")


def optimize_risk_parity(
    returns: StrategyReturns, constraints: dict[str, tuple[float, float]] | None = None
) -> Either[str, StrategyWeights]:
    """Optimize portfolio weights using risk parity approach."""

    def _optimize(cov_matrix: CovarianceMatrix) -> Either[str, StrategyWeights]:
        try:
            strategies = list(returns.keys())
            n_strategies = len(strategies)

            # Initial guess: equal weights
            initial_weights = np.ones(n_strategies) / n_strategies

            # Constraints: weights sum to 1, all weights >= 0
            bounds = [(0, 1) for _ in range(n_strategies)]
            if constraints:
                for i, strategy in enumerate(strategies):
                    if strategy in constraints:
                        bounds[i] = constraints[strategy]

            constraints_opt = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

            # Optimize
            result = minimize(
                equal_risk_contribution_objective,
                initial_weights,
                args=(cov_matrix.values,),
                method="SLSQP",
                bounds=bounds,
                constraints=constraints_opt,
                options={"ftol": 1e-8},
            )

            if result.success:
                weights_dict = dict(zip(strategies, result.x, strict=False))
                return Success(weights_dict)
            return Failure(f"Optimization failed: {result.message}")

        except Exception as e:
            return Failure(f"Risk parity optimization error: {e!s}")

    return calculate_strategy_covariance(returns).bind(_optimize)


def optimize_mean_variance(
    returns: StrategyReturns,
    risk_aversion: float = 1.0,
    constraints: dict[str, tuple[float, float]] | None = None,
) -> Either[str, StrategyWeights]:
    """Optimize portfolio weights using mean-variance optimization."""

    def _calculate_expected_returns(aligned_returns: pd.DataFrame) -> np.ndarray:
        return aligned_returns.mean().values

    def _optimize(cov_matrix: CovarianceMatrix) -> Either[str, StrategyWeights]:
        try:
            strategies = list(returns.keys())
            n_strategies = len(strategies)

            # Calculate expected returns
            aligned_returns = pd.DataFrame(returns)
            expected_returns = _calculate_expected_returns(aligned_returns)

            # Initial guess: equal weights
            initial_weights = np.ones(n_strategies) / n_strategies

            # Constraints
            bounds = [(0, 1) for _ in range(n_strategies)]
            if constraints:
                for i, strategy in enumerate(strategies):
                    if strategy in constraints:
                        bounds[i] = constraints[strategy]

            constraints_opt = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

            # Optimize
            result = minimize(
                mean_variance_objective,
                initial_weights,
                args=(expected_returns, cov_matrix.values, risk_aversion),
                method="SLSQP",
                bounds=bounds,
                constraints=constraints_opt,
                options={"ftol": 1e-8},
            )

            if result.success:
                weights_dict = dict(zip(strategies, result.x, strict=False))
                return Success(weights_dict)
            return Failure(f"Optimization failed: {result.message}")

        except Exception as e:
            return Failure(f"Mean-variance optimization error: {e!s}")

    return calculate_strategy_covariance(returns).bind(_optimize)


def optimize_max_sharpe(
    returns: StrategyReturns,
    risk_free_rate: float = 0.0,
    constraints: dict[str, tuple[float, float]] | None = None,
) -> Either[str, StrategyWeights]:
    """Optimize portfolio weights to maximize Sharpe ratio."""

    def _calculate_expected_returns(aligned_returns: pd.DataFrame) -> np.ndarray:
        return aligned_returns.mean().values

    def _optimize(cov_matrix: CovarianceMatrix) -> Either[str, StrategyWeights]:
        try:
            strategies = list(returns.keys())
            n_strategies = len(strategies)

            # Calculate expected returns
            aligned_returns = pd.DataFrame(returns)
            expected_returns = _calculate_expected_returns(aligned_returns)

            # Initial guess: equal weights
            initial_weights = np.ones(n_strategies) / n_strategies

            # Constraints
            bounds = [(0, 1) for _ in range(n_strategies)]
            if constraints:
                for i, strategy in enumerate(strategies):
                    if strategy in constraints:
                        bounds[i] = constraints[strategy]

            constraints_opt = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

            # Optimize
            result = minimize(
                sharpe_ratio_objective,
                initial_weights,
                args=(expected_returns, cov_matrix.values, risk_free_rate),
                method="SLSQP",
                bounds=bounds,
                constraints=constraints_opt,
                options={"ftol": 1e-8},
            )

            if result.success:
                weights_dict = dict(zip(strategies, result.x, strict=False))
                return Success(weights_dict)
            return Failure(f"Optimization failed: {result.message}")

        except Exception as e:
            return Failure(f"Max Sharpe optimization error: {e!s}")

    return calculate_strategy_covariance(returns).bind(_optimize)


def calculate_rebalancing_trades(
    current_weights: StrategyWeights,
    target_weights: StrategyWeights,
    portfolio_value: float,
    threshold: float = 0.01,
) -> Either[str, dict[str, float]]:
    """Calculate trades needed to rebalance portfolio to target weights."""
    try:
        trades = {}

        # Get all strategies
        all_strategies = set(current_weights.keys()) | set(target_weights.keys())

        for strategy in all_strategies:
            current = current_weights.get(strategy, 0.0)
            target = target_weights.get(strategy, 0.0)

            # Calculate difference
            weight_diff = target - current

            # Only rebalance if difference exceeds threshold
            if abs(weight_diff) > threshold:
                trades[strategy] = weight_diff * portfolio_value

        return Success(trades)

    except Exception as e:
        return Failure(f"Failed to calculate rebalancing trades: {e!s}")


def should_rebalance(
    current_weights: StrategyWeights,
    target_weights: StrategyWeights,
    threshold: float = 0.05,
) -> bool:
    """Determine if portfolio needs rebalancing based on weight drift."""
    for strategy in target_weights:
        current = current_weights.get(strategy, 0.0)
        target = target_weights[strategy]

        # Check absolute difference
        if abs(current - target) > threshold:
            return True

        # Check relative difference for non-zero targets
        if target > 0 and abs((current - target) / target) > threshold:
            return True

    return False


def calculate_performance_attribution(
    strategy_returns: StrategyReturns,
    weights: StrategyWeights,
    benchmark_returns: pd.Series | None = None,
) -> Either[str, dict[str, Any]]:
    """Calculate performance attribution for portfolio strategies."""
    try:
        # Align returns
        aligned_returns = pd.DataFrame(strategy_returns)

        # Calculate weighted returns
        weighted_returns = {}
        for strategy, weight in weights.items():
            if strategy in aligned_returns.columns:
                weighted_returns[strategy] = aligned_returns[strategy] * weight

        weighted_df = pd.DataFrame(weighted_returns)

        # Calculate portfolio returns
        portfolio_returns = weighted_df.sum(axis=1)

        # Calculate attribution metrics
        attribution = {
            "total_return": portfolio_returns.sum(),
            "strategy_contributions": {},
            "strategy_attributions": {},
        }

        # Strategy contributions
        for strategy in weights:
            if strategy in aligned_returns.columns:
                contribution = weighted_df[strategy].sum()
                attribution["strategy_contributions"][strategy] = contribution
                attribution["strategy_attributions"][strategy] = (
                    contribution / attribution["total_return"]
                    if attribution["total_return"] != 0
                    else 0
                )

        # Risk attribution
        attribution["risk_contributions"] = {}
        cov_matrix = aligned_returns.cov()
        weights_array = np.array([weights.get(s, 0) for s in aligned_returns.columns])

        if len(weights_array) > 0:
            risk_contribs = calculate_risk_contribution(
                weights_array, cov_matrix.values
            )
            for i, strategy in enumerate(aligned_returns.columns):
                attribution["risk_contributions"][strategy] = float(risk_contribs[i])

        # Active return vs benchmark if provided
        if benchmark_returns is not None:
            aligned_benchmark = benchmark_returns.reindex(portfolio_returns.index)
            attribution["active_return"] = (portfolio_returns - aligned_benchmark).sum()
            attribution["tracking_error"] = (
                portfolio_returns - aligned_benchmark
            ).std()
            attribution["information_ratio"] = (
                attribution["active_return"] / attribution["tracking_error"]
                if attribution["tracking_error"] > 0
                else 0
            )

        return Success(attribution)

    except Exception as e:
        return Failure(f"Failed to calculate performance attribution: {e!s}")


def analyze_strategy_diversification(
    weights: StrategyWeights, correlations: CorrelationMatrix
) -> Either[str, dict[str, float]]:
    """Analyze portfolio diversification metrics."""
    try:
        strategies = list(weights.keys())
        n_strategies = len(strategies)

        # Calculate effective number of strategies (using entropy)
        weights_array = np.array(list(weights.values()))
        weights_array = weights_array[weights_array > 0]  # Remove zero weights

        if len(weights_array) == 0:
            return Success(
                {
                    "effective_strategies": 0,
                    "concentration_ratio": 1.0,
                    "diversification_ratio": 0.0,
                    "average_correlation": 0.0,
                }
            )

        # Entropy-based effective number
        entropy = -np.sum(weights_array * np.log(weights_array))
        effective_strategies = np.exp(entropy)

        # Concentration ratio (Herfindahl index)
        concentration_ratio = np.sum(weights_array**2)

        # Average correlation
        if n_strategies > 1:
            # Get upper triangle of correlation matrix (excluding diagonal)
            mask = np.triu(np.ones_like(correlations), k=1).astype(bool)
            avg_correlation = correlations.values[mask].mean()
        else:
            avg_correlation = 0.0

        # Diversification ratio
        # DR = weighted avg volatility / portfolio volatility
        # Higher ratio means better diversification
        diversification_ratio = 1.0 / np.sqrt(concentration_ratio)

        return Success(
            {
                "effective_strategies": float(effective_strategies),
                "concentration_ratio": float(concentration_ratio),
                "diversification_ratio": float(diversification_ratio),
                "average_correlation": float(avg_correlation),
            }
        )

    except Exception as e:
        return Failure(f"Failed to analyze diversification: {e!s}")


def create_portfolio_report(
    weights: StrategyWeights,
    returns: StrategyReturns,
    benchmark_returns: pd.Series | None = None,
) -> Either[str, dict[str, Any]]:
    """Create comprehensive portfolio analysis report."""
    try:
        report = {
            "weights": weights,
            "metrics": {},
            "risk_analysis": {},
            "correlation_analysis": {},
            "attribution": {},
            "diversification": {},
        }

        # Calculate portfolio returns
        aligned_returns = pd.DataFrame(returns)
        weights_array = np.array([weights.get(s, 0) for s in aligned_returns.columns])
        portfolio_returns = (aligned_returns * weights_array).sum(axis=1)

        # Performance metrics
        report["metrics"] = {
            "total_return": float(portfolio_returns.sum()),
            "annualized_return": float(portfolio_returns.mean() * 252),
            "volatility": float(portfolio_returns.std() * np.sqrt(252)),
            "sharpe_ratio": float(calculate_sharpe_ratio(portfolio_returns)),
            "sortino_ratio": float(calculate_sortino_ratio(portfolio_returns)),
            "max_drawdown": float(calculate_max_drawdown(portfolio_returns)),
        }

        # Risk analysis
        cov_result = calculate_strategy_covariance(returns)
        if isinstance(cov_result, Success):
            cov_matrix = cov_result.value
            portfolio_variance = calculate_portfolio_variance(
                weights_array, cov_matrix.values
            )
            report["risk_analysis"]["portfolio_variance"] = float(portfolio_variance)
            report["risk_analysis"]["portfolio_std"] = float(
                np.sqrt(portfolio_variance)
            )

        # Correlation analysis
        corr_result = calculate_strategy_correlations(returns)
        if isinstance(corr_result, Success):
            report["correlation_analysis"] = corr_result.value.to_dict()

        # Performance attribution
        attr_result = calculate_performance_attribution(
            returns, weights, benchmark_returns
        )
        if isinstance(attr_result, Success):
            report["attribution"] = attr_result.value

        # Diversification analysis
        if isinstance(corr_result, Success):
            div_result = analyze_strategy_diversification(weights, corr_result.value)
            if isinstance(div_result, Success):
                report["diversification"] = div_result.value

        return Success(report)

    except Exception as e:
        return Failure(f"Failed to create portfolio report: {e!s}")


# Functional composition helpers
def optimize_and_analyze(
    returns: StrategyReturns, optimization_method: str = "risk_parity", **kwargs
) -> Either[str, tuple[StrategyWeights, dict[str, Any]]]:
    """Optimize portfolio and provide full analysis."""
    # Select optimization method
    if optimization_method == "risk_parity":

        def optimize_fn(r):
            return optimize_risk_parity(r, kwargs.get("constraints"))

    elif optimization_method == "mean_variance":

        def optimize_fn(r):
            return optimize_mean_variance(
                r, kwargs.get("risk_aversion", 1.0), kwargs.get("constraints")
            )

    elif optimization_method == "max_sharpe":

        def optimize_fn(r):
            return optimize_max_sharpe(
                r, kwargs.get("risk_free_rate", 0.0), kwargs.get("constraints")
            )

    else:
        return Failure(f"Unknown optimization method: {optimization_method}")

    # Optimize weights
    weights_result = optimize_fn(returns)

    if isinstance(weights_result, Failure):
        return weights_result

    # Create report
    report_result = create_portfolio_report(
        weights_result.value, returns, kwargs.get("benchmark_returns")
    )

    if isinstance(report_result, Failure):
        return report_result

    return Success((weights_result.value, report_result.value))
