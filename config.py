"""
Centralized Configuration for Investment Goal Analyzer
=======================================================
Alle standard-konfigurationsværdier samlet ét sted.
"""

from dataclasses import dataclass, field
from typing import Dict
from models import MarketScenario, TaxRules, ASKRules


# -------------------------------------------------------------
# DEFAULT VALUES (Ændr disse for at justere standardindstillinger)
# -------------------------------------------------------------

# Målsætning
DEFAULT_TARGET_GOAL = 1000000  # DKK
DEFAULT_YEARS = 10
DEFAULT_START_YEAR = 2025

# Budget
DEFAULT_INCOME_NET_MONTHLY = 25000  # DKK
DEFAULT_EXPENSES_MONTHLY = 18000  # DKK

# Marked
DEFAULT_INFLATION_RATE = 0.02  # 2%
DEFAULT_CASH_INTEREST = 0.01  # 1%

# Skatteregler (2025)
DEFAULT_ASK_TAX_RATE = 0.17  # 17% lagerskat
DEFAULT_FREE_TAX_LOW = 0.27  # 27% under knækgrænse
DEFAULT_FREE_TAX_HIGH = 0.42  # 42% over knækgrænse
DEFAULT_TAX_LIMIT_INITIAL = 64500  # Knækgrænse 2025
DEFAULT_TAX_LIMIT_INCREASE = 0.025  # Årlig stigning ~2.5%

# ASK regler (2025)
DEFAULT_ASK_CEILING = 166_400  # Indskudsloft 2025
DEFAULT_ASK_ANNUAL_INCREASE = 9000  # Årlig forhøjelse

# Allokering (skal summe til 1.0)
DEFAULT_ALLOCATION = {
    "boligopsparing_cash": 0.05,  # 5% til cash
    "aktiesparekonto_ask": 0.75,  # 75% til ASK
    "maandsopsparing_free": 0.20  # 20% til frie midler
}

# Startbeholdninger
DEFAULT_START_CASH = 92000  # Boligopsparing + andre kontanter
DEFAULT_START_ASK = 166400  # ASK beholdning
DEFAULT_START_FREE = 192000  # Frie midler (månedsopsparing)

# Simulation
DEFAULT_MC_PATHS = 1000  # Antal Monte Carlo simuleringer
DEFAULT_MC_PATHS_SEARCH = 400  # Til optimering/søgning
DEFAULT_MC_PATHS_SENSITIVITY = 200  # Til sensitivity analyse
DEFAULT_SEED = 42  # Random seed for reproducerbarhed

# Markedsscenarier
DEFAULT_SCENARIOS: Dict[str, MarketScenario] = {
    "bull": MarketScenario(
        name="Bull Market",
        annual_return=0.10,
        volatility=0.15,
        description="Stærk økonomi, høj vækst",
        color="#10B981"  # Grøn
    ),
    "normal": MarketScenario(
        name="Normal Market",
        annual_return=0.07,
        volatility=0.20,
        description="Historisk gennemsnit",
        color="#3B82F6"  # Blå
    ),
    "bear": MarketScenario(
        name="Bear Market",
        annual_return=0.04,
        volatility=0.25,
        description="Svag økonomi, høj volatilitet",
        color="#F59E0B"  # Orange
    ),
    "crash": MarketScenario(
        name="Krise",
        annual_return=0.02,
        volatility=0.30,
        description="Økonomisk krise",
        color="#EF4444"  # Rød
    )
}
DEFAULT_SCENARIO = "normal"

# Historiske afkast (S&P 500 Total Return, 1970-2024)
HISTORICAL_RETURNS = {
    1970: 0.0401, 1971: 0.1431, 1972: 0.1898, 1973: -0.1466, 1974: -0.2647,
    1975: 0.3720, 1976: 0.2384, 1977: -0.0718, 1978: 0.0656, 1979: 0.1844,
    1980: 0.3250, 1981: -0.0491, 1982: 0.2155, 1983: 0.2256, 1984: 0.0627,
    1985: 0.3173, 1986: 0.1867, 1987: 0.0525, 1988: 0.1661, 1989: 0.3169,
    1990: -0.0310, 1991: 0.3047, 1992: 0.0762, 1993: 0.1008, 1994: 0.0132,
    1995: 0.3758, 1996: 0.2296, 1997: 0.3336, 1998: 0.2858, 1999: 0.2104,
    2000: -0.0910, 2001: -0.1189, 2002: -0.2210, 2003: 0.2868, 2004: 0.1088,
    2005: 0.0491, 2006: 0.1579, 2007: 0.0549, 2008: -0.3700, 2009: 0.2646,
    2010: 0.1506, 2011: 0.0211, 2012: 0.1600, 2013: 0.3239, 2014: 0.1369,
    2015: 0.0138, 2016: 0.1196, 2017: 0.2183, 2018: -0.0438, 2019: 0.3149,
    2020: 0.1840, 2021: 0.2871, 2022: -0.1811, 2023: 0.2629, 2024: 0.2508,
    2025: 0.12  # Estimated/partial year
}


# -------------------------------------------------------------
# CONFIGURATION CLASS
# -------------------------------------------------------------
@dataclass
class SimulationConfig:
    """
    Samlet konfiguration til simulering.
    Kan initialiseres med defaults eller custom værdier.
    """
    # Målsætning
    target_goal: float = DEFAULT_TARGET_GOAL
    years: int = DEFAULT_YEARS
    start_year: int = DEFAULT_START_YEAR
    
    # Budget
    income_net_monthly: float = DEFAULT_INCOME_NET_MONTHLY
    expenses_monthly: float = DEFAULT_EXPENSES_MONTHLY
    
    # Marked
    inflation_rate: float = DEFAULT_INFLATION_RATE
    cash_interest: float = DEFAULT_CASH_INTEREST
    
    # Allokering
    allocation: Dict[str, float] = field(default_factory=lambda: DEFAULT_ALLOCATION.copy())
    
    # Startbeholdninger
    start_cash: float = DEFAULT_START_CASH
    start_ask: float = DEFAULT_START_ASK
    start_free: float = DEFAULT_START_FREE
    
    # Simulation
    n_paths: int = DEFAULT_MC_PATHS
    n_paths_search: int = DEFAULT_MC_PATHS_SEARCH
    n_paths_sensitivity: int = DEFAULT_MC_PATHS_SENSITIVITY
    seed: int = DEFAULT_SEED
    
    # Skatteregler
    ask_rate: float = DEFAULT_ASK_TAX_RATE
    free_low: float = DEFAULT_FREE_TAX_LOW
    free_high: float = DEFAULT_FREE_TAX_HIGH
    limit_initial: float = DEFAULT_TAX_LIMIT_INITIAL
    limit_increase: float = DEFAULT_TAX_LIMIT_INCREASE
    
    # ASK regler
    ask_ceiling: float = DEFAULT_ASK_CEILING
    ask_annual_increase: float = DEFAULT_ASK_ANNUAL_INCREASE
    
    # Scenarier
    scenarios: Dict[str, MarketScenario] = field(default_factory=lambda: DEFAULT_SCENARIOS.copy())
    default_scenario: str = DEFAULT_SCENARIO
    
    @property
    def monthly_contrib(self) -> float:
        """Beregnet månedligt bidrag"""
        return self.income_net_monthly - self.expenses_monthly
    
    @property
    def tax_rules(self) -> TaxRules:
        """Skatteregler som TaxRules objekt"""
        return TaxRules(
            ask_rate=self.ask_rate,
            free_low=self.free_low,
            free_high=self.free_high,
            limit_initial=self.limit_initial,
            limit_increase=self.limit_increase
        )
    
    @property
    def ask_rules(self) -> ASKRules:
        """ASK regler som ASKRules objekt"""
        return ASKRules(
            initial_ceiling=self.ask_ceiling,
            annual_increase=self.ask_annual_increase
        )
    
    def validate(self) -> None:
        """Validér konfigurationen"""
        assert self.target_goal > 0, "Mål skal være positivt"
        assert 1 <= self.years <= 50, "År skal være mellem 1 og 50"
        assert self.income_net_monthly >= self.expenses_monthly, "Indtægt skal være >= udgifter"
        
        alloc_sum = sum(self.allocation.values())
        assert abs(alloc_sum - 1.0) < 1e-6, f"Allokeringsvægte skal summe til 1.0, fik {alloc_sum}"
        assert all(v >= 0 for v in self.allocation.values()), "Alle vægte skal være ikke-negative"
        
        assert 0 <= self.inflation_rate <= 0.20, "Inflation skal være realistisk (0-20%)"
        assert self.n_paths >= 100, "Mindst 100 simuleringer anbefales"
    
    @classmethod
    def from_sidebar_dict(cls, sidebar_config: Dict) -> 'SimulationConfig':
        """Opret config fra sidebar dictionary (bruges i app.py)"""
        return cls(
            target_goal=sidebar_config['target_goal'],
            years=sidebar_config['years'],
            start_year=sidebar_config['start_year'],
            income_net_monthly=sidebar_config['income'],
            expenses_monthly=sidebar_config['expenses'],
            inflation_rate=sidebar_config['inflation_rate'],
            cash_interest=sidebar_config['cash_interest'],
            allocation={
                "boligopsparing_cash": sidebar_config['allocation_cash'],
                "aktiesparekonto_ask": sidebar_config['allocation_ask'],
                "maandsopsparing_free": sidebar_config['allocation_free']
            },
            start_cash=sidebar_config['start_cash'],
            start_ask=sidebar_config['start_ask'],
            start_free=sidebar_config['start_free'],
            n_paths=sidebar_config['n_paths'],
            seed=sidebar_config['seed']
        )
