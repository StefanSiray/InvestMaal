"""
Shared Data Models for Investment Goal Analyzer
================================================
Fælles dataklasser og forretningslogik.

Skattelogik:
- ASK (Aktiesparekonto): Lagerskatsprincippet - beskattes årligt på urealiserede gevinster
- Frie Midler: Realisationsbeskatning - beskattes KUN ved salg/realisering
"""

from dataclasses import dataclass
from typing import Dict, Optional


# -------------------------------------------------------------
# DATA CLASSES
# -------------------------------------------------------------
@dataclass
class MarketScenario:
    """Definerer et markedsscenarie"""
    name: str
    annual_return: float
    volatility: float
    description: str
    color: Optional[str] = None  # Bruges til visualisering i Streamlit


@dataclass
class TaxRules:
    """
    Skatteregler for danske investeringskonti
    
    ASK: 17% lagerskat (beskattes årligt på urealiseret gevinst)
    Frie midler: 27%/42% realisationsbeskatning (kun ved salg)
    """
    ask_rate: float  # Lagerskat på ASK (17%)
    free_low: float  # Lav sats på frie midler ved realisering (27%)
    free_high: float  # Høj sats på frie midler ved realisering (42%)
    limit_initial: float  # Progressionsgrænse (knækgrænse)
    limit_increase: float  # Årlig stigning i grænse (ca. 2.5%)
    
    def get_limit(self, years: int) -> float:
        """Beregn dynamisk knækgrænse baseret på antal år"""
        return self.limit_initial * ((1 + self.limit_increase) ** years)


@dataclass
class ASKRules:
    """ASK-specifikke regler (indskudsloft)"""
    initial_ceiling: float  # Startloft (fx 135.900 kr i 2025)
    annual_increase: float  # Årlig forhøjelse af loftet
    
    def get_ceiling(self, year_idx: int) -> float:
        """Beregn ASK-loft for et givet år"""
        return self.initial_ceiling + (year_idx * self.annual_increase)


class Portfolio:
    """
    Porteføljeklasse til at håndtere balancer og skattelogik
    
    Skattebehandling:
    - ASK: Lagerskatsprincippet - skat betales årligt på gevinst (både realiseret og urealiseret)
    - Frie Midler: Realisationsbeskatning - skat betales KUN når du sælger
    - Cash: Renteindtægter beskattes som kapitalindkomst (ikke modelleret separat her)
    """
    
    def __init__(self, ask_balance: float, free_balance: float, cash_balance: float):
        self.ask = ask_balance
        self.free = free_balance
        self.cash = cash_balance
        
        # Tracker kostbasis for frie midler (til realisationsbeskatning)
        self.invested_cost_free = free_balance
        
        # Tracker årets startværdi for ASK (til lagerskat)
        self.ask_start_year_val = ask_balance
    
    def total_value(self) -> float:
        """Total porteføljeværdi (brutto, før eventuel skat på frie midler)"""
        return self.ask + self.free + self.cash
    
    def apply_market_shock(self, stock_shock: float, cash_rate: float):
        """Anvend markedsbevægelse på alle konti"""
        self.ask *= (1 + stock_shock)
        self.free *= (1 + stock_shock)
        self.cash *= (1 + cash_rate)
    
    def apply_ask_tax(self, tax_rate: float):
        """
        Anvend lagerskat på ASK (kaldes årligt ved årsskifte)
        
        Lagerskatsprincippet: Skat beregnes på ÅRETS gevinst (værdi nu - værdi ved årets start)
        Skatten trækkes fra ASK-kontoen.
        """
        gain = max(0, self.ask - self.ask_start_year_val)
        tax = gain * tax_rate
        self.ask -= tax
        # Opdater startværdi til næste år
        self.ask_start_year_val = self.ask
    
    def contribute(self, amount: float, allocation: Dict[str, float], 
                   ask_ceiling: float, ask_rules: ASKRules):
        """
        Håndter månedligt bidrag med spillover-logik
        
        ASK-loftet gælder for den SAMLEDE VÆRDI af kontoen (ikke kun indskud).
        Hvis værdien er steget pga. afkast, reduceres pladsen til nye indskud.
        Overskydende beløb "spiller over" til frie midler.
        """
        # Cash (boligopsparing)
        to_cash = amount * allocation.get("boligopsparing_cash", 0)
        self.cash += to_cash
        
        # ASK - tjek om der er plads under loftet
        to_ask_weighted = amount * allocation.get("aktiesparekonto_ask", 0)
        value_room_left = max(0, ask_ceiling - self.ask)
        to_ask = min(to_ask_weighted, value_room_left)
        self.ask += to_ask
        
        # Spillover til frie midler (når ASK-loftet er nået)
        spillover = max(0, to_ask_weighted - value_room_left)
        
        # Frie midler (inkl. spillover fra ASK)
        to_free_weighted = amount * allocation.get("maandsopsparing_free", 0)
        to_free = to_free_weighted + spillover
        self.free += to_free
        
        # Opdater kostbasis for frie midler (til realisationsbeskatning)
        self.invested_cost_free += to_free
    
    def calculate_free_tax_at_realization(self, tax_rules: TaxRules, years: int) -> float:
        """
        Beregn progressiv skat på frie midler VED REALISERING.
        
        Realisationsbeskatning: Skat betales KUN når du sælger.
        - Gevinst op til knækgrænsen: 27% skat
        - Gevinst over knækgrænsen: 42% skat
        
        Knækgrænsen stiger årligt (ca. 2.5% p.a.)
        """
        total_gain = max(0, self.free - self.invested_cost_free)
        dynamic_limit = tax_rules.get_limit(years)
        
        if total_gain <= 0:
            return 0.0
        elif total_gain <= dynamic_limit:
            return total_gain * tax_rules.free_low
        else:
            # Progressiv beskatning
            tax_low = dynamic_limit * tax_rules.free_low
            tax_high = (total_gain - dynamic_limit) * tax_rules.free_high
            return tax_low + tax_high
    
    def get_net_value(self, tax_rules: TaxRules, years: int) -> float:
        """
        Få nettoværdi efter skat ved fuld realisering af frie midler.
        
        ASK er allerede beskattet løbende (lagerskat), så ASK-værdien er "ren".
        Frie midler skal fratrækkes realisationsskat.
        Cash er ikke beskattet (antager det er kontanter/opsparing).
        """
        free_tax = self.calculate_free_tax_at_realization(tax_rules, years)
        return self.ask + (self.free - free_tax) + self.cash
    
    def get_unrealized_gain(self) -> float:
        """Få urealiseret gevinst på frie midler"""
        return max(0, self.free - self.invested_cost_free)
    
    # Alias for backwards compatibility with investGoal.py
    def calculate_free_tax(self, tax_rules: TaxRules, years: int) -> float:
        """Alias for calculate_free_tax_at_realization"""
        return self.calculate_free_tax_at_realization(tax_rules, years)
