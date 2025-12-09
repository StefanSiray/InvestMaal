"""
Investment Goal Analyzer - Interactive Streamlit Dashboard
==========================================================
Moderne, interaktivt dashboard til at analysere investeringsmål.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import shared models and config
from models import MarketScenario, TaxRules, ASKRules, Portfolio
from config import (
    SimulationConfig,
    DEFAULT_TARGET_GOAL, DEFAULT_YEARS, DEFAULT_START_YEAR,
    DEFAULT_INCOME_NET_MONTHLY, DEFAULT_EXPENSES_MONTHLY,
    DEFAULT_INFLATION_RATE, DEFAULT_CASH_INTEREST,
    DEFAULT_ALLOCATION, DEFAULT_START_CASH, DEFAULT_START_ASK, DEFAULT_START_FREE,
    DEFAULT_MC_PATHS, DEFAULT_SEED,
    DEFAULT_ASK_TAX_RATE, DEFAULT_FREE_TAX_LOW, DEFAULT_FREE_TAX_HIGH,
    DEFAULT_TAX_LIMIT_INITIAL, DEFAULT_TAX_LIMIT_INCREASE,
    DEFAULT_ASK_CEILING, DEFAULT_ASK_ANNUAL_INCREASE,
    DEFAULT_SCENARIOS, DEFAULT_SCENARIO,
    HISTORICAL_RETURNS
)


# -------------------------------------------------------------
# PAGE CONFIG & STYLING
# -------------------------------------------------------------
st.set_page_config(
    page_title="InvestMål Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern dark theme
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #00D4AA;
        --secondary-color: #7C3AED;
        --background-dark: #0E1117;
        --card-background: #1E2530;
        --text-primary: #FAFAFA;
        --text-secondary: #9CA3AF;
        --success-color: #10B981;
        --warning-color: #F59E0B;
        --danger-color: #EF4444;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1E2530 0%, #2D3748 100%);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        margin-bottom: 16px;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00D4AA 0%, #7C3AED 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 8px 0;
    }
    
    .metric-label {
        color: #9CA3AF;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .metric-delta-positive {
        color: #10B981;
        font-size: 0.9rem;
    }
    
    .metric-delta-negative {
        color: #EF4444;
        font-size: 0.9rem;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #FAFAFA;
        margin: 24px 0 16px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid #00D4AA;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: #1E2530;
    }
    
    /* Success/Warning/Danger badges */
    .badge-success {
        background: rgba(16, 185, 129, 0.2);
        color: #10B981;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    .badge-warning {
        background: rgba(245, 158, 11, 0.2);
        color: #F59E0B;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    .badge-danger {
        background: rgba(239, 68, 68, 0.2);
        color: #EF4444;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    /* Recommendation box */
    .recommendation-box {
        background: linear-gradient(135deg, rgba(124, 58, 237, 0.1) 0%, rgba(0, 212, 170, 0.1) 100%);
        border: 1px solid rgba(124, 58, 237, 0.3);
        border-radius: 12px;
        padding: 20px;
        margin: 16px 0;
    }
    
    /* Plotly chart container */
    .stPlotlyChart {
        background: #1E2530;
        border-radius: 12px;
        padding: 16px;
        border: 1px solid rgba(255,255,255,0.05);
    }
    
    /* Slider styling */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #00D4AA, #7C3AED);
    }
    
    /* Number input */
    .stNumberInput > div > div > input {
        background: #2D3748;
        border: 1px solid rgba(255,255,255,0.1);
        color: #FAFAFA;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: #2D3748;
        border-radius: 8px;
        padding: 8px 16px;
        color: #9CA3AF;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #00D4AA 0%, #7C3AED 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------
# SIMULATION ENGINE
# -------------------------------------------------------------
from dataclasses import dataclass

@dataclass
class SimulationResult:
    """Resultat fra simulation med alle detaljer"""
    final_nominal: np.ndarray
    final_real: np.ndarray
    history_nominal: np.ndarray
    unrealized_gains: np.ndarray
    # Individuelle kontobeholdninger (median værdier over tid)
    history_ask: np.ndarray
    history_free: np.ndarray
    history_cash: np.ndarray
    # Slutværdier for hver konto
    final_ask: np.ndarray
    final_free: np.ndarray
    final_cash: np.ndarray
    final_free_after_tax: np.ndarray


@st.cache_data(show_spinner=False)
def run_simulation(
    target_goal: float,
    years: int,
    monthly_contrib: float,
    annual_return: float,
    volatility: float,
    inflation_rate: float,
    cash_interest: float,
    ask_rate: float,
    free_low: float,
    free_high: float,
    limit_initial: float,
    limit_increase: float,
    ask_initial_ceiling: float,
    ask_annual_increase: float,
    allocation_cash: float,
    allocation_ask: float,
    allocation_free: float,
    start_cash: float,
    start_ask: float,
    start_free: float,
    n_paths: int,
    seed: int
) -> SimulationResult:
    """
    Kør Monte Carlo simulation
    
    Returns:
        SimulationResult med alle beholdningsdetaljer
    """
    np.random.seed(seed)
    
    tax_rules = TaxRules(ask_rate, free_low, free_high, limit_initial, limit_increase)
    ask_rules = ASKRules(ask_initial_ceiling, ask_annual_increase)
    allocation = {
        "boligopsparing_cash": allocation_cash,
        "aktiesparekonto_ask": allocation_ask,
        "maandsopsparing_free": allocation_free
    }
    
    months = int(years * 12)
    
    mu_stock = annual_return / 12.0
    sigma_stock = volatility / np.sqrt(12.0)
    r_cash_mo = cash_interest / 12.0
    
    final_nom = np.zeros(n_paths)
    final_real = np.zeros(n_paths)
    history_nom = np.zeros((n_paths, months + 1))
    unrealized_gains = np.zeros(n_paths)
    
    # Track individuelle konti
    history_ask = np.zeros((n_paths, months + 1))
    history_free = np.zeros((n_paths, months + 1))
    history_cash = np.zeros((n_paths, months + 1))
    final_ask = np.zeros(n_paths)
    final_free = np.zeros(n_paths)
    final_cash = np.zeros(n_paths)
    final_free_after_tax = np.zeros(n_paths)
    
    discount_factor = (1 + inflation_rate) ** years
    
    for i in range(n_paths):
        portfolio = Portfolio(start_ask, start_free, start_cash)
        history_nom[i, 0] = portfolio.total_value()
        history_ask[i, 0] = portfolio.ask
        history_free[i, 0] = portfolio.free
        history_cash[i, 0] = portfolio.cash
        
        for m in range(1, months + 1):
            year_idx = (m - 1) // 12
            month_in_year = (m - 1) % 12
            
            # Årsskifte: Skat på ASK (lagerskat)
            if month_in_year == 0 and m > 1:
                portfolio.apply_ask_tax(tax_rules.ask_rate)
            
            # Indbetaling
            ask_ceiling = ask_rules.get_ceiling(year_idx)
            portfolio.contribute(monthly_contrib, allocation, ask_ceiling, ask_rules)
            
            # Markedsbevægelse
            shock = np.random.normal(mu_stock, sigma_stock)
            portfolio.apply_market_shock(shock, r_cash_mo)
            
            history_nom[i, m] = portfolio.total_value()
            history_ask[i, m] = portfolio.ask
            history_free[i, m] = portfolio.free
            history_cash[i, m] = portfolio.cash
        
        # Slutberegning - skat på frie midler beregnes KUN ved realisering
        free_tax = portfolio.calculate_free_tax_at_realization(tax_rules, years)
        net_value = portfolio.get_net_value(tax_rules, years)
        
        final_nom[i] = net_value
        final_real[i] = net_value / discount_factor
        unrealized_gains[i] = portfolio.get_unrealized_gain()
        final_ask[i] = portfolio.ask
        final_free[i] = portfolio.free
        final_cash[i] = portfolio.cash
        final_free_after_tax[i] = portfolio.free - free_tax
    
    return SimulationResult(
        final_nominal=final_nom,
        final_real=final_real,
        history_nominal=history_nom,
        unrealized_gains=unrealized_gains,
        history_ask=history_ask,
        history_free=history_free,
        history_cash=history_cash,
        final_ask=final_ask,
        final_free=final_free,
        final_cash=final_cash,
        final_free_after_tax=final_free_after_tax
    )


def calculate_tail_risk(final_real: np.ndarray, percentile: float = 5) -> Dict:
    """
    CVaR (Conditional Value at Risk) - hvad er gennemsnittet af de værste X% udfald?
    Også kaldet Expected Shortfall.
    """
    p_value = np.percentile(final_real, percentile)
    worst_cases = final_real[final_real <= p_value]
    cvar = np.mean(worst_cases) if len(worst_cases) > 0 else p_value
    
    # Value at Risk (simpelt percentil)
    var = p_value
    
    # Maximum Drawdown proxy (værste enkelt udfald)
    min_value = np.min(final_real)
    
    return {
        'var': var,
        'cvar': cvar,
        'min': min_value,
        'percentile': percentile,
        'worst_count': len(worst_cases)
    }


def calculate_metrics(history: np.ndarray, final_real: np.ndarray, 
                      monthly_contrib: float, target_goal: float) -> Dict:
    """Beregn udvidede performance metrics"""
    # Remove paths with zeros or invalid values
    valid_paths = history[~np.any(history <= 0, axis=1)]
    
    if len(valid_paths) == 0:
        return {
            'sharpe': 0.0,
            'avg_max_drawdown': 0.0,
            'total_invested': 0.0,
            'median_return': 0.0,
            'success_probability': 0.0,
            'tail_risk': {}
        }
    
    # Returns
    returns = np.diff(valid_paths, axis=1) / valid_paths[:, :-1]
    returns = returns[np.isfinite(returns).all(axis=1)]
    
    if len(returns) == 0:
        sharpe = 0.0
    else:
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        sharpe = (mean_return / std_return * np.sqrt(12)) if std_return > 0 else 0.0
    
    # Max drawdown
    running_max = np.maximum.accumulate(valid_paths, axis=1)
    drawdown = (running_max - valid_paths) / running_max
    max_dd = np.max(drawdown, axis=1)
    avg_max_drawdown = np.mean(max_dd) * 100
    
    # Total invested
    total_invested = monthly_contrib * history.shape[1]
    
    # Median return
    final_values = valid_paths[:, -1]
    median_final = np.median(final_values)
    years = history.shape[1] / 12
    median_return = ((median_final / valid_paths[0, 0]) ** (1/max(years, 1)) - 1) * 100
    
    # Success probability
    success_probability = np.mean(final_real >= target_goal) * 100
    
    # Tail risk
    tail_risk = calculate_tail_risk(final_real)
    
    return {
        'sharpe': sharpe,
        'avg_max_drawdown': avg_max_drawdown,
        'total_invested': total_invested,
        'median_return': median_return,
        'success_probability': success_probability,
        'tail_risk': tail_risk
    }


# -------------------------------------------------------------
# VISUALIZATION FUNCTIONS
# -------------------------------------------------------------
def create_trajectory_chart(history: np.ndarray, years: int, start_year: int, 
                           target_goal: float) -> go.Figure:
    """Skab formueudviklings-graf"""
    time_axis = np.linspace(start_year, start_year + years, history.shape[1])
    
    p5 = np.percentile(history, 5, axis=0)
    p10 = np.percentile(history, 10, axis=0)
    p25 = np.percentile(history, 25, axis=0)
    p50 = np.percentile(history, 50, axis=0)
    p75 = np.percentile(history, 75, axis=0)
    p90 = np.percentile(history, 90, axis=0)
    p95 = np.percentile(history, 95, axis=0)
    
    fig = go.Figure()
    
    # Confidence bands
    fig.add_trace(go.Scatter(
        x=np.concatenate([time_axis, time_axis[::-1]]),
        y=np.concatenate([p95, p5[::-1]]),
        fill='toself',
        fillcolor='rgba(0, 212, 170, 0.1)',
        line=dict(color='rgba(255,255,255,0)'),
        name='90% interval (P5-P95)',
        showlegend=True
    ))
    
    fig.add_trace(go.Scatter(
        x=np.concatenate([time_axis, time_axis[::-1]]),
        y=np.concatenate([p75, p25[::-1]]),
        fill='toself',
        fillcolor='rgba(0, 212, 170, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='50% interval (P25-P75)',
        showlegend=True
    ))
    
    # Median line
    fig.add_trace(go.Scatter(
        x=time_axis, y=p50,
        mode='lines',
        name='Median (P50)',
        line=dict(color='#00D4AA', width=3)
    ))
    
    # Target line
    fig.add_trace(go.Scatter(
        x=[time_axis[0], time_axis[-1]],
        y=[target_goal, target_goal],
        mode='lines',
        name=f'Mål: {target_goal/1e6:.1f} mio.',
        line=dict(color='#EF4444', width=2, dash='dash')
    ))
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=dict(text='Formueudvikling Over Tid', font=dict(size=20)),
        xaxis_title='År',
        yaxis_title='Formue (DKK)',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        yaxis=dict(tickformat=',.0f', gridcolor='rgba(255,255,255,0.1)'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)')
    )
    
    return fig


def create_account_breakdown_chart(sim_result: 'SimulationResult', years: int, 
                                    start_year: int) -> go.Figure:
    """Skab graf der viser udviklingen af hver konto over tid"""
    time_axis = np.linspace(start_year, start_year + years, sim_result.history_ask.shape[1])
    
    # Beregn median for hver konto
    ask_median = np.percentile(sim_result.history_ask, 50, axis=0)
    free_median = np.percentile(sim_result.history_free, 50, axis=0)
    cash_median = np.percentile(sim_result.history_cash, 50, axis=0)
    
    fig = go.Figure()
    
    # Stacked area chart
    fig.add_trace(go.Scatter(
        x=time_axis, y=cash_median,
        mode='lines',
        name='Cash (Boligopsparing)',
        line=dict(width=0),
        fillcolor='rgba(245, 158, 11, 0.6)',
        fill='tozeroy',
        stackgroup='one'
    ))
    
    fig.add_trace(go.Scatter(
        x=time_axis, y=ask_median,
        mode='lines',
        name='ASK (Aktiesparekonto)',
        line=dict(width=0),
        fillcolor='rgba(124, 58, 237, 0.6)',
        fill='tonexty',
        stackgroup='one'
    ))
    
    fig.add_trace(go.Scatter(
        x=time_axis, y=free_median,
        mode='lines',
        name='Frie Midler',
        line=dict(width=0),
        fillcolor='rgba(0, 212, 170, 0.6)',
        fill='tonexty',
        stackgroup='one'
    ))
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=dict(text='Kontobeholdninger Over Tid (Median)', font=dict(size=20)),
        xaxis_title='År',
        yaxis_title='Værdi (DKK)',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        yaxis=dict(tickformat=',.0f', gridcolor='rgba(255,255,255,0.1)'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)')
    )
    
    return fig


def create_final_breakdown_chart(sim_result: 'SimulationResult') -> go.Figure:
    """Skab søjlediagram over slutbeholdninger"""
    
    # Median værdier
    ask_median = np.median(sim_result.final_ask)
    free_median = np.median(sim_result.final_free)
    free_after_tax_median = np.median(sim_result.final_free_after_tax)
    cash_median = np.median(sim_result.final_cash)
    
    # P10 og P90 for error bars
    ask_p10, ask_p90 = np.percentile(sim_result.final_ask, [10, 90])
    free_p10, free_p90 = np.percentile(sim_result.final_free_after_tax, [10, 90])
    cash_p10, cash_p90 = np.percentile(sim_result.final_cash, [10, 90])
    
    categories = ['ASK', 'Frie Midler<br>(efter skat)', 'Cash']
    values = [ask_median, free_after_tax_median, cash_median]
    colors = ['#7C3AED', '#00D4AA', '#F59E0B']
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=categories,
        y=values,
        marker_color=colors,
        text=[f'{v/1e3:,.0f}k' for v in values],
        textposition='outside',
        error_y=dict(
            type='data',
            symmetric=False,
            array=[ask_p90 - ask_median, free_p90 - free_after_tax_median, cash_p90 - cash_median],
            arrayminus=[ask_median - ask_p10, free_after_tax_median - free_p10, cash_median - cash_p10],
            color='rgba(255,255,255,0.3)',
            thickness=2,
            width=10
        )
    ))
    
    # Tilføj annotation for skat på frie midler
    tax_on_free = np.median(sim_result.final_free) - np.median(sim_result.final_free_after_tax)
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=dict(text='Slutbeholdning per Konto (Median ± P10/P90)', font=dict(size=20)),
        xaxis_title='Konto',
        yaxis_title='Værdi (DKK)',
        yaxis=dict(tickformat=',.0f', gridcolor='rgba(255,255,255,0.1)'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        showlegend=False,
        annotations=[
            dict(
                x=1, y=free_median,
                text=f'Skat: {tax_on_free/1e3:,.0f}k',
                showarrow=True,
                arrowhead=2,
                arrowcolor='#EF4444',
                font=dict(color='#EF4444', size=10),
                ax=50, ay=-30
            )
        ]
    )
    
    return fig


def create_distribution_chart(final_nom: np.ndarray, final_real: np.ndarray, 
                             target_goal: float) -> go.Figure:
    """Skab slutformue distribution"""
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=final_nom,
        nbinsx=50,
        name='Nominel værdi',
        marker_color='rgba(124, 58, 237, 0.6)',
        opacity=0.7
    ))
    
    fig.add_trace(go.Histogram(
        x=final_real,
        nbinsx=50,
        name='Reel værdi (købekraft)',
        marker_color='rgba(0, 212, 170, 0.6)',
        opacity=0.7
    ))
    
    # Target line
    fig.add_vline(
        x=target_goal,
        line_dash="dash",
        line_color="#EF4444",
        annotation_text=f"Mål: {target_goal/1e6:.1f}M",
        annotation_position="top"
    )
    
    # Median lines
    fig.add_vline(x=np.median(final_real), line_dash="dot", line_color="#00D4AA",
                  annotation_text="Median (real)", annotation_position="bottom")
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=dict(text='Slutformue Distribution', font=dict(size=20)),
        xaxis_title='Værdi (DKK)',
        yaxis_title='Antal scenarier',
        barmode='overlay',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        xaxis=dict(tickformat=',.0f', gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
    )
    
    return fig


def create_probability_chart(config_params: Dict, scenarios: Dict[str, MarketScenario],
                            target_goal: float) -> go.Figure:
    """Skab S-kurve sandsynligheds-graf"""
    scan_amounts = np.linspace(0, 25000, 15)
    
    fig = go.Figure()
    
    for scenario_name, scenario in scenarios.items():
        probs = []
        for amt in scan_amounts:
            params = config_params.copy()
            params['monthly_contrib'] = amt
            params['annual_return'] = scenario.annual_return
            params['volatility'] = scenario.volatility
            params['n_paths'] = 200
            
            result = run_simulation(**params)
            probs.append(np.mean(result.final_real >= target_goal) * 100)
        
        fig.add_trace(go.Scatter(
            x=scan_amounts,
            y=probs,
            mode='lines+markers',
            name=scenario.name,
            line=dict(color=scenario.color, width=2),
            marker=dict(size=6)
        ))
    
    # 90% target line
    fig.add_hline(y=90, line_dash="dot", line_color="rgba(255,255,255,0.5)",
                  annotation_text="90% mål")
    
    # Current contribution line
    fig.add_vline(x=config_params['monthly_contrib'], line_dash="dash", 
                  line_color="#F59E0B", annotation_text="Dit bidrag")
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=dict(text='Sandsynlighed vs. Månedligt Bidrag', font=dict(size=20)),
        xaxis_title='Månedligt bidrag (DKK)',
        yaxis_title='Sandsynlighed for succes (%)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        xaxis=dict(tickformat=',.0f', gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(range=[0, 100], gridcolor='rgba(255,255,255,0.1)')
    )
    
    return fig


def create_scenario_comparison_chart(scenarios: Dict[str, MarketScenario],
                                     config_params: Dict, target_goal: float) -> go.Figure:
    """Skab scenarie-sammenligning"""
    results = []
    
    for name, scenario in scenarios.items():
        params = config_params.copy()
        params['annual_return'] = scenario.annual_return
        params['volatility'] = scenario.volatility
        params['n_paths'] = 500
        
        result = run_simulation(**params)
        
        results.append({
            'Scenarie': scenario.name,
            'Median': np.median(result.final_real),
            'P10': np.percentile(result.final_real, 10),
            'P90': np.percentile(result.final_real, 90),
            'Succes': np.mean(result.final_real >= target_goal) * 100,
            'color': scenario.color
        })
    
    df = pd.DataFrame(results)
    
    fig = go.Figure()
    
    for i, row in df.iterrows():
        # Error bar showing P10-P90 range
        fig.add_trace(go.Scatter(
            x=[row['Scenarie']],
            y=[row['Median']],
            error_y=dict(
                type='data',
                symmetric=False,
                array=[row['P90'] - row['Median']],
                arrayminus=[row['Median'] - row['P10']],
                color=row['color'],
                thickness=2,
                width=10
            ),
            mode='markers',
            marker=dict(size=20, color=row['color']),
            name=f"{row['Scenarie']} ({row['Succes']:.0f}% succes)",
            showlegend=True
        ))
    
    # Target line
    fig.add_hline(y=target_goal, line_dash="dash", line_color="#EF4444",
                  annotation_text=f"Mål: {target_goal/1e6:.1f}M")
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=dict(text='Scenarie Sammenligning (Median ± P10/P90)', font=dict(size=20)),
        xaxis_title='Scenarie',
        yaxis_title='Slutformue (DKK)',
        yaxis=dict(tickformat=',.0f', gridcolor='rgba(255,255,255,0.1)'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    return fig


def create_tail_risk_chart(final_real: np.ndarray, target_goal: float) -> go.Figure:
    """Visualiser tail risk / CVaR"""
    # Calculate VaR at different levels
    percentiles = [1, 5, 10, 25]
    var_values = [np.percentile(final_real, p) for p in percentiles]
    
    fig = make_subplots(rows=1, cols=2, 
                        subplot_titles=('Value at Risk (VaR)', 'Værste Udfald Distribution'))
    
    # VaR bar chart
    colors = ['#EF4444', '#F59E0B', '#FBBF24', '#10B981']
    fig.add_trace(go.Bar(
        x=[f'VaR {p}%' for p in percentiles],
        y=var_values,
        marker_color=colors,
        text=[f'{v/1e6:.2f}M' for v in var_values],
        textposition='outside',
        showlegend=False
    ), row=1, col=1)
    
    # Worst outcomes histogram
    p10 = np.percentile(final_real, 10)
    worst = final_real[final_real <= p10]
    
    fig.add_trace(go.Histogram(
        x=worst,
        nbinsx=30,
        marker_color='rgba(239, 68, 68, 0.6)',
        name='Værste 10%',
        showlegend=False
    ), row=1, col=2)
    
    # Add CVaR line
    cvar = np.mean(worst)
    fig.add_vline(x=cvar, line_dash="dash", line_color="#00D4AA", row=1, col=2,
                  annotation_text=f"CVaR: {cvar/1e6:.2f}M")
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=dict(text='Tail Risk Analyse (CVaR / Expected Shortfall)', font=dict(size=20)),
        showlegend=False
    )
    
    fig.update_yaxes(tickformat=',.0f', gridcolor='rgba(255,255,255,0.1)')
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
    
    return fig


def create_allocation_chart(allocation: Dict[str, float], monthly_contrib: float) -> go.Figure:
    """Skab allokerings pie chart"""
    labels = []
    values = []
    colors = ['#00D4AA', '#7C3AED', '#F59E0B']
    
    name_map = {
        'boligopsparing_cash': 'Boligopsparing (Cash)',
        'aktiesparekonto_ask': 'Aktiesparekonto (ASK)',
        'maandsopsparing_free': 'Frie Midler'
    }
    
    for key, weight in allocation.items():
        labels.append(name_map.get(key, key))
        values.append(weight * monthly_contrib)
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.5,
        marker_colors=colors,
        textinfo='label+percent',
        textposition='outside',
        pull=[0.02, 0.02, 0.02]
    )])
    
    fig.add_annotation(
        text=f'{monthly_contrib:,.0f}<br>kr./md',
        x=0.5, y=0.5,
        font_size=16,
        showarrow=False
    )
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=dict(text='Månedlig Allokering', font=dict(size=20)),
        showlegend=False
    )
    
    return fig


# -------------------------------------------------------------
# NEW VISUALIZATIONS: Contribution vs Growth Breakdown (#3)
# -------------------------------------------------------------
def create_contribution_growth_breakdown(
    sim_result: 'SimulationResult',
    monthly_contrib: float,
    years: int,
    start_cash: float,
    start_ask: float,
    start_free: float
) -> go.Figure:
    """
    Skab breakdown af indbetalinger vs. vækst vs. skat.
    Viser hvor meget af slutformuen kommer fra hvad.
    """
    months = years * 12
    total_contributions = monthly_contrib * months
    starting_capital = start_cash + start_ask + start_free
    total_invested = starting_capital + total_contributions
    
    # Median slutværdier
    median_final_gross = np.median(sim_result.final_ask) + np.median(sim_result.final_free) + np.median(sim_result.final_cash)
    median_final_net = np.median(sim_result.final_nominal)
    
    # Beregn komponenter
    investment_growth = median_final_gross - total_invested
    tax_paid = median_final_gross - median_final_net
    
    # Stacked bar data
    categories = ['Slutformue Breakdown']
    
    fig = go.Figure()
    
    # Starting capital
    fig.add_trace(go.Bar(
        name='Startkapital',
        x=categories,
        y=[starting_capital],
        marker_color='#6366F1',
        text=[f'{starting_capital/1e3:,.0f}k'],
        textposition='inside'
    ))
    
    # Contributions
    fig.add_trace(go.Bar(
        name='Indbetalinger',
        x=categories,
        y=[total_contributions],
        marker_color='#8B5CF6',
        text=[f'{total_contributions/1e3:,.0f}k'],
        textposition='inside'
    ))
    
    # Investment growth
    fig.add_trace(go.Bar(
        name='Investeringsafkast',
        x=categories,
        y=[investment_growth],
        marker_color='#10B981',
        text=[f'{investment_growth/1e3:,.0f}k'],
        textposition='inside'
    ))
    
    # Tax (negative effect shown as reduction)
    fig.add_trace(go.Bar(
        name='Skat (fratrukket)',
        x=categories,
        y=[-tax_paid],
        marker_color='#EF4444',
        text=[f'-{tax_paid/1e3:,.0f}k'],
        textposition='inside'
    ))
    
    fig.update_layout(
        barmode='relative',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=dict(text='Formue Breakdown: Indbetaling vs. Vækst', font=dict(size=18)),
        yaxis=dict(tickformat=',.0f', gridcolor='rgba(255,255,255,0.1)', title='DKK'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        height=400
    )
    
    return fig


def create_waterfall_chart(
    sim_result: 'SimulationResult',
    monthly_contrib: float,
    years: int,
    start_cash: float,
    start_ask: float,
    start_free: float
) -> go.Figure:
    """
    Skab waterfall chart der viser flow fra start til slut.
    """
    months = years * 12
    total_contributions = monthly_contrib * months
    starting_capital = start_cash + start_ask + start_free
    
    # Beregn komponenter
    median_gross = np.median(sim_result.final_ask) + np.median(sim_result.final_free) + np.median(sim_result.final_cash)
    median_net = np.median(sim_result.final_nominal)
    
    investment_growth = median_gross - starting_capital - total_contributions
    tax_paid = median_gross - median_net
    
    fig = go.Figure(go.Waterfall(
        name="Formue Flow",
        orientation="v",
        measure=["absolute", "relative", "relative", "relative", "total"],
        x=["Startkapital", "Indbetalinger", "Afkast", "Skat", "Slutformue"],
        textposition="outside",
        text=[
            f"{starting_capital/1e3:,.0f}k",
            f"+{total_contributions/1e3:,.0f}k",
            f"+{investment_growth/1e3:,.0f}k",
            f"-{tax_paid/1e3:,.0f}k",
            f"{median_net/1e3:,.0f}k"
        ],
        y=[starting_capital, total_contributions, investment_growth, -tax_paid, 0],
        connector={"line": {"color": "rgba(255,255,255,0.3)"}},
        increasing={"marker": {"color": "#10B981"}},
        decreasing={"marker": {"color": "#EF4444"}},
        totals={"marker": {"color": "#00D4AA"}}
    ))
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=dict(text='Waterfall: Fra Start til Slut', font=dict(size=18)),
        yaxis=dict(tickformat=',.0f', gridcolor='rgba(255,255,255,0.1)', title='DKK'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        showlegend=False,
        height=400
    )
    
    return fig


def create_growth_ratio_over_time(
    sim_result: 'SimulationResult',
    monthly_contrib: float,
    years: int,
    start_year: int,
    start_cash: float,
    start_ask: float,
    start_free: float
) -> go.Figure:
    """
    Vis hvordan forholdet mellem indbetalinger og vækst ændrer sig over tid.
    Illustrerer "renters rente" effekten.
    """
    months = sim_result.history_nominal.shape[1]
    time_axis = np.linspace(start_year, start_year + years, months)
    starting_capital = start_cash + start_ask + start_free
    
    # Beregn kumulerede indbetalinger over tid
    cumulative_contributions = np.array([starting_capital + monthly_contrib * m for m in range(months)])
    
    # Median porteføljeværdi over tid
    median_portfolio = np.percentile(sim_result.history_nominal, 50, axis=0)
    
    # Vækst = Portefølje - Indbetalinger
    growth_component = median_portfolio - cumulative_contributions
    
    fig = go.Figure()
    
    # Stacked area
    fig.add_trace(go.Scatter(
        x=time_axis,
        y=cumulative_contributions,
        mode='lines',
        name='Indbetalinger',
        fill='tozeroy',
        fillcolor='rgba(139, 92, 246, 0.5)',
        line=dict(color='#8B5CF6', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=time_axis,
        y=median_portfolio,
        mode='lines',
        name='Total værdi (median)',
        fill='tonexty',
        fillcolor='rgba(16, 185, 129, 0.5)',
        line=dict(color='#10B981', width=2)
    ))
    
    # Annotations for key points
    mid_idx = months // 2
    end_idx = months - 1
    
    mid_contrib_pct = (cumulative_contributions[mid_idx] / median_portfolio[mid_idx]) * 100
    end_contrib_pct = (cumulative_contributions[end_idx] / median_portfolio[end_idx]) * 100
    
    fig.add_annotation(
        x=time_axis[mid_idx], y=median_portfolio[mid_idx],
        text=f"Indbetalinger: {mid_contrib_pct:.0f}%",
        showarrow=True, arrowhead=2, ax=50, ay=-40,
        font=dict(size=10, color='white')
    )
    
    fig.add_annotation(
        x=time_axis[end_idx], y=median_portfolio[end_idx],
        text=f"Indbetalinger: {end_contrib_pct:.0f}%",
        showarrow=True, arrowhead=2, ax=-50, ay=-40,
        font=dict(size=10, color='white')
    )
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=dict(text='Indbetalinger vs. Vækst Over Tid', font=dict(size=18)),
        xaxis_title='År',
        yaxis_title='Værdi (DKK)',
        yaxis=dict(tickformat=',.0f', gridcolor='rgba(255,255,255,0.1)'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        height=400
    )
    
    return fig


# -------------------------------------------------------------
# NEW VISUALIZATIONS: Sequence of Returns Risk (#1)
# -------------------------------------------------------------
@st.cache_data(show_spinner=False)
def run_deterministic_sequence_simulation(
    years: int,
    monthly_contrib: float,
    early_bear_returns: List[float],
    late_bear_returns: List[float],
    consistent_returns: List[float],
    inflation_rate: float,
    cash_interest: float,
    ask_rate: float,
    free_low: float,
    free_high: float,
    limit_initial: float,
    limit_increase: float,
    ask_initial_ceiling: float,
    ask_annual_increase: float,
    allocation_cash: float,
    allocation_ask: float,
    allocation_free: float,
    start_cash: float,
    start_ask: float,
    start_free: float
) -> Dict[str, np.ndarray]:
    """
    Kør deterministiske simuleringer med foruddefinerede afkastsekvenser.
    Returnerer historik for hver sekvenstype.
    """
    tax_rules = TaxRules(ask_rate, free_low, free_high, limit_initial, limit_increase)
    ask_rules = ASKRules(ask_initial_ceiling, ask_annual_increase)
    allocation = {
        "boligopsparing_cash": allocation_cash,
        "aktiesparekonto_ask": allocation_ask,
        "maandsopsparing_free": allocation_free
    }
    
    months = years * 12
    r_cash_mo = cash_interest / 12.0
    discount_factor = (1 + inflation_rate) ** years
    
    results = {}
    sequences = {
        'early_bear': early_bear_returns,
        'late_bear': late_bear_returns,
        'consistent': consistent_returns
    }
    
    for seq_name, yearly_returns in sequences.items():
        # Convert yearly returns to monthly
        monthly_returns = []
        for yearly_ret in yearly_returns:
            monthly_ret = (1 + yearly_ret) ** (1/12) - 1
            monthly_returns.extend([monthly_ret] * 12)
        
        # Ensure we have enough returns for all months
        while len(monthly_returns) < months:
            monthly_returns.append(monthly_returns[-1] if monthly_returns else 0)
        
        portfolio = Portfolio(start_ask, start_free, start_cash)
        history = np.zeros(months + 1)
        history[0] = portfolio.total_value()
        
        for m in range(1, months + 1):
            year_idx = (m - 1) // 12
            month_in_year = (m - 1) % 12
            
            # Årsskifte: Skat på ASK
            if month_in_year == 0 and m > 1:
                portfolio.apply_ask_tax(tax_rules.ask_rate)
            
            # Indbetaling
            ask_ceiling = ask_rules.get_ceiling(year_idx)
            portfolio.contribute(monthly_contrib, allocation, ask_ceiling, ask_rules)
            
            # Markedsbevægelse (deterministisk)
            shock = monthly_returns[m - 1]
            portfolio.apply_market_shock(shock, r_cash_mo)
            
            history[m] = portfolio.total_value()
        
        # Slutberegning
        net_value = portfolio.get_net_value(tax_rules, years)
        real_value = net_value / discount_factor
        
        results[seq_name] = {
            'history': history,
            'final_nominal': net_value,
            'final_real': real_value,
            'yearly_returns': yearly_returns
        }
    
    return results


def create_sequence_comparison_chart(
    sequence_results: Dict,
    years: int,
    start_year: int,
    target_goal: float
) -> go.Figure:
    """
    Skab sammenligning af de tre afkastsekvenser.
    """
    months = len(sequence_results['consistent']['history'])
    time_axis = np.linspace(start_year, start_year + years, months)
    
    colors = {
        'early_bear': '#EF4444',  # Rød
        'late_bear': '#F59E0B',   # Orange
        'consistent': '#10B981'   # Grøn
    }
    
    names = {
        'early_bear': 'Tidligt bear market (tab først)',
        'late_bear': 'Sent bear market (tab sidst)',
        'consistent': 'Konstant afkast'
    }
    
    fig = go.Figure()
    
    for seq_name, data in sequence_results.items():
        fig.add_trace(go.Scatter(
            x=time_axis,
            y=data['history'],
            mode='lines',
            name=names[seq_name],
            line=dict(color=colors[seq_name], width=3)
        ))
    
    # Target line
    fig.add_hline(
        y=target_goal,
        line_dash="dash",
        line_color="rgba(255,255,255,0.5)",
        annotation_text=f"Mål: {target_goal/1e6:.1f}M"
    )
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=dict(text='Timing Risiko: Samme Gennemsnit, Forskellig Sekvens', font=dict(size=18)),
        xaxis_title='År',
        yaxis_title='Porteføljeværdi (DKK)',
        yaxis=dict(tickformat=',.0f', gridcolor='rgba(255,255,255,0.1)'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        hovermode='x unified',
        height=450
    )
    
    return fig


def create_sequence_final_comparison(
    sequence_results: Dict,
    target_goal: float
) -> go.Figure:
    """
    Bar chart der viser slutværdier for hver sekvens.
    """
    names = {
        'early_bear': 'Tidligt Bear',
        'late_bear': 'Sent Bear',
        'consistent': 'Konstant'
    }
    
    colors = {
        'early_bear': '#EF4444',
        'late_bear': '#F59E0B',
        'consistent': '#10B981'
    }
    
    categories = [names[k] for k in ['early_bear', 'consistent', 'late_bear']]
    values = [sequence_results[k]['final_real'] for k in ['early_bear', 'consistent', 'late_bear']]
    bar_colors = [colors[k] for k in ['early_bear', 'consistent', 'late_bear']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=categories,
        y=values,
        marker_color=bar_colors,
        text=[f'{v/1e6:.2f}M' for v in values],
        textposition='outside'
    ))
    
    # Target line
    fig.add_hline(
        y=target_goal,
        line_dash="dash",
        line_color="rgba(255,255,255,0.5)",
        annotation_text=f"Mål: {target_goal/1e6:.1f}M"
    )
    
    # Calculate differences
    consistent_val = sequence_results['consistent']['final_real']
    early_diff = ((sequence_results['early_bear']['final_real'] / consistent_val) - 1) * 100
    late_diff = ((sequence_results['late_bear']['final_real'] / consistent_val) - 1) * 100
    
    fig.add_annotation(
        x='Tidligt Bear', y=values[0],
        text=f"{early_diff:+.1f}% vs konstant",
        showarrow=False, yshift=40,
        font=dict(size=10, color='#EF4444')
    )
    
    fig.add_annotation(
        x='Sent Bear', y=values[2],
        text=f"{late_diff:+.1f}% vs konstant",
        showarrow=False, yshift=40,
        font=dict(size=10, color='#F59E0B')
    )
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=dict(text='Slutværdi per Sekvens (Real Købekraft)', font=dict(size=18)),
        yaxis=dict(tickformat=',.0f', gridcolor='rgba(255,255,255,0.1)', title='DKK'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        showlegend=False,
        height=400
    )
    
    return fig


def create_yearly_returns_chart(sequence_results: Dict, years: int, start_year: int) -> go.Figure:
    """
    Vis de årlige afkast for hver sekvens.
    """
    year_labels = [str(start_year + i) for i in range(years)]
    
    fig = go.Figure()
    
    colors = {
        'early_bear': '#EF4444',
        'late_bear': '#F59E0B',
        'consistent': '#10B981'
    }
    
    names = {
        'early_bear': 'Tidligt Bear',
        'late_bear': 'Sent Bear',
        'consistent': 'Konstant'
    }
    
    for seq_name in ['early_bear', 'consistent', 'late_bear']:
        returns = sequence_results[seq_name]['yearly_returns'][:years]
        fig.add_trace(go.Scatter(
            x=year_labels,
            y=[r * 100 for r in returns],
            mode='lines+markers',
            name=names[seq_name],
            line=dict(color=colors[seq_name], width=2),
            marker=dict(size=8)
        ))
    
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.3)")
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=dict(text='Årlige Afkast per Sekvens', font=dict(size=18)),
        xaxis_title='År',
        yaxis_title='Afkast (%)',
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        height=350
    )
    
    return fig


# -------------------------------------------------------------
# NEW VISUALIZATIONS: Spaghetti Chart (#2)
# -------------------------------------------------------------
def create_spaghetti_chart(
    history: np.ndarray,
    years: int,
    start_year: int,
    target_goal: float,
    n_paths_to_show: int = 50
) -> go.Figure:
    """
    Skab spaghetti chart med individuelle Monte Carlo stier.
    Viser volatilitet og usikkerhed visuelt.
    """
    time_axis = np.linspace(start_year, start_year + years, history.shape[1])
    n_paths = history.shape[0]
    
    # Vælg tilfældige stier at vise
    np.random.seed(42)  # For reproducerbarhed
    indices = np.random.choice(n_paths, min(n_paths_to_show, n_paths), replace=False)
    
    fig = go.Figure()
    
    # Individuelle stier (meget transparent)
    for i, idx in enumerate(indices):
        fig.add_trace(go.Scatter(
            x=time_axis,
            y=history[idx],
            mode='lines',
            line=dict(color='rgba(0, 212, 170, 0.08)', width=1),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Percentil bands
    p10 = np.percentile(history, 10, axis=0)
    p90 = np.percentile(history, 90, axis=0)
    
    fig.add_trace(go.Scatter(
        x=np.concatenate([time_axis, time_axis[::-1]]),
        y=np.concatenate([p90, p10[::-1]]),
        fill='toself',
        fillcolor='rgba(124, 58, 237, 0.15)',
        line=dict(color='rgba(255,255,255,0)'),
        name='80% interval (P10-P90)',
        showlegend=True
    ))
    
    # Median (bold)
    p50 = np.percentile(history, 50, axis=0)
    fig.add_trace(go.Scatter(
        x=time_axis,
        y=p50,
        mode='lines',
        name='Median',
        line=dict(color='#00D4AA', width=4)
    ))
    
    # Target line
    fig.add_hline(
        y=target_goal,
        line_dash="dash",
        line_color="#EF4444",
        annotation_text=f"Mål: {target_goal/1e6:.1f}M"
    )
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=dict(text=f'Individuelle Simuleringer ({n_paths_to_show} stier)', font=dict(size=18)),
        xaxis_title='År',
        yaxis_title='Porteføljeværdi (DKK)',
        yaxis=dict(tickformat=',.0f', gridcolor='rgba(255,255,255,0.1)'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        hovermode='x unified',
        height=500
    )
    
    return fig


# -------------------------------------------------------------
# NEW VISUALIZATIONS: Historical Backtesting (#4)
# -------------------------------------------------------------
@st.cache_data(show_spinner=False)
def run_historical_backtest(
    start_year_backtest: int,
    years: int,
    monthly_contrib: float,
    inflation_rate: float,
    cash_interest: float,
    ask_rate: float,
    free_low: float,
    free_high: float,
    limit_initial: float,
    limit_increase: float,
    ask_initial_ceiling: float,
    ask_annual_increase: float,
    allocation_cash: float,
    allocation_ask: float,
    allocation_free: float,
    start_cash: float,
    start_ask: float,
    start_free: float
) -> Optional[Dict]:
    """
    Kør backtest med historiske S&P 500 afkast.
    Returnerer None hvis ikke nok historiske data.
    """
    # Check om vi har nok historiske data
    required_years = list(range(start_year_backtest, start_year_backtest + years))
    available_years = [y for y in required_years if y in HISTORICAL_RETURNS]
    
    if len(available_years) < years:
        return None
    
    # Hent historiske afkast
    yearly_returns = [HISTORICAL_RETURNS[y] for y in required_years]
    
    tax_rules = TaxRules(ask_rate, free_low, free_high, limit_initial, limit_increase)
    ask_rules = ASKRules(ask_initial_ceiling, ask_annual_increase)
    allocation = {
        "boligopsparing_cash": allocation_cash,
        "aktiesparekonto_ask": allocation_ask,
        "maandsopsparing_free": allocation_free
    }
    
    months = years * 12
    r_cash_mo = cash_interest / 12.0
    discount_factor = (1 + inflation_rate) ** years
    
    # Convert til månedlige afkast
    monthly_returns = []
    for yearly_ret in yearly_returns:
        monthly_ret = (1 + yearly_ret) ** (1/12) - 1
        monthly_returns.extend([monthly_ret] * 12)
    
    portfolio = Portfolio(start_ask, start_free, start_cash)
    history = np.zeros(months + 1)
    history[0] = portfolio.total_value()
    
    for m in range(1, months + 1):
        year_idx = (m - 1) // 12
        month_in_year = (m - 1) % 12
        
        if month_in_year == 0 and m > 1:
            portfolio.apply_ask_tax(tax_rules.ask_rate)
        
        ask_ceiling = ask_rules.get_ceiling(year_idx)
        portfolio.contribute(monthly_contrib, allocation, ask_ceiling, ask_rules)
        
        shock = monthly_returns[m - 1]
        portfolio.apply_market_shock(shock, r_cash_mo)
        
        history[m] = portfolio.total_value()
    
    net_value = portfolio.get_net_value(tax_rules, years)
    real_value = net_value / discount_factor
    
    return {
        'history': history,
        'final_nominal': net_value,
        'final_real': real_value,
        'yearly_returns': yearly_returns,
        'start_year': start_year_backtest,
        'avg_return': np.mean(yearly_returns) * 100
    }


def create_backtest_chart(
    backtest_result: Dict,
    mc_history: np.ndarray,
    years: int,
    target_goal: float
) -> go.Figure:
    """
    Sammenlign historisk backtest med Monte Carlo percentiler.
    """
    start_year = backtest_result['start_year']
    time_axis = np.linspace(start_year, start_year + years, len(backtest_result['history']))
    
    # MC percentiles (justeret til samme tidsakse)
    p10 = np.percentile(mc_history, 10, axis=0)
    p50 = np.percentile(mc_history, 50, axis=0)
    p90 = np.percentile(mc_history, 90, axis=0)
    
    fig = go.Figure()
    
    # MC confidence bands
    fig.add_trace(go.Scatter(
        x=np.concatenate([time_axis, time_axis[::-1]]),
        y=np.concatenate([p90, p10[::-1]]),
        fill='toself',
        fillcolor='rgba(124, 58, 237, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Monte Carlo 80% interval',
        showlegend=True
    ))
    
    # MC median
    fig.add_trace(go.Scatter(
        x=time_axis,
        y=p50,
        mode='lines',
        name='Monte Carlo Median',
        line=dict(color='#7C3AED', width=2, dash='dot')
    ))
    
    # Historical backtest (solid, highlighted)
    fig.add_trace(go.Scatter(
        x=time_axis,
        y=backtest_result['history'],
        mode='lines',
        name=f'Historisk ({backtest_result["start_year"]}-{backtest_result["start_year"]+years})',
        line=dict(color='#00D4AA', width=3)
    ))
    
    # Target
    fig.add_hline(
        y=target_goal,
        line_dash="dash",
        line_color="#EF4444",
        annotation_text=f"Mål: {target_goal/1e6:.1f}M"
    )
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=dict(text=f'Historisk Backtest vs. Monte Carlo Prediktion', font=dict(size=18)),
        xaxis_title='År',
        yaxis_title='Porteføljeværdi (DKK)',
        yaxis=dict(tickformat=',.0f', gridcolor='rgba(255,255,255,0.1)'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        height=450
    )
    
    return fig


# -------------------------------------------------------------
# NEW VISUALIZATIONS: Interactive Scenario Comparison (#6)
# -------------------------------------------------------------
def run_comparison_simulation(
    params: Dict,
    annual_return: float,
    volatility: float,
    monthly_contrib: float,
    n_paths: int = 300
) -> 'SimulationResult':
    """
    Kør simulation for scenarie sammenligning.
    """
    sim_params = params.copy()
    sim_params['annual_return'] = annual_return
    sim_params['volatility'] = volatility
    sim_params['monthly_contrib'] = monthly_contrib
    sim_params['n_paths'] = n_paths
    
    return run_simulation(**sim_params)


def create_comparison_trajectories(
    result_a: 'SimulationResult',
    result_b: 'SimulationResult',
    years: int,
    start_year: int,
    name_a: str,
    name_b: str
) -> go.Figure:
    """
    Side-by-side trajectory sammenligning.
    """
    time_axis = np.linspace(start_year, start_year + years, result_a.history_nominal.shape[1])
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(name_a, name_b),
        horizontal_spacing=0.1
    )
    
    for col, (result, name) in enumerate([(result_a, name_a), (result_b, name_b)], 1):
        p10 = np.percentile(result.history_nominal, 10, axis=0)
        p50 = np.percentile(result.history_nominal, 50, axis=0)
        p90 = np.percentile(result.history_nominal, 90, axis=0)
        
        color = '#00D4AA' if col == 1 else '#7C3AED'
        
        # Confidence band
        fig.add_trace(go.Scatter(
            x=np.concatenate([time_axis, time_axis[::-1]]),
            y=np.concatenate([p90, p10[::-1]]),
            fill='toself',
            fillcolor=f'rgba({124 if col == 2 else 0}, {58 if col == 2 else 212}, {237 if col == 2 else 170}, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False
        ), row=1, col=col)
        
        # Median
        fig.add_trace(go.Scatter(
            x=time_axis,
            y=p50,
            mode='lines',
            line=dict(color=color, width=3),
            name=f'{name} Median',
            showlegend=True
        ), row=1, col=col)
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=dict(text='Scenarie Sammenligning: Trajektorier', font=dict(size=18)),
        height=400
    )
    
    fig.update_yaxes(tickformat=',.0f', gridcolor='rgba(255,255,255,0.1)')
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)', title_text='År')
    
    return fig


def create_comparison_distributions(
    result_a: 'SimulationResult',
    result_b: 'SimulationResult',
    target_goal: float,
    name_a: str,
    name_b: str
) -> go.Figure:
    """
    Side-by-side distribution sammenligning.
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(name_a, name_b),
        horizontal_spacing=0.1
    )
    
    for col, (result, name) in enumerate([(result_a, name_a), (result_b, name_b)], 1):
        color = '#00D4AA' if col == 1 else '#7C3AED'
        
        fig.add_trace(go.Histogram(
            x=result.final_real,
            nbinsx=40,
            marker_color=color,
            opacity=0.7,
            showlegend=False
        ), row=1, col=col)
        
        # Target line
        fig.add_vline(
            x=target_goal,
            line_dash="dash",
            line_color="#EF4444",
            row=1, col=col
        )
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=dict(text='Scenarie Sammenligning: Slutfordeling', font=dict(size=18)),
        height=350
    )
    
    fig.update_xaxes(tickformat=',.0f', gridcolor='rgba(255,255,255,0.1)', title_text='Slutværdi (DKK)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
    
    return fig


# -------------------------------------------------------------
# SIDEBAR CONFIGURATION
# -------------------------------------------------------------
def render_sidebar():
    """Render sidebar med konfiguration - bruger defaults fra config.py"""
    st.sidebar.markdown("## Konfiguration")
    
    # Beregn default allokering som procent
    default_alloc_cash = int(DEFAULT_ALLOCATION["boligopsparing_cash"] * 100)
    default_alloc_ask = int(DEFAULT_ALLOCATION["aktiesparekonto_ask"] * 100)
    
    with st.sidebar.expander("Målsætning", expanded=True):
        target_goal = st.number_input(
            "Mål (DKK)", 
            min_value=100_000, 
            max_value=10_000_000, 
            value=DEFAULT_TARGET_GOAL,
            step=100_000,
            format="%d"
        )
        years = st.slider("Tidshorisont (år)", 1, 30, DEFAULT_YEARS)
        start_year = st.number_input("Startår", 2024, 2050, DEFAULT_START_YEAR)
    
    with st.sidebar.expander("Budget", expanded=True):
        income = st.number_input("Månedlig nettoindkomst", 0, 100_000, int(DEFAULT_INCOME_NET_MONTHLY))
        expenses = st.number_input("Månedlige udgifter", 0, 100_000, int(DEFAULT_EXPENSES_MONTHLY))
        monthly_contrib = income - expenses
        st.metric("Månedligt bidrag", f"{monthly_contrib:,.0f} kr.")
    
    with st.sidebar.expander("Allokering", expanded=False):
        alloc_cash = st.slider("Boligopsparing (Cash) %", 0, 100, default_alloc_cash)
        alloc_ask = st.slider("Aktiesparekonto (ASK) %", 0, 100, default_alloc_ask)
        alloc_free = 100 - alloc_cash - alloc_ask
        st.info(f"Frie midler: {alloc_free}%")
        
        if alloc_cash + alloc_ask > 100:
            st.error("Allokering overstiger 100%!")
    
    with st.sidebar.expander("Marked & Økonomi", expanded=True):
        annual_return = st.slider("Årligt afkast (%)", 0.0, 15.0, 7.0, 0.5) / 100
        st.caption("Standard: 7% (historisk gennemsnit)")
        volatility = st.slider("Volatilitet (%)", 5.0, 40.0, 20.0, 1.0) / 100
        st.caption("Standard: 20% (historisk aktiemarkeds-volatilitet)")
        inflation = st.slider("Inflation (%)", 0.0, 10.0, DEFAULT_INFLATION_RATE * 100, 0.5) / 100
        cash_interest = st.slider("Kontantrente (%)", 0.0, 5.0, DEFAULT_CASH_INTEREST * 100, 0.25) / 100
    
    with st.sidebar.expander("Startbeholdning", expanded=False):
        start_cash = st.number_input("Cash konti total", 0, 1_000_000, int(DEFAULT_START_CASH))
        start_ask = st.number_input("ASK beholdning", 0, 1_000_000, int(DEFAULT_START_ASK))
        start_free = st.number_input("Frie midler", 0, 1_000_000, int(DEFAULT_START_FREE))
    
    with st.sidebar.expander("Simulation", expanded=False):
        n_paths = st.slider("Antal simuleringer", 100, 5000, DEFAULT_MC_PATHS, 100)
        seed = st.number_input("Random seed", 1, 9999, DEFAULT_SEED)
    
    return {
        'target_goal': target_goal,
        'years': years,
        'start_year': start_year,
        'monthly_contrib': monthly_contrib,
        'income': income,
        'expenses': expenses,
        'allocation_cash': alloc_cash / 100,
        'allocation_ask': alloc_ask / 100,
        'allocation_free': alloc_free / 100,
        'inflation_rate': inflation,
        'cash_interest': cash_interest,
        'start_cash': start_cash,
        'start_ask': start_ask,
        'start_free': start_free,
        'n_paths': n_paths,
        'seed': seed,
        'volatility': volatility,
        'annual_return': annual_return,
        # Tax rules from config
        'ask_rate': DEFAULT_ASK_TAX_RATE,
        'free_low': DEFAULT_FREE_TAX_LOW,
        'free_high': DEFAULT_FREE_TAX_HIGH,
        'limit_initial': DEFAULT_TAX_LIMIT_INITIAL,
        'limit_increase': DEFAULT_TAX_LIMIT_INCREASE,
        # ASK rules from config
        'ask_initial_ceiling': DEFAULT_ASK_CEILING,
        'ask_annual_increase': DEFAULT_ASK_ANNUAL_INCREASE
    }


# -------------------------------------------------------------
# MAIN APP
# -------------------------------------------------------------
def main():
    """Hovedapplikation"""
    
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <h1 style="font-size: 3rem; background: linear-gradient(135deg, #00D4AA 0%, #7C3AED 100%); 
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            InvestMål Analyzer
        </h1>
        <p style="color: #9CA3AF; font-size: 1.2rem;">
            Monte Carlo simulation til analyse af investeringsmål
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    config = render_sidebar()
    
    # Define market scenarios
    scenarios = {
        'bull': MarketScenario("Bull Market", 0.10, 0.15, "Stærk økonomi", "#10B981"),
        'normal': MarketScenario("Normal", 0.07, 0.20, "Historisk gennemsnit", "#00D4AA"),
        'bear': MarketScenario("Bear Market", 0.04, 0.25, "Svag økonomi", "#F59E0B"),
        'crash': MarketScenario("Krise", 0.02, 0.30, "Økonomisk krise", "#EF4444")
    }
    
    # Prepare simulation parameters
    sim_params = {
        'target_goal': config['target_goal'],
        'years': config['years'],
        'monthly_contrib': config['monthly_contrib'],
        'annual_return': config['annual_return'],
        'volatility': config['volatility'],
        'inflation_rate': config['inflation_rate'],
        'cash_interest': config['cash_interest'],
        'ask_rate': config['ask_rate'],
        'free_low': config['free_low'],
        'free_high': config['free_high'],
        'limit_initial': config['limit_initial'],
        'limit_increase': config['limit_increase'],
        'ask_initial_ceiling': config['ask_initial_ceiling'],
        'ask_annual_increase': config['ask_annual_increase'],
        'allocation_cash': config['allocation_cash'],
        'allocation_ask': config['allocation_ask'],
        'allocation_free': config['allocation_free'],
        'start_cash': config['start_cash'],
        'start_ask': config['start_ask'],
        'start_free': config['start_free'],
        'n_paths': config['n_paths'],
        'seed': config['seed']
    }
    
    # Run simulation
    with st.spinner("Kører Monte Carlo simulation..."):
        sim_result = run_simulation(**sim_params)
        metrics = calculate_metrics(sim_result.history_nominal, sim_result.final_real, 
                                   config['monthly_contrib'], config['target_goal'])
    
    # Udpak for nemheds skyld
    final_nom = sim_result.final_nominal
    final_real = sim_result.final_real
    history = sim_result.history_nominal
    
    # Calculate monthly contribution needed for 90% success
    def find_required_contribution(target_prob: float = 0.90) -> float:
        """Binary search to find required monthly contribution for target success probability"""
        low, high = 0, 50000
        best = high
        for _ in range(12):  # 12 iterations gives good precision
            mid = (low + high) / 2
            test_params = sim_params.copy()
            test_params['monthly_contrib'] = mid
            test_params['n_paths'] = 200  # Fewer paths for speed
            test_result = run_simulation(**test_params)
            prob = np.mean(test_result.final_real >= config['target_goal'])
            if prob >= target_prob:
                best = mid
                high = mid
            else:
                low = mid
        return best
    
    # Calculate time to goal at median growth
    def calculate_time_to_goal() -> float:
        """Estimate months to reach goal at median growth rate"""
        median_final = np.median(final_real)
        if median_final <= 0:
            return float('inf')
        # Use compound growth formula to estimate time
        start_value = config['start_cash'] + config['start_ask'] + config['start_free']
        monthly_contrib = config['monthly_contrib']
        monthly_return = (1 + scenarios['normal'].annual_return) ** (1/12) - 1
        
        # Simulate month by month until goal is reached
        value = start_value
        for month in range(1, 12 * 50 + 1):  # Max 50 years
            value = value * (1 + monthly_return) + monthly_contrib
            if value >= config['target_goal']:
                return month
        return float('inf')
    
    # Calculate required contribution (cached)
    required_contrib_90 = find_required_contribution(0.90)
    months_to_goal = calculate_time_to_goal()
    years_to_goal = months_to_goal / 12
    
    # Key Metrics Row
    st.markdown('<div class="section-header">Nøgletal</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        success_color = "✓" if metrics['success_probability'] >= 90 else ("~" if metrics['success_probability'] >= 50 else "✗")
        st.metric(
            label="Succesrate",
            value=f"{metrics['success_probability']:.1f}%",
            delta=f"{success_color} {'Høj' if metrics['success_probability'] >= 90 else 'Moderat' if metrics['success_probability'] >= 50 else 'Lav'}"
        )
    
    with col2:
        st.metric(
            label="Median Slutformue",
            value=f"{np.median(final_real)/1e6:.2f} M",
            delta=f"{((np.median(final_real)/config['target_goal'])-1)*100:+.0f}% vs mål"
        )
    
    with col3:
        if years_to_goal < 100:
            st.metric(
                label="Tid til Mål",
                value=f"{years_to_goal:.1f} år",
                delta=f"{int(months_to_goal)} måneder"
            )
        else:
            st.metric(
                label="Tid til Mål",
                value="50+ år",
                delta="Mål ikke nåeligt"
            )
    
    with col4:
        contrib_diff = required_contrib_90 - config['monthly_contrib']
        st.metric(
            label="Bidrag for 90%",
            value=f"{required_contrib_90:,.0f} kr.",
            delta=f"{contrib_diff:+,.0f} kr. vs nu" if abs(contrib_diff) > 100 else "✓ Tilstrækkeligt"
        )
    
    with col5:
        st.metric(
            label="Max Drawdown",
            value=f"{metrics['avg_max_drawdown']:.1f}%",
            delta="Gns. værste fald"
        )
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Formueudvikling", 
        "Konti",
        "Vækst Analyse",
        "Sandsynlighed", 
        "Historisk Test"
    ])
    
    # -------------------------------------------------------------
    # TAB 1: Formueudvikling
    # -------------------------------------------------------------
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig_trajectory = create_trajectory_chart(
                history, config['years'], config['start_year'], config['target_goal']
            )
            st.plotly_chart(fig_trajectory, use_container_width=True)
        
        with col2:
            fig_dist = create_distribution_chart(final_nom, final_real, config['target_goal'])
            st.plotly_chart(fig_dist, use_container_width=True)
    
    # -------------------------------------------------------------
    # TAB 2: Konti (Existing)
    # -------------------------------------------------------------
    with tab2:
        st.markdown("### Beholdning per Konto")
        
        # Vis slutværdier som metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            ask_median = np.median(sim_result.final_ask)
            st.metric(
                label="ASK (Aktiesparekonto)",
                value=f"{ask_median/1e3:,.0f}k",
                delta=f"Allerede beskattet (17% lagerskat)"
            )
        
        with col2:
            free_median = np.median(sim_result.final_free)
            free_after_tax = np.median(sim_result.final_free_after_tax)
            tax_amount = free_median - free_after_tax
            st.metric(
                label="Frie Midler (før skat)",
                value=f"{free_median/1e3:,.0f}k",
                delta=f"-{tax_amount/1e3:,.0f}k skat ved salg"
            )
        
        with col3:
            st.metric(
                label="Frie Midler (efter skat)",
                value=f"{free_after_tax/1e3:,.0f}k",
                delta="Realiseret værdi"
            )
        
        with col4:
            cash_median = np.median(sim_result.final_cash)
            st.metric(
                label="Cash (Boligopsparing)",
                value=f"{cash_median/1e3:,.0f}k",
                delta="Ingen skat"
            )
        
        # Grafer
        col1, col2 = st.columns([1, 1])
        
        with col1:
            fig_account_time = create_account_breakdown_chart(
                sim_result, config['years'], config['start_year']
            )
            st.plotly_chart(fig_account_time, use_container_width=True)
        
        with col2:
            fig_final_breakdown = create_final_breakdown_chart(sim_result)
            st.plotly_chart(fig_final_breakdown, use_container_width=True)
        
        # Detalje-tabel
        st.markdown("### Statistik per Konto")
        
        stats_data = {
            'Konto': ['ASK', 'Frie Midler (før skat)', 'Frie Midler (efter skat)', 'Cash', 'TOTAL'],
            'Median': [
                f"{np.median(sim_result.final_ask):,.0f}",
                f"{np.median(sim_result.final_free):,.0f}",
                f"{np.median(sim_result.final_free_after_tax):,.0f}",
                f"{np.median(sim_result.final_cash):,.0f}",
                f"{np.median(sim_result.final_nominal):,.0f}"
            ],
            'P10 (Værst)': [
                f"{np.percentile(sim_result.final_ask, 10):,.0f}",
                f"{np.percentile(sim_result.final_free, 10):,.0f}",
                f"{np.percentile(sim_result.final_free_after_tax, 10):,.0f}",
                f"{np.percentile(sim_result.final_cash, 10):,.0f}",
                f"{np.percentile(sim_result.final_nominal, 10):,.0f}"
            ],
            'P90 (Bedst)': [
                f"{np.percentile(sim_result.final_ask, 90):,.0f}",
                f"{np.percentile(sim_result.final_free, 90):,.0f}",
                f"{np.percentile(sim_result.final_free_after_tax, 90):,.0f}",
                f"{np.percentile(sim_result.final_cash, 90):,.0f}",
                f"{np.percentile(sim_result.final_nominal, 90):,.0f}"
            ]
        }
        
        st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)
    
    # -------------------------------------------------------------
    # TAB 3: Contribution vs Growth Breakdown (NEW)
    # -------------------------------------------------------------
    with tab3:
        st.markdown("### Indbetalinger vs. Investeringsafkast")
        
        # Key metrics for contribution vs growth
        months = config['years'] * 12
        total_contributions = config['monthly_contrib'] * months
        starting_capital = config['start_cash'] + config['start_ask'] + config['start_free']
        total_invested = starting_capital + total_contributions
        
        median_gross = np.median(sim_result.final_ask) + np.median(sim_result.final_free) + np.median(sim_result.final_cash)
        median_net = np.median(sim_result.final_nominal)
        investment_growth = median_gross - total_invested
        total_tax = median_gross - median_net
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Investeret",
                value=f"{total_invested/1e3:,.0f}k",
                delta=f"Start: {starting_capital/1e3:,.0f}k + Bidrag: {total_contributions/1e3:,.0f}k"
            )
        
        with col2:
            growth_pct = (investment_growth / total_invested) * 100 if total_invested > 0 else 0
            st.metric(
                label="Investeringsafkast",
                value=f"{investment_growth/1e3:,.0f}k",
                delta=f"+{growth_pct:.0f}% af investeret"
            )
        
        with col3:
            multiplier = median_net / total_invested if total_invested > 0 else 1
            st.metric(
                label="Afkast Multiplikator",
                value=f"{multiplier:.2f}x",
                delta=f"Slutværdi / Investeret"
            )
        
        with col4:
            effective_return = ((median_net / total_invested) ** (1/config['years']) - 1) * 100 if total_invested > 0 else 0
            st.metric(
                label="Effektivt Årligt Afkast",
                value=f"{effective_return:.1f}%",
                delta="Efter skat"
            )
        
        # Charts
        col1, col2 = st.columns([1, 1])
        
        with col1:
            fig_waterfall = create_waterfall_chart(
                sim_result, config['monthly_contrib'], config['years'],
                config['start_cash'], config['start_ask'], config['start_free']
            )
            st.plotly_chart(fig_waterfall, use_container_width=True)
        
        with col2:
            fig_breakdown = create_contribution_growth_breakdown(
                sim_result, config['monthly_contrib'], config['years'],
                config['start_cash'], config['start_ask'], config['start_free']
            )
            st.plotly_chart(fig_breakdown, use_container_width=True)
        
        # Growth over time
        st.markdown("### Renters Rente Effekten Over Tid")
        
        fig_growth_time = create_growth_ratio_over_time(
            sim_result, config['monthly_contrib'], config['years'], config['start_year'],
            config['start_cash'], config['start_ask'], config['start_free']
        )
        st.plotly_chart(fig_growth_time, use_container_width=True)
        
        st.info("Jo længere tidshorisont, jo større andel fra investeringsafkast (renters rente).")
    
    # -------------------------------------------------------------
    # TAB 4: Sandsynlighed
    # -------------------------------------------------------------
    with tab4:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            fig_prob = create_probability_chart(sim_params, scenarios, config['target_goal'])
            st.plotly_chart(fig_prob, use_container_width=True)
        
        with col2:
            fig_scenario = create_scenario_comparison_chart(scenarios, sim_params, config['target_goal'])
            st.plotly_chart(fig_scenario, use_container_width=True)
    
    # -------------------------------------------------------------
    # TAB 5: Historical Backtesting
    # -------------------------------------------------------------
    with tab5:
        st.markdown("### Historisk Backtest")
        
        # Available years for backtesting
        available_years = sorted(HISTORICAL_RETURNS.keys())
        min_year = min(available_years)
        max_start_year = max(available_years) - config['years'] + 1
        
        if max_start_year >= min_year:
            st.markdown(f"**Tilgængelig data:** {min_year}-{max(available_years)} (S&P 500 Total Return)")
            
            backtest_start = st.slider(
                "Vælg startår for backtest",
                min_value=min_year,
                max_value=max_start_year,
                value=min(2000, max_start_year),
                key="backtest_start"
            )
            
            if st.button("Kør Historisk Backtest", type="primary"):
                with st.spinner("Kører historisk backtest..."):
                    backtest_result = run_historical_backtest(
                        start_year_backtest=backtest_start,
                        years=config['years'],
                        monthly_contrib=config['monthly_contrib'],
                        inflation_rate=config['inflation_rate'],
                        cash_interest=config['cash_interest'],
                        ask_rate=config['ask_rate'],
                        free_low=config['free_low'],
                        free_high=config['free_high'],
                        limit_initial=config['limit_initial'],
                        limit_increase=config['limit_increase'],
                        ask_initial_ceiling=config['ask_initial_ceiling'],
                        ask_annual_increase=config['ask_annual_increase'],
                        allocation_cash=config['allocation_cash'],
                        allocation_ask=config['allocation_ask'],
                        allocation_free=config['allocation_free'],
                        start_cash=config['start_cash'],
                        start_ask=config['start_ask'],
                        start_free=config['start_free']
                    )
                
                if backtest_result:
                    # Key metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Periode",
                            f"{backtest_start}-{backtest_start + config['years']}"
                        )
                    
                    with col2:
                        st.metric(
                            "Gns. Årligt Afkast",
                            f"{backtest_result['avg_return']:.1f}%"
                        )
                    
                    with col3:
                        reached_goal = backtest_result['final_real'] >= config['target_goal']
                        st.metric(
                            "Nåede Målet",
                            "Ja" if reached_goal else "Nej"
                        )
                    
                    with col4:
                        st.metric(
                            "Slutformue (Real)",
                            f"{backtest_result['final_real']/1e6:.2f} M"
                        )
                    
                    # Comparison chart
                    fig_backtest = create_backtest_chart(
                        backtest_result, history, config['years'], config['target_goal']
                    )
                    st.plotly_chart(fig_backtest, use_container_width=True)
                    
                    # Historical returns for the period
                    st.markdown("### Historiske Afkast i Perioden")
                    
                    period_years = list(range(backtest_start, backtest_start + config['years']))
                    period_returns = backtest_result['yearly_returns']
                    
                    returns_df = pd.DataFrame({
                        'År': period_years,
                        'Afkast (%)': [f"{r*100:.1f}%" for r in period_returns],
                        'Type': ['Positivt' if r >= 0 else 'Negativt' for r in period_returns]
                    })
                    
                    st.dataframe(returns_df, use_container_width=True, hide_index=True)
                    
                    # Comparison with MC
                    mc_median = np.median(final_real)
                    mc_p10 = np.percentile(final_real, 10)
                    mc_p90 = np.percentile(final_real, 90)
                    hist_val = backtest_result['final_real']
                    
                    if hist_val >= mc_p90:
                        st.success(f"""
                        Fremragende historisk periode!
                        Din strategi i {backtest_start}-{backtest_start + config['years']} ville have givet 
                        {hist_val/1e6:.2f} mio. kr. (top 10% af Monte Carlo, P90: {mc_p90/1e6:.2f} mio.)
                        """)
                    elif hist_val >= mc_median:
                        st.info(f"""
                        God historisk periode.
                        Resultat: {hist_val/1e6:.2f} mio. kr. (over median: {mc_median/1e6:.2f} mio.)
                        """)
                    elif hist_val >= mc_p10:
                        st.warning(f"""
                        Udfordrende historisk periode.
                        Resultat: {hist_val/1e6:.2f} mio. kr. (under median men over P10: {mc_p10/1e6:.2f} mio.)
                        """)
                    else:
                        st.error(f"""
                        Svær historisk periode.
                        Resultat: {hist_val/1e6:.2f} mio. kr. (i bund 10%, P10: {mc_p10/1e6:.2f} mio.)
                        """)
                else:
                    st.error("Ikke nok historiske data til den valgte periode.")
        else:
            st.warning(f"Din tidshorisont ({config['years']} år) er for lang til de tilgængelige historiske data.")
    
    # Footer
    st.markdown("""
    ---
    <div style="text-align: center; color: #9CA3AF; padding: 20px;">
        <p>InvestMål Analyzer v3.1 | Monte Carlo Simulation med {n} scenarier</p>
        <p style="font-size: 0.8rem;">
            Disclaimer: Dette er ikke finansiel rådgivning. 
            Historiske afkast garanterer ikke fremtidige resultater.
        </p>
    </div>
    """.format(n=config['n_paths']), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
