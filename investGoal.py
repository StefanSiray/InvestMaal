import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional

# Import shared models and centralized config
from models import MarketScenario, TaxRules, ASKRules, Portfolio
from config import SimulationConfig, DEFAULT_SCENARIOS, DEFAULT_SCENARIO


# -------------------------------------------------------------
# 3. SIMULATIONS MOTOR (Forbedret med Portfolio klasse)
# -------------------------------------------------------------
def run_simulation(
    config: SimulationConfig,
    monthly_contrib: float,
    scenario: MarketScenario,
    return_full_history: bool = False
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Kør Monte Carlo simulation
    
    Returns:
        (final_nominal, final_real, history_nominal)
    """
    months = int(config.years * 12)
    n_paths = config.n_paths if return_full_history else config.n_paths_search
    
    mu_stock = scenario.annual_return / 12.0
    sigma_stock = scenario.volatility / np.sqrt(12.0)
    r_cash_mo = config.cash_interest / 12.0
    
    final_nom = np.zeros(n_paths)
    final_real = np.zeros(n_paths)
    history_nom = np.zeros((n_paths, months + 1)) if return_full_history else None
    
    # Brug de nye attributnavne
    start_cash_sum = config.start_cash
    start_stock_ask = config.start_ask
    start_stock_free = config.start_free
    
    discount_factor = (1 + config.inflation_rate) ** config.years
    
    for i in range(n_paths):
        portfolio = Portfolio(start_stock_ask, start_stock_free, start_cash_sum)
        
        if return_full_history:
            history_nom[i, 0] = portfolio.total_value()
        
        for m in range(1, months + 1):
            year_idx = (m - 1) // 12
            month_in_year = (m - 1) % 12
            
            # Årsskifte: Skat på ASK
            if month_in_year == 0 and m > 1:
                portfolio.apply_ask_tax(config.tax_rules.ask_rate)
            
            # Indbetaling
            ask_ceiling = config.ask_rules.get_ceiling(year_idx)
            portfolio.contribute(
                monthly_contrib,
                config.allocation,
                ask_ceiling,
                config.ask_rules
            )
            
            # Markedsbevægelse
            shock = np.random.normal(mu_stock, sigma_stock)
            portfolio.apply_market_shock(shock, r_cash_mo)
            
            if return_full_history:
                history_nom[i, m] = portfolio.total_value()
        
        # Slutberegning
        net_value = portfolio.get_net_value(config.tax_rules, config.years)
        final_nom[i] = net_value
        final_real[i] = net_value / discount_factor
    
    return final_nom, final_real, history_nom


# -------------------------------------------------------------
# 4. OPTIMERING & ANALYSE
# -------------------------------------------------------------
def find_monthly_requirement(
    config: SimulationConfig,
    scenario: MarketScenario,
    target: float,
    probability: float = 0.90,
    use_real_value: bool = False
) -> float:
    """Find nødvendigt månedligt bidrag"""
    low, high = 0, 50000
    best_guess = high
    
    for _ in range(15):
        mid = (low + high) / 2
        f_nom, f_real, _ = run_simulation(config, mid, scenario)
        
        values = f_real if use_real_value else f_nom
        prob = np.mean(values >= target)
        
        if prob >= probability:
            best_guess = mid
            high = mid
        else:
            low = mid
    
    return best_guess


# -------------------------------------------------------------
# 5. SCENARIE ANALYSE (Forbedring #5)
# -------------------------------------------------------------
def analyze_scenarios(config: SimulationConfig, monthly_contrib: float) -> Dict:
    """Kør simulationer for alle scenarier"""
    results = {}
    
    for scenario_name, scenario in config.scenarios.items():
        f_nom, f_real, _ = run_simulation(config, monthly_contrib, scenario)
        
        results[scenario_name] = {
            'scenario': scenario,
            'median_nom': np.median(f_nom),
            'median_real': np.median(f_real),
            'p10': np.percentile(f_real, 10),
            'p90': np.percentile(f_real, 90),
            'success_prob': np.mean(f_real >= config.target_goal) * 100
        }
    
    return results


# -------------------------------------------------------------
# 6. SENSITIVITY ANALYSE (Forbedring #10)
# -------------------------------------------------------------
def sensitivity_analysis(config: SimulationConfig, monthly_contrib: float) -> Dict:
    """Analysér følsomhed overfor parametre"""
    scenario = config.scenarios[config.default_scenario]
    base_nom, base_real, _ = run_simulation(config, monthly_contrib, scenario)
    base_success = np.mean(base_real >= config.target_goal) * 100
    
    results = {
        'base': {'success': base_success, 'median': np.median(base_real)}
    }
    
    # Test inflation
    inflation_range = np.linspace(0.01, 0.04, 5)
    results['inflation'] = {'values': [], 'success': [], 'median': []}
    
    for infl in inflation_range:
        config_copy = SimulationConfig()
        config_copy.inflation_rate = infl
        f_nom, f_real, _ = run_simulation(config_copy, monthly_contrib, scenario)
        results['inflation']['values'].append(infl * 100)
        results['inflation']['success'].append(np.mean(f_real >= config.target_goal) * 100)
        results['inflation']['median'].append(np.median(f_real))
    
    # Test afkast
    return_range = np.linspace(0.04, 0.10, 5)
    results['return'] = {'values': [], 'success': [], 'median': []}
    
    for ret in return_range:
        scenario_copy = MarketScenario("Test", ret, scenario.volatility, "Test")
        f_nom, f_real, _ = run_simulation(config, monthly_contrib, scenario_copy)
        results['return']['values'].append(ret * 100)
        results['return']['success'].append(np.mean(f_real >= config.target_goal) * 100)
        results['return']['median'].append(np.median(f_real))
    
    # Test volatilitet
    vol_range = np.linspace(0.10, 0.30, 5)
    results['volatility'] = {'values': [], 'success': [], 'median': []}
    
    for vol in vol_range:
        scenario_copy = MarketScenario("Test", scenario.annual_return, vol, "Test")
        f_nom, f_real, _ = run_simulation(config, monthly_contrib, scenario_copy)
        results['volatility']['values'].append(vol * 100)
        results['volatility']['success'].append(np.mean(f_real >= config.target_goal) * 100)
        results['volatility']['median'].append(np.median(f_real))
    
    return results


# -------------------------------------------------------------
# 7. PERFORMANCE METRICS (Forbedring #4)
# -------------------------------------------------------------
def calculate_metrics(history: np.ndarray, monthly_contrib: float) -> Dict:
    """Beregn performance metrics"""
    # Remove paths with zeros or invalid values
    valid_paths = history[~np.any(history <= 0, axis=1)]
    
    if len(valid_paths) == 0:
        return {
            'sharpe': 0.0,
            'avg_max_drawdown': 0.0,
            'total_invested': 0.0,
            'median_return': 0.0
        }
    
    # Returns
    returns = np.diff(valid_paths, axis=1) / valid_paths[:, :-1]
    returns = returns[np.isfinite(returns).all(axis=1)]  # Remove inf/nan
    
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
    median_return = ((median_final / valid_paths[0, 0]) ** (1/10) - 1) * 100
    
    return {
        'sharpe': sharpe,
        'avg_max_drawdown': avg_max_drawdown,
        'total_invested': total_invested,
        'median_return': median_return
    }


# -------------------------------------------------------------
# 8. VISUALISERING (Forbedring #8: Risk/Return scatter)
# -------------------------------------------------------------
def create_visualizations(config: SimulationConfig, monthly_contrib: float, 
                         final_nom: np.ndarray, final_real: np.ndarray,
                         history: np.ndarray, scenario_results: Dict,
                         sensitivity_results: Dict, req_real_90: float):
    """Skab alle visualiseringer"""
    
    time_axis = np.linspace(config.start_year, config.start_year + config.years, 
                           history.shape[1])
    
    p10 = np.percentile(history, 10, axis=0)
    p50 = np.percentile(history, 50, axis=0)
    p90 = np.percentile(history, 90, axis=0)
    
    fig = plt.figure(figsize=(16, 12))
    plt.subplots_adjust(hspace=0.35, wspace=0.3)
    
    # --- GRAF 1: Trajektorier ---
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(time_axis, p50, color='blue', linewidth=2, label='Median (50%)')
    ax1.fill_between(time_axis, p10, p90, color='blue', alpha=0.2, 
                     label='80% sandsynligt udfald (P10-P90)')
    ax1.axhline(config.target_goal, color='red', linestyle='--', 
                label=f'Mål: {config.target_goal/1e6:.1f} mio.')
    ax1.set_title(f"Formueudvikling (ved {monthly_contrib:,.0f} kr./md)")
    ax1.set_ylabel("Formue (DKK)")
    ax1.set_xlabel("År")
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
    
    # --- GRAF 2: Histogram ---
    ax2 = plt.subplot(3, 2, 2)
    ax2.hist(final_nom, bins=40, alpha=0.6, color='blue', label='Nominel')
    ax2.hist(final_real, bins=40, alpha=0.6, color='green', label='Reel (Købekraft)')
    ax2.axvline(config.target_goal, color='red', linestyle='--', linewidth=2, label='Mål')
    ax2.set_title("Slutformue om 10 år")
    ax2.set_xlabel("Værdi (DKK)")
    ax2.set_ylabel("Antal scenarier")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
    
    # --- GRAF 3: S-kurve ---
    ax3 = plt.subplot(3, 2, 3)
    scan_amounts = np.linspace(0, 30000, 20)
    probs_nom = []
    probs_real = []
    
    scenario = config.scenarios[config.default_scenario]
    for amt in scan_amounts:
        fn, fr, _ = run_simulation(config, amt, scenario)
        probs_nom.append(np.mean(fn >= config.target_goal) * 100)
        probs_real.append(np.mean(fr >= config.target_goal) * 100)
    
    ax3.plot(scan_amounts, probs_nom, 'o-', color='blue', label='Nominel')
    ax3.plot(scan_amounts, probs_real, 'o--', color='green', label='Reel')
    ax3.axhline(90, color='gray', linestyle=':', alpha=0.8, label='90% mål')
    ax3.axvline(monthly_contrib, color='orange', linestyle='-', linewidth=2, 
                label='Dit bidrag')
    ax3.axvline(req_real_90, color='green', linestyle=':', alpha=0.7, 
                label='Nødvendig (90%)')
    ax3.set_title("Sandsynlighed vs. Månedligt bidrag")
    ax3.set_xlabel("Månedligt bidrag (DKK)")
    ax3.set_ylabel("Sandsynlighed (%)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # --- GRAF 4: Risk/Return Scatter (Scenarie) (Forbedring #8) ---
    ax4 = plt.subplot(3, 2, 4)
    
    scenario_names = []
    medians = []
    risks = []
    colors_map = {'bull': 'green', 'normal': 'blue', 'bear': 'orange', 'crash': 'red'}
    
    for name, result in scenario_results.items():
        scenario_names.append(name)
        medians.append(result['median_real'])
        risk = result['p90'] - result['p10']
        risks.append(risk)
    
    for i, name in enumerate(scenario_names):
        ax4.scatter(risks[i]/1e6, medians[i]/1e6, s=200, 
                   color=colors_map.get(name, 'gray'), alpha=0.6, label=name)
        ax4.annotate(name, (risks[i]/1e6, medians[i]/1e6), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax4.axhline(config.target_goal/1e6, color='red', linestyle='--', 
                alpha=0.5, label='Mål')
    ax4.set_title("Risk/Return Trade-off (Scenarier)")
    ax4.set_xlabel("Risiko: P90-P10 spænd (mio. DKK)")
    ax4.set_ylabel("Forventet værdi: Median (mio. DKK)")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # --- GRAF 5: Sensitivity - Inflation (Forbedring #10) ---
    ax5 = plt.subplot(3, 2, 5)
    ax5.plot(sensitivity_results['inflation']['values'], 
             sensitivity_results['inflation']['success'], 
             'o-', color='purple', linewidth=2, label='Succesrate')
    ax5.axhline(90, color='gray', linestyle=':', alpha=0.8)
    ax5.axvline(config.inflation_rate * 100, color='orange', linestyle='--', 
                label='Nuværende')
    ax5.set_title("Følsomhed: Inflation")
    ax5.set_xlabel("Inflationsrate (%)")
    ax5.set_ylabel("Sandsynlighed for succes (%)")
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # --- GRAF 6: Sensitivity - Afkast & Volatilitet (Forbedring #10) ---
    ax6 = plt.subplot(3, 2, 6)
    ax6.plot(sensitivity_results['return']['values'], 
             sensitivity_results['return']['success'], 
             'o-', color='green', linewidth=2, label='Afkast-følsomhed')
    ax6.plot(sensitivity_results['volatility']['values'], 
             sensitivity_results['volatility']['success'], 
             's--', color='red', linewidth=2, label='Volatilitets-følsomhed')
    ax6.axhline(90, color='gray', linestyle=':', alpha=0.8)
    ax6.set_title("Følsomhed: Afkast & Risiko")
    ax6.set_xlabel("Parameter værdi (%)")
    ax6.set_ylabel("Sandsynlighed for succes (%)")
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.show()


# -------------------------------------------------------------
# 9. RAPPORT (Forbedring #4: Detaljeret)
# -------------------------------------------------------------
def print_report(config: SimulationConfig, monthly_contrib: float,
                final_nom: np.ndarray, final_real: np.ndarray,
                scenario_results: Dict, metrics: Dict, req_real_90: float):
    """Print detaljeret rapport"""
    
    median_final_nom = np.median(final_nom)
    median_final_real = np.median(final_real)
    success_nom = np.mean(final_nom >= config.target_goal) * 100
    success_real = np.mean(final_real >= config.target_goal) * 100
    
    print("=" * 70)
    print(f"INVESTERINGS ANALYSE RAPPORT")
    print("=" * 70)
    print(f"Analysedato: 2025-12-08")
    print(f"Simulerede scenarier: {config.n_paths}")
    
    print("\n" + "─" * 70)
    print("GRUNDLÆGGENDE INFORMATION")
    print("─" * 70)
    print(f"Mål:                    {config.target_goal:>12,.0f} kr.")
    print(f"Tidshorisont:           {config.years:>12} år")
    print(f"Startår:                {config.start_year:>12}")
    print(f"Antaget inflation:      {config.inflation_rate*100:>12.1f}%")
    print(f"Dynamisk knækgrænse:    {config.tax_rules.get_limit(config.years):>12,.0f} kr.")
    
    print("\n" + "─" * 70)
    print("BUDGET & BIDRAG")
    print("─" * 70)
    print(f"Månedlig nettoindkomst: {config.income_net_monthly:>12,.0f} kr.")
    print(f"Månedlige udgifter:     {config.expenses_monthly:>12,.0f} kr.")
    print(f"Dit bidrag:             {monthly_contrib:>12,.0f} kr./md")
    print(f"Opsparingsrate:         {(monthly_contrib/config.income_net_monthly)*100:>12.1f}%")
    
    print("\n" + "─" * 70)
    print("ALLOKERING")
    print("─" * 70)
    for key, value in config.allocation.items():
        amount = monthly_contrib * value
        print(f"{key:30} {value*100:>6.1f}%  ({amount:>8,.0f} kr./md)")
    
    print("\n" + "─" * 70)
    print("RESULTATER (NORMAL SCENARIE)")
    print("─" * 70)
    print(f"Sandsynlighed (Nominel): {success_nom:>11.1f}%")
    print(f"Sandsynlighed (Reel):    {success_real:>11.1f}%")
    print(f"Median slutformue (Nom): {median_final_nom:>12,.0f} kr.")
    print(f"Median købekraft (Real): {median_final_real:>12,.0f} kr.")
    print(f"Nødvendigt for 90%:      {req_real_90:>12,.0f} kr./md")
    
    print("\n" + "─" * 70)
    print("PERFORMANCE METRICS")
    print("─" * 70)
    print(f"Sharpe Ratio:            {metrics['sharpe']:>12.2f}")
    print(f"Gns. Max Drawdown:       {metrics['avg_max_drawdown']:>12.1f}%")
    print(f"Total investeret:        {metrics['total_invested']:>12,.0f} kr.")
    print(f"Median årligt afkast:    {metrics['median_return']:>12.1f}%")
    
    print("\n" + "─" * 70)
    print("SCENARIE ANALYSE")
    print("─" * 70)
    print(f"{'Scenarie':<15} {'Median (Real)':>15} {'P10-P90 Spænd':>15} {'Succes%':>10}")
    print("─" * 70)
    
    for name, result in scenario_results.items():
        median = result['median_real']
        span = result['p90'] - result['p10']
        prob = result['success_prob']
        print(f"{name:<15} {median:>12,.0f} kr. {span:>12,.0f} kr. {prob:>9.1f}%")
    
    print("\n" + "─" * 70)
    print("ANBEFALINGER")
    print("─" * 70)
    
    if success_real < 50:
        print("⚠️  KRITISK: Lav sandsynlighed for succes.")
        print(f"   → Overvej at øge bidrag til mindst {req_real_90:,.0f} kr./md")
        print("   → Eller forlæng tidshorisonten")
    elif success_real < 75:
        print("⚠️  MODERAT: Middel sandsynlighed for succes.")
        print(f"   → For 90% sikkerhed, øg bidrag til {req_real_90:,.0f} kr./md")
    elif success_real < 90:
        print("✓  GOD: Høj sandsynlighed for succes.")
        print("   → Du er på rette vej, men der er stadig risiko")
    else:
        print("✓✓ FREMRAGENDE: Meget høj sandsynlighed for succes!")
        print("   → Du har god margin til dit mål")
    
    print("\n" + "=" * 70)


# -------------------------------------------------------------
# 10. HOVEDPROGRAM
# -------------------------------------------------------------
def main():
    """Hovedprogram"""
    
    # Initialiser konfiguration
    config = SimulationConfig()
    np.random.seed(config.seed)
    
    print("\n🚀 STARTER INVESTERINGS SIMULATOR V2.0\n")
    print(f"Simulerer {config.n_paths} scenarier...\n")
    
    # Vælg scenarie
    scenario = config.scenarios[config.default_scenario]
    
    # Find nødvendigt bidrag (bruges til reference)
    print("📊 Beregner nødvendigt bidrag for 90% succes...")
    req_real_90 = find_monthly_requirement(
        config, scenario, config.target_goal, 0.90, use_real_value=True
    )
    
    # Brug beregnet månedligt bidrag
    monthly_contrib = config.monthly_contrib
    print(f"✓ Bruger budget: {monthly_contrib:,.0f} kr./md\n")
    
    # Kør hovedsimulation
    print("🎲 Kører Monte Carlo simulation...")
    final_nom, final_real, history = run_simulation(
        config, monthly_contrib, scenario, return_full_history=True
    )
    print("✓ Simulation færdig\n")
    
    # Scenarie analyse
    print("🌍 Analyserer forskellige markedsscenarier...")
    scenario_results = analyze_scenarios(config, monthly_contrib)
    print("✓ Scenarie analyse færdig\n")
    
    # Sensitivity analyse
    print("🔬 Kører sensitivity analyse...")
    sensitivity_results = sensitivity_analysis(config, monthly_contrib)
    print("✓ Sensitivity analyse færdig\n")
    
    # Performance metrics
    print("📈 Beregner performance metrics...")
    metrics = calculate_metrics(history, monthly_contrib)
    print("✓ Metrics beregnet\n")
    
    # Print rapport
    print_report(config, monthly_contrib, final_nom, final_real, 
                 scenario_results, metrics, req_real_90)
    
    # Vis grafer
    print("\n📊 Genererer visualiseringer...")
    create_visualizations(config, monthly_contrib, final_nom, final_real,
                         history, scenario_results, sensitivity_results, req_real_90)
    print("✓ Færdig!\n")


# -------------------------------------------------------------
# 11. KØR PROGRAMMET
# -------------------------------------------------------------
if __name__ == "__main__":
    main()