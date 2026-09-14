"""
LICA-SE UCF Engine — Motor de Funciones de Costo de Incertidumbre
Basado en las formulaciones del Prof. Sergio Rivera, UNAL
Reimplementación educativa para demostración interactiva.

Referencia principal:
Arevalo, C.; Santos, F.; Rivera, S. "Uncertainty Cost Functions for Solar PV,
Wind Energy, and PEVs." Int. J. Power and Energy Conversion, 2019.
"""
from dataclasses import dataclass
from typing import Optional, Dict
import numpy as np
from scipy.special import erf
from scipy.optimize import minimize_scalar

@dataclass
class UCFResult:
    expected_cost: float
    expected_upward: float
    expected_downward: float
    prob_upward: float
    prob_downward: float
    samples: np.ndarray
    costs: np.ndarray
    analytical_cost: Optional[float] = None

def ucf_from_samples(scheduled, samples, up_cost, down_cost):
    """Calcula UCF usando muestras de Monte Carlo."""
    samples = np.asarray(samples, dtype=float)
    pos_dev = np.maximum(samples - scheduled, 0.0)
    neg_dev = np.maximum(scheduled - samples, 0.0)
    costs = up_cost * pos_dev + down_cost * neg_dev
    return UCFResult(
        expected_cost=float(np.mean(costs)),
        expected_upward=float(np.mean(pos_dev)),
        expected_downward=float(np.mean(neg_dev)),
        prob_upward=float(np.mean(pos_dev > 0)),
        prob_downward=float(np.mean(neg_dev > 0)),
        samples=samples,
        costs=costs,
    )

def ucf_analytical_normal(mu, sigma, Ps, Cu, Co):
    """UCF analítica cerrada para distribución normal."""
    d = (mu - Ps) / (sigma * np.sqrt(2))
    Cua = (Cu/2)*(mu-Ps)*(1+erf(d)) + (Cu*sigma/np.sqrt(2*np.pi))*np.exp(-d**2)
    Coa = -(Co*sigma/np.sqrt(2*np.pi))*(np.exp(-mu**2/(2*sigma**2)) - np.exp(-d**2)) + \
          (Co/2)*(Ps-mu)*(erf(mu/(sigma*np.sqrt(2))) - erf(d))
    return Cua + Coa

def ucf_analytical_uniform(Pmin, Pmax, Ps, Cu, Co):
    """UCF analítica cerrada para distribución uniforme."""
    if Ps <= Pmin:
        E_Cu = Cu * ((Pmax+Pmin)/2 - Ps)
        E_Co = 0
    elif Ps >= Pmax:
        E_Cu = 0
        E_Co = Co * (Ps - (Pmax+Pmin)/2)
    else:
        E_Cu = Cu/(Pmax-Pmin) * (Ps**2/2 - Ps*Pmax + Pmax**2/2)
        E_Co = Co/(Pmax-Pmin) * (Ps**2/2 - Ps*Pmin + Pmin**2/2)
    return E_Cu + E_Co, E_Cu, E_Co

def optimal_schedule_uniform(Pmin, Pmax, Cu, Co):
    """Programación óptima para UCF uniforme."""
    # d/dPs E[UCF] = 0 => Ps* = (Cu*Pmax + Co*Pmin)/(Cu+Co)
    return (Cu * Pmax + Co * Pmin) / (Cu + Co)

def solar_beta_samples(capacity, alpha, beta, n=100000, seed=42):
    """Generación solar con distribución Beta."""
    rng = np.random.default_rng(seed)
    return capacity * rng.beta(alpha, beta, n)

def wind_normal_samples(mean, std, capacity, n=100000, seed=42):
    """Generación eólica con distribución normal truncada."""
    rng = np.random.default_rng(seed)
    samples = rng.normal(mean, std, n)
    return np.clip(samples, 0, capacity)

def ev_gaussian_samples(mu_energy, sigma_energy, n_vehicles=1000, seed=42):
    """Demanda de vehículos eléctricos con distribución gaussiana."""
    rng = np.random.default_rng(seed)
    return rng.normal(mu_energy, sigma_energy, n_vehicles)

def hydrogen_ucf_simulation(W_PVr=65, W_wr=150, Ps=150,
                             eta_H2=20, P_H2=5, P_kWh=20,
                             lam=6, beta_ln=0.25, Gr=1000, Rc=150,
                             vi=5, vr=15, vo=45, sigma_ray=15.95,
                             n_sim=100000, seed=42):
    """Simulación UCF para sistema de hidrógeno verde híbrido solar+eólico."""
    rng = np.random.default_rng(seed)
    
    # Solar: Lognormal -> potencia
    G = rng.lognormal(lam, beta_ln, n_sim)
    W_PV = np.where(G <= Rc,
                     W_PVr * G**2 / (Gr * Rc),
                     W_PVr * G / Gr)
    W_PV = np.clip(W_PV, 0, W_PVr)
    
    # Eólica: Rayleigh -> potencia
    v = rng.rayleigh(sigma_ray, n_sim)
    m = W_wr / (vr - vi)
    b = -W_wr * vi / (vr - vi)
    W_w = np.where(v < vi, 0,
          np.where(v <= vr, m * v + b,
          np.where(v <= vo, W_wr, 0)))
    
    # Potencia total
    P_total = W_PV + W_w
    mu_T = np.mean(P_total)
    sigma_T = np.std(P_total)
    
    # Costos de incertidumbre para H2
    c_under = eta_H2 * P_H2  # H2 no producido
    c_over = P_kWh            # Energía comprada de red
    
    # UCF
    pos_dev = np.maximum(P_total - Ps, 0)
    neg_dev = np.maximum(Ps - P_total, 0)
    costs = c_under * pos_dev + c_over * neg_dev
    
    # Analítico (aproximación normal)
    analytical = ucf_analytical_normal(mu_T, sigma_T, Ps, c_under, c_over)
    
    # Curva UCF vs Ps
    Ps_range = np.linspace(50, 300, 100)
    ucf_mc_curve = np.zeros(100)
    ucf_an_curve = np.zeros(100)
    for i, ps in enumerate(Ps_range):
        pos = np.maximum(P_total - ps, 0)
        neg = np.maximum(ps - P_total, 0)
        ucf_mc_curve[i] = np.mean(c_under * pos + c_over * neg)
        ucf_an_curve[i] = ucf_analytical_normal(mu_T, sigma_T, ps, c_under, c_over)
    
    return {
        "W_PV": W_PV, "W_w": W_w, "P_total": P_total,
        "mu_T": mu_T, "sigma_T": sigma_T,
        "costs": costs, "expected_cost": np.mean(costs),
        "analytical_cost": analytical,
        "error_pct": abs(np.mean(costs) - analytical) / analytical * 100,
        "Ps_range": Ps_range,
        "ucf_mc_curve": ucf_mc_curve,
        "ucf_an_curve": ucf_an_curve,
        "c_under": c_under, "c_over": c_over,
    }

def cslcoe_simulation(capex_pv=60e6, capex_bess=20e6, opex_annual=1.2e6,
                       T=25, r=0.06, P_scheduled=50,
                       P_min=30, P_max=70, Cu=30, Co=70,
                       P_eff=15, n_sim=5000, seed=42):
    """
    Simulación del Controllable System LCOE (CS-LCOE).
    
    Modelo BESS consistente (analítico = MC):
    - BESS absorbe hasta P_eff MW de desviación en ambas direcciones
    - Surplus después BESS: max(P_av - Ps - P_eff, 0) → costo Cu
    - Déficit después BESS: max(Ps - P_eff - P_av, 0) → costo Co
    - Distribución uniforme U(P_min, P_max) para P_av
    """
    rng = np.random.default_rng(seed)
    
    # === ANALYTICAL (closed-form, uniform + BESS buffer) ===
    a, b = P_min, P_max
    threshold_s = P_scheduled + P_eff   # surplus beyond this → Cu cost
    threshold_d = P_scheduled - P_eff   # deficit below this → Co cost
    
    # E[max(P_av - threshold_s, 0)] for P_av ~ U(a, b)
    if threshold_s >= b:
        E_surplus_after = 0.0
    elif threshold_s <= a:
        E_surplus_after = (a + b) / 2.0 - threshold_s
    else:
        E_surplus_after = (b - threshold_s) ** 2 / (2.0 * (b - a))
    
    # E[max(threshold_d - P_av, 0)] for P_av ~ U(a, b)
    if threshold_d <= a:
        E_deficit_after = 0.0
    elif threshold_d >= b:
        E_deficit_after = threshold_d - (a + b) / 2.0
    else:
        E_deficit_after = (threshold_d - a) ** 2 / (2.0 * (b - a))
    
    ucf_hourly = Cu * E_surplus_after + Co * E_deficit_after  # $/h
    
    # Discount factors
    discount = np.array([1.0 / (1 + r) ** t for t in range(T)])
    pv_factor = float(np.sum(discount))
    
    # Total discounted costs (CAPEX at t=0 + annualized OPEX + UCF)
    annual_opex_ucf = opex_annual + ucf_hourly * 8760.0
    total_costs_an = (capex_pv + capex_bess) + annual_opex_ucf * pv_factor
    total_energy = P_scheduled * 8760.0 * pv_factor
    
    cslcoe_analytical = total_costs_an / total_energy  # $/MWh
    
    # LCOE tradicional (sin costo de incertidumbre)
    lcoe_tradicional = ((capex_pv + capex_bess) + opex_annual * pv_factor) / total_energy
    
    # UCF sin BESS (para comparación)
    e_ucf_no_bess, _, _ = ucf_analytical_uniform(P_min, P_max, P_scheduled, Cu, Co)
    
    # === MONTE CARLO (same BESS model as analytical) ===
    mc_cslcoes = np.zeros(n_sim)
    for i in range(n_sim):
        total_cost = capex_pv + capex_bess
        total_en = 0.0
        for t in range(T):
            P_av = rng.uniform(P_min, P_max, 8760)
            
            surplus = np.maximum(P_av - P_scheduled, 0.0)
            deficit = np.maximum(P_scheduled - P_av, 0.0)
            
            # BESS absorbs up to P_eff in BOTH directions
            surplus_after = np.maximum(surplus - P_eff, 0.0)
            deficit_after = np.maximum(deficit - P_eff, 0.0)
            
            ucf_cost_h = Cu * np.mean(surplus_after) + Co * np.mean(deficit_after)
            
            # Energy delivered: Ps minus unserved deficit
            delivered_h = P_scheduled - np.mean(deficit_after)
            
            d = discount[t]
            total_cost += (opex_annual + ucf_cost_h * 8760.0) * d
            total_en += delivered_h * 8760.0 * d
        
        mc_cslcoes[i] = total_cost / total_en if total_en > 0 else 0.0
    
    ucf_premium = cslcoe_analytical - lcoe_tradicional
    
    return {
        "cslcoe_analytical": cslcoe_analytical,
        "cslcoe_mc_mean": float(np.mean(mc_cslcoes)),
        "cslcoe_mc_std": float(np.std(mc_cslcoes)),
        "cslcoe_mc_p95": float(np.percentile(mc_cslcoes, 95)),
        "ucf_premium": ucf_premium,
        "mc_cslcoes": mc_cslcoes,
        "lcoe_tradicional": lcoe_tradicional,
        "ucf_hourly": ucf_hourly,
        "ucf_no_bess": e_ucf_no_bess,
        "bess_reduction_pct": (1.0 - ucf_hourly / max(e_ucf_no_bess, 1e-9)) * 100,
        "T": T, "r": r,
    }

def monte_carlo_animation_frames(mu, sigma, Ps, Cu, Co, n_frames=200, seed=42):
    """Genera datos para animación incremental de Monte Carlo."""
    rng = np.random.default_rng(seed)
    N = 1000000
    Pe = rng.normal(mu, sigma, N)
    
    CO = Co * np.maximum(Ps - Pe, 0)
    CU = Cu * np.maximum(Pe - Ps, 0)
    CUCO = CU + CO
    
    analytical = ucf_analytical_normal(mu, sigma, Ps, Cu, Co)
    
    frames = []
    for i in range(1, n_frames + 1):
        n = min(i * 5000, N)
        frames.append({
            "n": n,
            "Pe": Pe[:n],
            "CU": CU[:n],
            "CO": CO[:n],
            "CUCO": CUCO[:n],
            "mean_cost": np.mean(CUCO[:n]),
            "analytical": analytical,
        })
    return frames, analytical

