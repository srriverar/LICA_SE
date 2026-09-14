"""
LICA-SE Simulations Module — Interactive simulations for power systems
Based on research by Prof. Sergio Rivera, UNAL
"""
import numpy as np

def simular_opf_simplificado(n_barras=14, carga_total_mw=250, gen_renovable_pct=30, seed=42):
    """Simulación simplificada de OPF (Optimal Power Flow)"""
    np.random.seed(seed)
    t = np.linspace(0, 24, 96)
    
    # Generación
    gen_termica = carga_total_mw * (1 - gen_renovable_pct/100)
    gen_solar_base = carga_total_mw * gen_renovable_pct/100 * np.maximum(0, np.sin(np.pi * (t - 6) / 12))
    gen_solar = gen_solar_base * (1 + np.random.randn(96) * 0.15)
    gen_solar = np.maximum(0, gen_solar)
    
    # Demanda
    demanda = carga_total_mw * (0.7 + 0.3 * np.sin(np.pi * (t - 3) / 12))
    demanda *= (1 + np.random.randn(96) * 0.03)
    
    # Excedente/déficit
    total_gen = gen_termica + gen_solar
    balance = total_gen - demanda
    
    # Costo de generación (térmica: cuadrática)
    costo_termico = 25 + 0.02 * gen_termica**2  # $/MWh
    costo_total = costo_termico * gen_termica * (24/96)
    
    # Voltajes por barra (simplificado)
    voltajes = 1.0 + np.random.randn(n_barras) * 0.03
    voltajes = np.clip(voltajes, 0.95, 1.05)
    
    # Pérdidas
    perdidas_pct = 2.5 + np.random.randn() * 0.5
    perdidas_mw = carga_total_mw * perdidas_pct / 100
    
    return {
        "horas": t,
        "gen_termica_mw": np.full(96, gen_termica),
        "gen_solar_mw": gen_solar,
        "demanda_mw": demanda,
        "balance_mw": balance,
        "costo_total_musd": np.sum(costo_total),
        "voltajes_pu": voltajes,
        "perdidas_mw": perdidas_mw,
        "perdidas_pct": perdidas_pct,
        "n_barras": n_barras,
    }

def simular_microgrid_despacho(solar_mw=5, eolica_mw=3, bess_kwh=2000, diesel_mw=4,
                                demanda_pico_mw=8, dr_pct=15, seed=42):
    """Simulación de despacho óptimo en microgrid"""
    np.random.seed(seed)
    t = np.linspace(0, 24, 96)
    
    # Generación solar
    solar = solar_mw * np.maximum(0, np.sin(np.pi * (t - 6) / 12)) * (1 + np.random.randn(96) * 0.12)
    solar = np.maximum(0, solar)
    
    # Generación eólica (más estocástica)
    viento = 0.3 + 0.5 * np.abs(np.sin(np.pi * t / 8 + 1.5))
    eolica = eolica_mw * viento * (1 + np.random.randn(96) * 0.2)
    eolica = np.clip(eolica, 0, eolica_mw)
    
    # Demanda
    demanda = demanda_pico_mw * (0.5 + 0.5 * np.sin(np.pi * (t - 3) / 12))
    demanda = np.maximum(demanda * 0.4, demanda)
    
    # Demand Response
    dr_mw = demanda * dr_pct / 100
    dr_activado = np.where(solar + eolica > demanda * 0.8, dr_mw, 0)
    demanda_opt = demanda - dr_activado
    
    # BESS operation
    soc = np.zeros(96)
    soc[0] = 50
    bess_carga = np.zeros(96)
    bess_descarga = np.zeros(96)
    bess_max_kw = bess_kwh / 4  # C-rate 0.25
    
    renov = solar + eolica
    for i in range(1, 96):
        exceso = renov[i] - demanda_opt[i]
        if exceso > 0 and soc[i-1] < 95:
            carga = min(exceso, bess_max_kw / 1000, (95 - soc[i-1]) * bess_kwh / 100 / 1000)
            bess_carga[i] = carga
            soc[i] = soc[i-1] + carga * 1000 * 0.25 / bess_kwh * 100
        elif renov[i] < demanda_opt[i] and soc[i-1] > 20:
            deficit = demanda_opt[i] - renov[i]
            descarga = min(deficit, bess_max_kw / 1000, (soc[i-1] - 20) * bess_kwh / 100 / 1000)
            bess_descarga[i] = descarga
            soc[i] = soc[i-1] - descarga * 1000 * 0.25 / bess_kwh * 100
        else:
            soc[i] = soc[i-1]
    
    # Diesel (backup)
    diesel = np.maximum(0, demanda_opt - renov - bess_descarga + bess_carga)
    diesel = np.minimum(diesel, diesel_mw)
    
    # Costos
    costo_diesel = diesel * 180  # USD/MWh
    costo_renov = (solar + eolica) * 35  # USD/MWh
    costo_bess = (bess_carga + bess_descarga) * 15  # USD/MWh degradación
    costo_total = np.sum((costo_diesel + costo_renov + costo_bess) * 0.25)  # 15-min intervals
    
    # Curtailment
    curtailment = np.maximum(0, renov - demanda_opt - bess_carga)
    
    return {
        "horas": t,
        "solar_mw": solar,
        "eolica_mw": eolica,
        "demanda_mw": demanda,
        "demanda_opt_mw": demanda_opt,
        "dr_activado_mw": dr_activado,
        "bess_carga_mw": bess_carga,
        "bess_descarga_mw": bess_descarga,
        "soc_pct": soc,
        "diesel_mw": diesel,
        "curtailment_mw": curtailment,
        "costo_total_usd": costo_total,
        "costo_diesel_usd": np.sum(costo_diesel * 0.25),
        "costo_renov_usd": np.sum(costo_renov * 0.25),
        "energia_solar_mwh": np.sum(solar * 0.25),
        "energia_eolica_mwh": np.sum(eolica * 0.25),
        "energia_diesel_mwh": np.sum(diesel * 0.25),
    }

def simular_gestion_activos(n_transformadores=20, horizonte_anios=10, seed=42):
    """Simulación de gestión de activos tipo PywerAPM"""
    np.random.seed(seed)
    
    # Flota de transformadores
    trafos = []
    for i in range(n_transformadores):
        edad = np.random.randint(1, 35)
        salud = max(0, 100 - edad * 2.5 + np.random.randn() * 10)
        criticidad = np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.2, 0.3, 0.3, 0.1])
        carga_pct = 50 + np.random.randn() * 20
        carga_pct = np.clip(carga_pct, 30, 100)
        
        # Probabilidad de falla (Weibull)
        beta = 2.5  # shape parameter
        eta = 30  # scale parameter (vida característica)
        prob_falla = 1 - np.exp(-((edad / eta) ** beta))
        
        riesgo = prob_falla * criticidad * carga_pct / 100
        
        trafos.append({
            "id": f"TRF-{i+1:03d}",
            "edad": edad,
            "salud": salud,
            "criticidad": criticidad,
            "carga_pct": carga_pct,
            "prob_falla": prob_falla,
            "riesgo": riesgo,
            "decision": "Reemplazar" if riesgo > 2.5 else "Mantener" if riesgo > 1.0 else "Monitorear"
        })
    
    # Proyección de horizonte
    anios = np.arange(horizonte_anios + 1)
    fallas_esperadas = np.zeros(horizonte_anios + 1)
    costo_total = np.zeros(horizonte_anios + 1)
    
    for yr in range(horizonte_anios + 1):
        for t in trafos:
            edad_futura = t["edad"] + yr
            prob = 1 - np.exp(-((edad_futura / 30) ** 2.5))
            fallas_esperadas[yr] += prob
            costo_total[yr] += prob * t["criticidad"] * 50000  # Costo ENS
        
    return {
        "trafos": trafos,
        "anios": anios,
        "fallas_esperadas": fallas_esperadas,
        "costo_riesgo_usd": costo_total,
        "n_reemplazar": sum(1 for t in trafos if t["decision"] == "Reemplazar"),
        "n_mantener": sum(1 for t in trafos if t["decision"] == "Mantener"),
        "n_monitorear": sum(1 for t in trafos if t["decision"] == "Monitorear"),
    }

def simular_ucf_solar(capacidad_mw=10, irradiancia_media=5.0, sigma_irr=0.8, 
                       precio_mwh=50, seed=42):
    """Simulación de Funciones de Costo de Incertidumbre (UCF) para solar"""
    np.random.seed(seed)
    
    # Distribución de irradiancia
    n_samples = 10000
    irradiancia = np.random.normal(irradiancia_media, sigma_irr, n_samples)
    irradiancia = np.clip(irradiancia, 0, 7)
    
    # Generación = f(irradiancia)
    gen = capacidad_mw * irradiancia / 6.0  # Normalizado a 6 kWh/m²/d
    gen = np.clip(gen, 0, capacidad_mw)
    
    # Programación (dispatch programado)
    dispatch_prog = np.median(gen)
    
    # Costo de incertidumbre
    sobre_gen = np.maximum(gen - dispatch_prog, 0)  # Exceso → curtailment
    sub_gen = np.maximum(dispatch_prog - gen, 0)     # Déficit → reserva
    
    costo_sobre = sobre_gen * precio_mwh * 0.3  # Costo de curtailment
    costo_sub = sub_gen * precio_mwh * 1.5  # Costo de reserva (más caro)
    
    ucf_total = costo_sobre + costo_sub
    ucf_esperado = np.mean(ucf_total)
    
    # Curva UCF vs dispatch
    dispatches = np.linspace(0, capacidad_mw, 50)
    ucf_vs_dispatch = np.zeros(50)
    for i, d in enumerate(dispatches):
        sobre = np.maximum(gen - d, 0)
        sub = np.maximum(d - gen, 0)
        ucf_vs_dispatch[i] = np.mean(sobre * precio_mwh * 0.3 + sub * precio_mwh * 1.5)
    
    return {
        "irradiancia_samples": irradiancia[:500],
        "gen_samples": gen[:500],
        "dispatch_programado_mw": dispatch_prog,
        "ucf_esperado_usd_h": ucf_esperado,
        "dispatches": dispatches,
        "ucf_vs_dispatch": ucf_vs_dispatch,
        "dispatch_optimo": dispatches[np.argmin(ucf_vs_dispatch)],
        "costo_sobre_medio": np.mean(costo_sobre),
        "costo_sub_medio": np.mean(costo_sub),
    }

def simular_deepso(n_particulas=30, n_iteraciones=100, n_variables=5, seed=42):
    """Simulación del algoritmo VNS-DEEPSO para despacho"""
    np.random.seed(seed)
    
    # Función objetivo: despacho económico con 5 generadores
    costos_a = np.array([0.004, 0.006, 0.003, 0.005, 0.007])
    costos_b = np.array([12, 10, 15, 11, 9])
    costos_c = np.array([200, 180, 250, 190, 220])
    p_min = np.array([10, 20, 15, 10, 25])
    p_max = np.array([100, 150, 120, 130, 80])
    demanda = 350  # MW
    
    # Inicialización DEEPSO
    posiciones = np.random.uniform(p_min, p_max, (n_particulas, n_variables))
    velocidades = np.random.randn(n_particulas, n_variables) * 5
    pbest = posiciones.copy()
    gbest = posiciones[0].copy()
    
    def fitness(p):
        costo = np.sum(costos_a * p**2 + costos_b * p + costos_c)
        penalty = 1000 * (np.sum(p) - demanda)**2
        return costo + penalty
    
    # Evolución
    mejor_fitness = np.full(n_iteraciones, 1e10)
    fitness_medio = np.full(n_iteraciones, 1e10)
    
    for it in range(n_iteraciones):
        fits = np.array([fitness(posiciones[i]) for i in range(n_particulas)])
        fitness_medio[it] = np.mean(fits)
        
        for i in range(n_particulas):
            if fits[i] < fitness(pbest[i]):
                pbest[i] = posiciones[i].copy()
        
        best_idx = np.argmin([fitness(pbest[i]) for i in range(n_particulas)])
        if fitness(pbest[best_idx]) < fitness(gbest):
            gbest = pbest[best_idx].copy()
        
        mejor_fitness[it] = fitness(gbest)
        
        # DEEPSO update
        w = 0.7 - 0.3 * it / n_iteraciones
        c1, c2 = 1.5, 1.5
        r1, r2 = np.random.rand(n_variables), np.random.rand(n_variables)
        
        velocidades = w * velocidades + c1 * r1 * (pbest - posiciones) + c2 * r2 * (gbest - posiciones)
        # Mutación diferencial
        mutacion = 0.3 * np.random.randn(n_particulas, n_variables)
        velocidades += mutacion
        
        posiciones += velocidades
        posiciones = np.clip(posiciones, p_min, p_max)
    
    # Resultado final
    gen_optima = gbest
    costo_optimo = np.sum(costos_a * gen_optima**2 + costos_b * gen_optima + costos_c)
    
    return {
        "iteraciones": np.arange(n_iteraciones),
        "mejor_fitness": mejor_fitness,
        "fitness_medio": fitness_medio,
        "generacion_optima_mw": gen_optima,
        "costo_optimo_usd_h": costo_optimo,
        "generacion_total_mw": np.sum(gen_optima),
        "demanda_mw": demanda,
        "error_balance_mw": abs(np.sum(gen_optima) - demanda),
    }

def pronostico_demanda_24h(demanda_base_mw=100, tipo_dia="laboral", seed=42):
    """Pronóstico de demanda con series de tiempo (Fourier + ruido)"""
    np.random.seed(seed)
    t = np.linspace(0, 48, 192)  # 48 horas, paso 15 min
    
    # Componentes de Fourier
    base = demanda_base_mw * 0.6
    fundamental = demanda_base_mw * 0.25 * np.sin(2 * np.pi * t / 24 - np.pi/3)
    armonico2 = demanda_base_mw * 0.08 * np.sin(4 * np.pi * t / 24 + np.pi/4)
    armonico3 = demanda_base_mw * 0.03 * np.sin(6 * np.pi * t / 24)
    
    if tipo_dia == "fin_de_semana":
        base *= 0.85
        fundamental *= 0.8
    
    # Pronóstico
    pronostico = base + fundamental + armonico2 + armonico3
    
    # Generación real (con ruido)
    ruido = np.random.randn(192) * demanda_base_mw * 0.04
    real = pronostico + ruido
    
    # Bandas de confianza
    sigma = demanda_base_mw * 0.05
    banda_sup = pronostico + 1.96 * sigma
    banda_inf = pronostico - 1.96 * sigma
    
    # Métricas
    error = real - pronostico
    rmse = np.sqrt(np.mean(error**2))
    mape = np.mean(np.abs(error / real)) * 100
    
    return {
        "horas": t,
        "pronostico": pronostico,
        "real": real,
        "banda_sup": banda_sup,
        "banda_inf": banda_inf,
        "error": error,
        "rmse_mw": rmse,
        "mape_pct": mape,
    }

def simular_monte_carlo_ens(n_simulaciones=1000, demanda_mw=300, gen_renovable_pct=40, seed=42):
    """Simulación Monte Carlo para cuantificar incertidumbre en despacho"""
    np.random.seed(seed)
    
    gen_ren_base = demanda_mw * gen_renovable_pct / 100
    gen_term_base = demanda_mw - gen_ren_base
    
    costos = np.zeros(n_simulaciones)
    curt = np.zeros(n_simulaciones)
    ens = np.zeros(n_simulaciones)
    
    for i in range(n_simulaciones):
        # Incertidumbre en renovable
        factor_ren = np.random.triangular(0.5, 1.0, 1.5)
        gen_ren = gen_ren_base * factor_ren
        
        # Incertidumbre en demanda
        factor_dem = np.random.normal(1.0, 0.05)
        dem = demanda_mw * factor_dem
        
        # Incertidumbre en precio
        precio = np.random.lognormal(np.log(50), 0.3)
        
        # Balance
        if gen_ren > dem:
            curtailment = gen_ren - dem
            ens_val = 0
            costo = gen_term_base * precio * 0.5 + curtailment * precio * 0.2
        else:
            curtailment = 0
            deficit = dem - gen_ren - gen_term_base
            ens_val = max(0, deficit)
            costo = (gen_term_base + min(deficit, gen_term_base * 0.3)) * precio + ens_val * 500
        
        costos[i] = costo
        curt[i] = curtailment
        ens[i] = ens_val
    
    return {
        "costos": costos,
        "curtailment": curt,
        "ens": ens,
        "costo_medio": np.mean(costos),
        "costo_std": np.std(costos),
        "costo_cvar_95": np.mean(np.sort(costos)[int(0.95 * n_simulaciones):]),
        "curt_medio_mw": np.mean(curt),
        "ens_medio_mw": np.mean(ens),
        "n_simulaciones": n_simulaciones,
    }
