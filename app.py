"""
╔══════════════════════════════════════════════════════════════════════════╗
║  LICA-SE v2.0 · Laboratorio de Inteligencia Computacional               ║
║  Aplicada al Sector Eléctrico                                           ║
║  Director: Prof. Sergio Raúl Rivera Rodríguez, PhD, SMIEEE              ║
║  Universidad Nacional de Colombia — Sede Bogotá                         ║
║  ★ BANDERA: Uncertainty Cost Functions (UCF) ★                         ║
║  Versión 2: UCF central + TODO el contenido de v1 preservado           ║
╚══════════════════════════════════════════════════════════════════════════╝
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from data import (PERFIL, COOPERACION, PUBLICACIONES, Q_RESUMEN, NA_DESGLOSE,
                  APPS_GITHUB, TESIS, OSS_TOOLS)
from simulations import (simular_opf_simplificado, simular_microgrid_despacho,
                          simular_gestion_activos, simular_deepso,
                          pronostico_demanda_24h, simular_monte_carlo_ens)
from ucf_engine import (ucf_from_samples, ucf_analytical_normal,
                         ucf_analytical_uniform, optimal_schedule_uniform,
                         solar_beta_samples, wind_normal_samples,
                         ev_gaussian_samples, hydrogen_ucf_simulation, 
                         cslcoe_simulation, monte_carlo_animation_frames)

st.set_page_config(page_title="LICA-SE · UCF Laboratory", page_icon="⚡",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 100%); color: #1a1a2e; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0d1b3e 0%, #1a237e 100%); color: #e0e0e0; }
[data-testid="stSidebar"] .stMarkdown { color: #e0e0e0; }
h1, h2, h3, h4 { color: #1a237e !important; }
.stMetric { background: white; border-radius: 12px; padding: 15px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08); border-left: 4px solid #1a237e; }
.stMetric label { color: #555 !important; font-size: 0.85rem !important; }
.stMetric [data-testid="stMetricValue"] { color: #1a237e !important; font-weight: 700 !important; }
.stTabs [data-baseweb="tab-list"] { gap: 3px; flex-wrap: wrap; }
.stTabs [data-baseweb="tab"] { padding: 7px 12px; border-radius: 8px 8px 0 0;
    background: #e8ecf1; color: #333; font-size: 0.75rem; font-weight: 600; }
.stTabs [aria-selected="true"] { background: #1a237e !important; color: white !important; }
div.info-box { background: linear-gradient(135deg, #E8EAF6, #C5CAE9); padding: 15px 20px;
    border-radius: 10px; border-left: 5px solid #1a237e; margin: 10px 0; }
div.ucf-box { background: linear-gradient(135deg, #FFF3E0, #FFE0B2); padding: 15px 20px;
    border-radius: 10px; border-left: 5px solid #E65100; margin: 10px 0; }
div.success-box { background: linear-gradient(135deg, #E8F5E9, #C8E6C9); padding: 15px 20px;
    border-radius: 10px; border-left: 5px solid #2E7D32; margin: 10px 0; }
div.competition-box { background: linear-gradient(135deg, #FFF8E1, #FFD54F); padding: 12px 16px;
    border-radius: 10px; border-left: 5px solid #F57F17; margin: 8px 0; }
</style>
""", unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:10px 0;">
        <div style="font-size:2.5rem;">⚡🧠</div>
        <div style="font-size:1.2rem; font-weight:800; color:#7C4DFF;">LICA-SE</div>
        <div style="font-size:0.65rem; color:#aaa; letter-spacing:1px;">LABORATORIO DE INTELIGENCIA<br>COMPUTACIONAL — SECTOR ELÉCTRICO</div>
        <div style="font-size:0.7rem; color:#FFD54F; margin-top:5px;">★ UCF: Uncertainty Cost Functions ★</div>
    </div>
    <hr style="border-color:#3949AB;">
    """, unsafe_allow_html=True)
    st.markdown("### 👤 Director")
    st.markdown("**Prof. Sergio R. Rivera**\nPhD, SMIEEE\nUNAL — Bogotá\nMIT · Khalifa Univ.")
    st.markdown("### 🔗 Enlaces")
    st.markdown(f"- [Scholar]({PERFIL['scholar']}) · [GitHub]({PERFIL['github']})\n- [Hermes-UNAL]({PERFIL['hermes']})")
    st.markdown("### 🏆 Competencias Ganadas")
    st.markdown("- 🥇 IEEE 2017 Smart Grid\n- 🥇 IEEE 2018 Grid Opt.\n- 🥇 IEEE 2020 Expansion\n- 🥇 IEEE 2025 AI-ccelerating\n- 🥇 WCCI 2018 Competition")
    st.markdown("---")
    st.markdown('<div style="text-align:center; font-size:0.65rem; color:#7C4DFF;">Grupo EMC-UN · UNAL 2026</div>', unsafe_allow_html=True)

# TABS
tab_names = [
    "🏠 Inicio", "🎯 UCF Lab", "🔬 UCF Solar+Eólica",
    "💚 UCF Hidrógeno", "💰 CS-LCOE", "🧪 Simulaciones",
    "📊 Publicaciones", "💻 Apps GitHub",
    "🌍 Colab.", "🏆 Logros", "👤 Perfil", "📝 Únete"
]
tabs = st.tabs(tab_names)

# ═══════════════════════════════════════════════════════════
# TAB 0: INICIO
# ═══════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("# ⚡ LICA-SE — Laboratorio de Inteligencia Computacional")
    st.markdown(f"""
    <div class="info-box">
    <b>Bienvenido al LICA-SE</b> — La plataforma interactiva del grupo de investigación del 
    <b>Prof. Sergio Raúl Rivera Rodríguez</b> (PhD, SMIEEE) en la Universidad Nacional de Colombia.
    <br><br>
    <b>★ Nuestra bandera: las Uncertainty Cost Functions (UCF) ★</b><br>
    Somos pioneros mundiales en la <b>cuantificación matemática del costo de la incertidumbre</b> 
    en sistemas eléctricos con renovables. Las UCF permiten transformar la variabilidad solar, 
    eólica, de demanda y vehículos eléctricos en <b>costos económicos cuantificables</b> que se 
    integran directamente en despacho económico, OPF y planificación.
    <br><br>
    <i>Esta es una muestra representativa de nuestro trabajo. El grupo tiene +{PERFIL['publicaciones']} 
    publicaciones, +{PERFIL['citas']:,} citas, y múltiples desarrollos adicionales no mostrados aquí.</i>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Publicaciones", f"{PERFIL['publicaciones']}+", f"{Q_RESUMEN['Q1']} en Q1")
    c2.metric("Citas Scholar", f"{PERFIL['citas']:,}+")
    c3.metric("h-index", f"{PERFIL['h_index']}")
    c4.metric("Competencias IEEE", "4× Ganador")
    c5.metric("Países Colab.", f"{len(COOPERACION)}")
    
    st.markdown("---")
    st.markdown("### 🎯 ¿Qué son las Uncertainty Cost Functions (UCF)?")
    
    st.latex(r"""
    UCF(W_s) = \mathbb{E}\left[C_u \cdot \max(W_{av} - W_s,\; 0) + C_o \cdot \max(W_s - W_{av},\; 0)\right]
    """)
    
    st.markdown("""
    <div class="ucf-box">
    <b>Interpretación:</b><br>
    • <b>W_s</b> = potencia programada (lo que prometes inyectar)<br>
    • <b>W_av</b> = potencia real disponible (lo que realmente generas — incierto)<br>
    • <b>C_u</b> = costo por subestimar (excedente → curtailment, vertimiento)<br>
    • <b>C_o</b> = costo por sobrestimar (déficit → reserva, compra de emergencia)<br>
    <br>
    <b>¿Por qué importan?</b> Con la masificación de renovables + BESS, las UCF son la herramienta 
    matemática que convierte la incertidumbre climática en un costo real dentro del despacho económico.
    Esto es <b>crítico</b> para operadores de red, planificadores y mercados eléctricos.
    </div>
    """, unsafe_allow_html=True)
    
    # Demo rápida interactiva
    st.markdown("### 🎮 Demo Rápida: Arrastra el slider y observa las UCF en acción")
    st.markdown("""
    <div class="info-box">
    <b>Nota:</b> La <b>potencia programada Ps</b> típicamente proviene de un proceso de 
    <b>Programación de la Operación del Sistema</b> (OPF — Optimal Power Flow, UC — Unit Commitment, 
    o despacho económico). Es el resultado de una formulación de optimización que determina 
    cuánta potencia debe inyectar cada generador en cada período.
    </div>
    """, unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        demo_Ps = st.slider("Potencia programada Ps (MW)", 10, 90, 50, key="demo_ps")
    with c2:
        demo_Cu = st.slider("Costo subestimar Cu ($/MW)", 5, 100, 30, key="demo_cu")
    with c3:
        demo_Co = st.slider("Costo sobrestimar Co ($/MW)", 5, 100, 70, key="demo_co")
    
    # Calcular UCF analítica vs Ps
    Ps_range = np.linspace(10, 90, 100)
    ucf_vals = [ucf_analytical_uniform(20, 80, ps, demo_Cu, demo_Co)[0] for ps in Ps_range]
    Ps_opt = optimal_schedule_uniform(20, 80, demo_Cu, demo_Co)
    
    # Monte Carlo para validar
    rng_demo = np.random.default_rng(42)
    demo_samples = rng_demo.uniform(20, 80, 100000)
    demo_mc_result = ucf_from_samples(demo_Ps, demo_samples, demo_Cu, demo_Co)
    demo_analytical = ucf_analytical_uniform(20, 80, demo_Ps, demo_Cu, demo_Co)
    demo_error_pct = abs(demo_mc_result.expected_cost - demo_analytical[0]) / demo_analytical[0] * 100
    
    fig_demo = go.Figure()
    fig_demo.add_trace(go.Scatter(x=Ps_range, y=ucf_vals, name="E[UCF] Analítico", 
        line=dict(color="#E65100", width=3), fill="tozeroy", fillcolor="rgba(230,81,0,0.1)"))
    fig_demo.add_vline(x=demo_Ps, line_dash="dash", line_color="#1a237e",
        annotation_text=f"Ps={demo_Ps} MW (de OPF/UC)")
    fig_demo.add_vline(x=Ps_opt, line_dash="dot", line_color="#2E7D32",
        annotation_text=f"Óptimo={Ps_opt:.1f}")
    fig_demo.update_layout(height=350, title="UCF vs Potencia Programada (Uniforme [20,80] MW)",
        xaxis_title="Potencia Programada Ps (MW) — resultado de OPF/UC", yaxis_title="E[UCF] ($/h)",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.8)",
        font=dict(color="#1a1a2e"))
    st.plotly_chart(fig_demo, use_container_width=True, key="fig_demo_ucf")
    
    ucf_val = ucf_analytical_uniform(20, 80, demo_Ps, demo_Cu, demo_Co)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("E[UCF] Analítico", f"${ucf_val[0]:.2f}/h")
    c2.metric("E[UCF] Monte Carlo", f"${demo_mc_result.expected_cost:.2f}/h")
    c3.metric("Error MC vs Analítico", f"{demo_error_pct:.3f}%", "100K escenarios")
    c4.metric("Costo subestimar", f"${ucf_val[1]:.2f}/h")
    c5.metric("Costo sobrestimar", f"${ucf_val[2]:.2f}/h")
    
    st.markdown("""
    <div class="success-box">
    <b>✅ Validación cruzada:</b> El error entre el costo esperado calculado <b>analíticamente</b> 
    (fórmula cerrada) y el valor esperado con la <b>simulación de Monte Carlo</b> (100,000 escenarios) 
    es extremadamente pequeño (~{:.3f}%), lo que demuestra la precisión de la formulación analítica.
    </div>
    """.format(demo_error_pct), unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# TAB 1: PERFIL COMPLETO (de v1)
# ═══════════════════════════════════════════════════════════
with tabs[10]:
    st.markdown("# 👤 Perfil del Director")
    
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown(f"""
        ### {PERFIL['nombre']}
        **{PERFIL['titulo']}**  
        {PERFIL['institucion']} · {PERFIL['departamento']} · {PERFIL['facultad']}
        
        **Email:** [{PERFIL['email']}](mailto:{PERFIL['email']})  
        **Google Scholar:** [{PERFIL['scholar']}]({PERFIL['scholar']})  
        **GitHub:** [{PERFIL['github']}]({PERFIL['github']})  
        **Hermes UNAL:** [{PERFIL['hermes']}]({PERFIL['hermes']})
        """)
    with c2:
        st.metric("Citas", f"{PERFIL['citas']:,}+")
        st.metric("h-index", PERFIL['h_index'])
        st.metric("Publicaciones", f"{PERFIL['publicaciones']}+")
    
    st.markdown("---")
    st.markdown("### 🎓 Formación Académica")
    for f in PERFIL['formacion']:
        st.markdown(f"- {f}")
    
    st.markdown("### 💼 Experiencia Profesional")
    for e in PERFIL['experiencia_previa']:
        st.markdown(f"- {e}")
    
    st.markdown("### 🏛️ Cargos Académicos")
    for c in PERFIL['cargos']:
        st.markdown(f"- {c}")
    
    st.markdown("### 🔬 Líneas de Investigación")
    for l in PERFIL['lineas_investigacion']:
        st.markdown(f"- {l}")
    
    st.markdown("### 🎓 Cursos y Certificaciones Especializadas")
    st.markdown("""
    - **MIT MicroMaster en Ciencia de Datos y Estadística** — Massachusetts Institute of Technology
    - **IEEE Battery Energy Storage Systems (BESS)** — Especialización profesional IEEE
    - **MIT Colaboración Academia-Industria** — Massachusetts Institute of Technology
    """)

# ═══════════════════════════════════════════════════════════
# TAB 2: UCF LAB — Laboratorio Central
# ═══════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("# 🎯 UCF Laboratory — Motor de Simulación de Incertidumbre")
    st.markdown("""
    <div class="ucf-box">
    <b>Laboratorio interactivo de Uncertainty Cost Functions.</b><br>
    Ajuste todos los parámetros y observe cómo la incertidumbre se traduce en costos económicos.
    Compare el método <b>analítico</b> (cerrado) con <b>Monte Carlo</b> (estocástico).
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📐 Formulación Matemática Fundamental")
    st.latex(r"""
    E[C_{unc}] = \int \left[ C_u \cdot \max(X - x, 0) + C_o \cdot \max(x - X, 0) \right] f_X(X) \, dX
    """)
    st.markdown("Donde $f_X(X)$ es la PDF de la variable real (generación solar, eólica, demanda, VE).")
    
    st.markdown("---")
    ucf_tabs = st.tabs(["📊 Normal", "📊 Uniforme", "☀️ Beta (Solar)", "💨 Wind", "🔋 EV", "⚡ Comparar"])
    
    with ucf_tabs[0]:
        st.markdown("### Distribución Normal — Generación Renovable")
        c1, c2, c3 = st.columns(3)
        with c1:
            mu_n = st.slider("μ (media MW)", 10, 100, 50, key="ucf_n_mu")
            sig_n = st.slider("σ (desviación MW)", 1, 30, 10, key="ucf_n_sig")
        with c2:
            Ps_n = st.slider("Ps (programada MW)", 10, 100, 55, key="ucf_n_ps")
            Cu_n = st.slider("Cu ($/MW)", 5, 100, 30, key="ucf_n_cu")
        with c3:
            Co_n = st.slider("Co ($/MW)", 5, 100, 70, key="ucf_n_co")
            N_n = st.slider("N escenarios", 1000, 500000, 100000, 10000, key="ucf_n_N")
        
        rng = np.random.default_rng(42)
        samples_n = np.clip(rng.normal(mu_n, sig_n, N_n), 0, 200)
        r_mc = ucf_from_samples(Ps_n, samples_n, Cu_n, Co_n)
        r_an = ucf_analytical_normal(mu_n, sig_n, Ps_n, Cu_n, Co_n)
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MC Expected Cost", f"${r_mc.expected_cost:.2f}")
        c2.metric("Analytical Cost", f"${r_an:.2f}")
        c3.metric("Error MC vs Analítico", f"{abs(r_mc.expected_cost-r_an)/r_an*100:.2f}%")
        c4.metric("P(déficit)", f"{r_mc.prob_downward:.1%}")
        
        fig_n = make_subplots(rows=2, cols=2,
            subplot_titles=["Potencia (MW)", "Costo Subestimar Cu", "Costo Sobrestimar Co", "UCF Total"])
        fig_n.add_trace(go.Histogram(x=samples_n[:2000], nbinsx=60, name="Potencia", marker_color="#1565C0"), row=1, col=1)
        fig_n.add_vline(x=Ps_n, line_dash="dash", line_color="red", row=1, col=1)
        fig_n.add_trace(go.Histogram(x=r_mc.costs[:2000]*np.where(samples_n[:2000]>Ps_n,1,0)[:2000], 
            nbinsx=60, name="Cu", marker_color="#43A047"), row=1, col=2)
        fig_n.add_trace(go.Histogram(x=r_mc.costs[:2000]*np.where(samples_n[:2000]<=Ps_n,1,0)[:2000], 
            nbinsx=60, name="Co", marker_color="#C62828"), row=2, col=1)
        fig_n.add_trace(go.Histogram(x=r_mc.costs[:2000], nbinsx=60, name="UCF", marker_color="#6A1B9A"), row=2, col=2)
        fig_n.update_layout(height=500, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.8)", font=dict(color="#1a1a2e"), showlegend=False)
        st.plotly_chart(fig_n, use_container_width=True, key="fig_ucf_normal")
        
        # Percentiles de riesgo
        st.markdown("### 📊 Percentiles de Riesgo")
        pct = np.percentile(r_mc.costs, [50, 90, 95, 99])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("P50 (mediana)", f"${pct[0]:.2f}")
        c2.metric("P90", f"${pct[1]:.2f}")
        c3.metric("P95 (CVaR)", f"${pct[2]:.2f}")
        c4.metric("P99 (peor caso)", f"${pct[3]:.2f}")
        
        # Animación GIF del caso base
        st.markdown("---")
        st.markdown("### 🎬 Animación: UCF Caso Base")
        st.markdown("""
        <div class="ucf-box">
        Animación del caso base de las Uncertainty Cost Functions: generación renovable vs 
        potencia programada, excedentes/déficits y costo de incertidumbre acumulado.
        </div>
        """, unsafe_allow_html=True)
        try:
            st.image(os.path.join(os.path.dirname(__file__), "ucf_caso_base_small.gif"),
                     caption="UCF Caso Base — Generación renovable, potencia programada y costos de incertidumbre",
                     use_container_width=True)
        except Exception:
            st.info("📽️ Animación GIF disponible al ejecutar localmente: `ucf_caso_base_small.gif`")

    with ucf_tabs[1]:
        st.markdown("### Distribución Uniforme — Caso sin datos históricos")
        c1, c2, c3 = st.columns(3)
        with c1:
            Pmin = st.slider("P_min (MW)", 0, 80, 20, key="ucf_u_min")
            Pmax = st.slider("P_max (MW)", 20, 150, 80, key="ucf_u_max")
        with c2:
            Ps_u = st.slider("Ps (MW)", 0, 150, 50, key="ucf_u_ps")
            Cu_u = st.slider("Cu ($/MW)", 5, 500, 300, key="ucf_u_cu")
        with c3:
            Co_u = st.slider("Co ($/MW)", 5, 1000, 700, key="ucf_u_co")
        
        r_u = ucf_analytical_uniform(Pmin, Pmax, Ps_u, Cu_u, Co_u)
        Ps_opt_u = optimal_schedule_uniform(Pmin, Pmax, Cu_u, Co_u)
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("E[UCF]", f"${r_u[0]:.2f}")
        c2.metric("E[Cu]", f"${r_u[1]:.2f}")
        c3.metric("E[Co]", f"${r_u[2]:.2f}")
        c4.metric("Ps óptimo", f"{Ps_opt_u:.1f} MW")
        
        # Curva UCF vs Ps
        Ps_range_u = np.linspace(Pmin, Pmax, 100)
        ucf_curve = [ucf_analytical_uniform(Pmin, Pmax, ps, Cu_u, Co_u)[0] for ps in Ps_range_u]
        
        fig_u = go.Figure()
        fig_u.add_trace(go.Scatter(x=Ps_range_u, y=ucf_curve, name="E[UCF]",
            line=dict(color="#E65100", width=3), fill="tozeroy", fillcolor="rgba(230,81,0,0.1)"))
        fig_u.add_vline(x=Ps_u, line_dash="dash", line_color="#1a237e",
            annotation_text=f"Ps={Ps_u}")
        fig_u.add_vline(x=Ps_opt_u, line_dash="dot", line_color="#2E7D32",
            annotation_text=f"Ópt={Ps_opt_u:.1f}")
        fig_u.update_layout(height=400, title="UCF vs Ps (Uniforme)",
            xaxis_title="Ps (MW)", yaxis_title="E[UCF] ($)",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.8)",
            font=dict(color="#1a1a2e"))
        st.plotly_chart(fig_u, use_container_width=True, key="fig_ucf_uniform")
        
        st.latex(r"""W_s^* = \frac{C_u \cdot P_{max} + C_o \cdot P_{min}}{C_u + C_o}""")
        st.markdown(f"**Programación óptima:** Ps* = ({Cu_u}×{Pmax} + {Co_u}×{Pmin}) / ({Cu_u}+{Co_u}) = **{Ps_opt_u:.1f} MW**")

    with ucf_tabs[2]:
        st.markdown("### ☀️ Distribución Beta — Generación Solar Fotovoltaica")
        c1, c2, c3 = st.columns(3)
        with c1:
            cap_s = st.slider("Capacidad solar (MW)", 10, 200, 100, key="ucf_s_cap")
            alpha_s = st.slider("α (Beta)", 0.5, 10.0, 4.0, 0.5, key="ucf_s_a")
        with c2:
            beta_s = st.slider("β (Beta)", 0.5, 10.0, 2.5, 0.5, key="ucf_s_b")
            Ps_s = st.slider("Ps (MW)", 10, 200, 65, key="ucf_s_ps")
        with c3:
            Cu_s = st.slider("Cu ($/MW)", 1, 50, 8, key="ucf_s_cu")
            Co_s = st.slider("Co ($/MW)", 1, 100, 35, key="ucf_s_co")
        
        solar_s = solar_beta_samples(cap_s, alpha_s, beta_s, 100000, 42)
        r_s = ucf_from_samples(Ps_s, solar_s, Cu_s, Co_s)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("E[UCF] solar", f"${r_s.expected_cost:.2f}/h")
        c2.metric("Gen media", f"{np.mean(solar_s):.1f} MW")
        c3.metric("P(excedente)", f"{r_s.prob_upward:.1%}")
        
        fig_s = make_subplots(rows=1, cols=2,
            subplot_titles=["Distribución Solar (Beta)", "UCF vs Dispatch"])
        fig_s.add_trace(go.Histogram(x=solar_s[:5000], nbinsx=80, name="Solar",
            marker_color="#FFB300"), row=1, col=1)
        fig_s.add_vline(x=Ps_s, line_dash="dash", line_color="red", row=1, col=1)
        
        Ps_range_s = np.linspace(0, cap_s, 50)
        ucf_s = [ucf_from_samples(ps, solar_s, Cu_s, Co_s).expected_cost for ps in Ps_range_s]
        fig_s.add_trace(go.Scatter(x=Ps_range_s, y=ucf_s, name="UCF",
            line=dict(color="#E65100", width=2), fill="tozeroy",
            fillcolor="rgba(230,81,0,0.1)"), row=1, col=2)
        fig_s.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.8)", font=dict(color="#1a1a2e"))
        st.plotly_chart(fig_s, use_container_width=True, key="fig_ucf_solar")

    with ucf_tabs[3]:
        st.markdown("### 💨 Distribución Normal Truncada — Generación Eólica")
        c1, c2, c3 = st.columns(3)
        with c1:
            mu_w = st.slider("μ viento (MW)", 10, 100, 50, key="ucf_w_mu")
            sig_w = st.slider("σ viento (MW)", 2, 30, 15, key="ucf_w_sig")
        with c2:
            cap_w = st.slider("Capacidad (MW)", 50, 200, 100, key="ucf_w_cap")
            Ps_w = st.slider("Ps (MW)", 10, 150, 55, key="ucf_w_ps")
        with c3:
            Cu_w = st.slider("Cu ($/MW)", 1, 50, 10, key="ucf_w_cu")
            Co_w = st.slider("Co ($/MW)", 1, 100, 40, key="ucf_w_co")
        
        wind_s = wind_normal_samples(mu_w, sig_w, cap_w, 100000, 42)
        r_w = ucf_from_samples(Ps_w, wind_s, Cu_w, Co_w)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("E[UCF] eólica", f"${r_w.expected_cost:.2f}/h")
        c2.metric("Gen media", f"{np.mean(wind_s):.1f} MW")
        c3.metric("P(déficit)", f"{r_w.prob_downward:.1%}")
        
        fig_w = go.Figure()
        fig_w.add_trace(go.Histogram(x=wind_s[:5000], nbinsx=80, name="Eólica",
            marker_color="#1565C0"))
        fig_w.add_vline(x=Ps_w, line_dash="dash", line_color="red",
            annotation_text=f"Ps={Ps_w}")
        fig_w.update_layout(height=350, title="Distribución Eólica + Ps",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.8)",
            font=dict(color="#1a1a2e"))
        st.plotly_chart(fig_w, use_container_width=True, key="fig_ucf_wind")

    with ucf_tabs[4]:
        st.markdown("### 🔋 Vehículos Eléctricos — Demanda Incierta")
        c1, c2, c3 = st.columns(3)
        with c1:
            n_ev = st.slider("N° vehículos", 100, 5000, 1000, 100, key="ucf_ev_n")
            mu_ev = st.slider("μ energía (kWh)", 5, 50, 20, key="ucf_ev_mu")
        with c2:
            sig_ev = st.slider("σ energía (kWh)", 1, 15, 5, key="ucf_ev_sig")
            Ps_ev = st.slider("Programado (kWh/veh)", 5, 50, 18, key="ucf_ev_ps")
        with c3:
            Cu_ev = st.slider("Cu ($/kWh)", 1, 20, 3, key="ucf_ev_cu")
            Co_ev = st.slider("Co ($/kWh)", 1, 30, 12, key="ucf_ev_co")
        
        ev_s = ev_gaussian_samples(mu_ev, sig_ev, n_ev, 42)
        r_ev = ucf_from_samples(Ps_ev, ev_s, Cu_ev, Co_ev)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("E[UCF] por vehículo", f"${r_ev.expected_cost:.2f}")
        c2.metric("Costo total flota", f"${r_ev.expected_cost*n_ev:.0f}")
        c3.metric("P(exceso demanda)", f"{r_ev.prob_upward:.1%}")
        
        fig_ev = go.Figure()
        fig_ev.add_trace(go.Histogram(x=ev_s, nbinsx=60, name="Demanda EV",
            marker_color="#9C27B0"))
        fig_ev.add_vline(x=Ps_ev, line_dash="dash", line_color="red",
            annotation_text=f"Programado={Ps_ev}")
        fig_ev.update_layout(height=350, title="Demanda de VE por vehículo",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.8)",
            font=dict(color="#1a1a2e"))
        st.plotly_chart(fig_ev, use_container_width=True, key="fig_ucf_ev")

    with ucf_tabs[5]:
        st.markdown("### ⚡ Comparación de Escenarios de Penalización")
        c1, c2 = st.columns(2)
        with c1:
            comp_mu = st.slider("μ generación (MW)", 20, 80, 50, key="ucf_c_mu")
            comp_sig = st.slider("σ generación (MW)", 2, 20, 10, key="ucf_c_sig")
        with c2:
            comp_Ps = st.slider("Ps (MW)", 20, 80, 55, key="ucf_c_ps")
            comp_N = st.slider("N escenarios", 5000, 200000, 50000, 5000, key="ucf_c_N")
        
        scenarios = {
            "Conservador (Cu=12, Co=50)": (12, 50),
            "Equilibrado (Cu=20, Co=20)": (20, 20),
            "Excedente costoso (Cu=40, Co=10)": (40, 10),
            "Déficit crítico (Cu=5, Co=80)": (5, 80),
        }
        
        rng_c = np.random.default_rng(42)
        samples_c = np.clip(rng_c.normal(comp_mu, comp_sig, comp_N), 0, 150)
        
        results = {}
        for name, (cu, co) in scenarios.items():
            results[name] = ucf_from_samples(comp_Ps, samples_c, cu, co)
        
        df_comp = pd.DataFrame({
            "Escenario": list(scenarios.keys()),
            "E[UCF]": [f"${results[n].expected_cost:.2f}" for n in scenarios],
            "E[Sub]": [f"${results[n].expected_upward:.2f}" for n in scenarios],
            "E[Sobre]": [f"${results[n].expected_downward:.2f}" for n in scenarios],
            "P(excedente)": [f"{results[n].prob_upward:.1%}" for n in scenarios],
            "P(déficit)": [f"{results[n].prob_downward:.1%}" for n in scenarios],
        })
        st.dataframe(df_comp, use_container_width=True, hide_index=True)
        
        # Animación de comparación de escenarios
        st.markdown("### 🎬 Animación: Evolución de Costos por Escenario")
        st.markdown("""
        <div class="info-box">
        Esta animación muestra cómo varían los costos de incertidumbre bajo diferentes 
        políticas de penalización. Observe cómo el escenario <b>Conservador</b> minimiza 
        el costo esperado total, mientras que el escenario <b>Déficit crítico</b> penaliza 
        fuertemente las desviaciones negativas.
        </div>
        """, unsafe_allow_html=True)
        
        # Crear animación con frames
        frames_anim = []
        for i, (scenario_name, (cu, co)) in enumerate(scenarios.items()):
            result = results[scenario_name]
            frame = go.Frame(
                data=[
                    go.Bar(
                        x=["Costo Subestimar", "Costo Sobrestimar", "Costo Total"],
                        y=[result.expected_upward * cu, result.expected_downward * co, result.expected_cost],
                        marker_color=["#43A047", "#C62828", "#1a237e"],
                        name=scenario_name
                    )
                ],
                name=f"frame_{i}"
            )
            frames_anim.append(frame)
        
        fig_comp_anim = go.Figure(
            data=[frames_anim[0].data[0]],
            layout=go.Layout(
                title="Comparación de Escenarios de Penalización",
                yaxis=dict(range=[0, max(r.expected_cost for r in results.values()) * 1.2], title="Costo ($/h)"),
                updatemenus=[dict(
                    type="buttons",
                    showactive=False,
                    buttons=[dict(
                        label="Play",
                        method="animate",
                        args=[None, dict(frame=dict(duration=1500, redraw=True), fromcurrent=True)]
                    )]
                )]
            ),
            frames=frames_anim
        )
        fig_comp_anim.update_layout(height=400, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.8)", font=dict(color="#1a1a2e"))
        st.plotly_chart(fig_comp_anim, use_container_width=True, key="fig_comp_anim")
        
        # Animación de decisión UCF completa
        st.markdown("---")
        st.markdown("### 🎬 Animación: UCF para Toma de Decisiones")
        st.markdown("""
        <div class="success-box">
        <b>De la teoría a la decisión:</b> Esta animación muestra cómo las UCF no solo cuantifican 
        la incertidumbre, sino que guían decisiones operativas. Incluye validación analítica vs 
        Monte Carlo, convergencia, descomposición de costos y la <b>Ps óptima</b> que minimiza 
        el costo esperado de incertidumbre.
        </div>
        """, unsafe_allow_html=True)
        st.image(os.path.join(os.path.dirname(__file__), "01_ucf_decision_completa_small.gif"),
                 caption="UCF Decision Framework — Validación, convergencia y programación óptima para minimizar costos de incertidumbre",
                 use_container_width=True)
        st.markdown("""
        <div class="info-box">
        <b>Aplicación práctica:</b> Operadores de red y planificadores pueden usar este framework para:
        <br>• Determinar la potencia programada óptima que minimiza costos de desbalance
        <br>• Validar que las simulaciones Monte Carlo convergen al valor analítico
        <br>• Descomponer costos entre déficit y excedente para priorizar acciones
        <br>• Tomar decisiones informadas sobre inversión en BESS o flexibilidad de demanda
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# TAB 3: UCF SOLAR + EÓLICA COMBINADO
# ═══════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("# 🔬 UCF Solar + Eólica — Validación Analítico vs Monte Carlo")
    st.markdown("""
    <div class="ucf-box">
    <b>Validación cruzada:</b> Se compara el valor analítico cerrado (fórmula con erf) contra 
    la simulación Monte Carlo con 1,000,000 de escenarios. Basado en las animaciones del grupo.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📐 Desarrollo Analítico (Distribución Normal)")
    st.latex(r"""
    d = \frac{\mu - P_s}{\sigma\sqrt{2}}
    """)
    st.latex(r"""
    C_u^a = \frac{C_u}{2}(\mu - P_s)(1 + \text{erf}(d)) + \frac{C_u \sigma}{\sqrt{2\pi}} e^{-d^2}
    """)
    st.latex(r"""
    C_o^a = -\frac{C_o \sigma}{\sqrt{2\pi}}\left(e^{-\mu^2/(2\sigma^2)} - e^{-d^2}\right) + \frac{C_o}{2}(P_s - \mu)\left(\text{erf}\left(\frac{\mu}{\sigma\sqrt{2}}\right) - \text{erf}(d)\right)
    """)
    
    st.markdown("---")
    st.markdown("### 🎮 Simulación Interactiva (basada en animación del grupo)")
    st.markdown("""
    <div class="info-box">
    <b>Nota:</b> La <b>línea roja</b> en los histogramas representa la potencia programada <b>Ps</b>, 
    que proviene de un proceso de <b>Programación de la Operación</b> — es decir, del resultado 
    de una formulación de optimización (OPF, UC, despacho económico) y su respectiva solución. 
    Ps es la decisión óptima que minimiza el costo total incluyendo las UCF.
    </div>
    """, unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        mu_pv = st.slider("μ PV (MW)", 10.0, 30.0, 19.54, 0.1, key="anim_mu")
        sig_pv = st.slider("σ PV", 0.1, 3.0, 0.54, 0.01, key="anim_sig")
    with c2:
        mu_mgp = st.slider("μ MGP (MW)", 0.0, 2.0, 0.5, 0.1, key="anim_mgp_mu")
        sig_mgp = st.slider("σ MGP", 0.01, 0.5, 0.1, 0.01, key="anim_mgp_sig")
    with c3:
        Ps_anim = st.slider("Ps (MW)", 10, 30, 20, key="anim_ps")
        Cu_anim = st.slider("Cu", 5, 50, 30, key="anim_cu")
        Co_anim = st.slider("Co", 10, 100, 70, key="anim_co")
    
    mu_eff = mu_pv + mu_mgp
    sig_eff = np.sqrt(sig_pv**2 + sig_mgp**2)
    
    rng_a = np.random.default_rng(42)
    P_PV = rng_a.normal(mu_pv, sig_pv, 1000000)
    P_MGP = rng_a.normal(mu_mgp, sig_mgp, 1000000)
    P_eff = P_PV + P_MGP
    CU_a = Cu_anim * np.maximum(P_eff - Ps_anim, 0)
    CO_a = Co_anim * np.maximum(Ps_anim - P_eff, 0)
    COST_MC = np.mean(CU_a + CO_a)
    COST_An = ucf_analytical_normal(mu_eff, sig_eff, Ps_anim, Cu_anim, Co_anim)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Analítico", f"${COST_An:.4f}")
    c2.metric("Monte Carlo (1M)", f"${COST_MC:.4f}")
    c3.metric("Error", f"{abs(COST_MC-COST_An)/COST_An*100:.3f}%")
    c4.metric("μ_eff ± σ_eff", f"{mu_eff:.2f} ± {sig_eff:.2f}")
    
    # Histogramas como en la animación
    n_show = st.slider("Escenarios a mostrar", 500, 500000, 5000, 500, key="anim_n")
    
    fig_anim = make_subplots(rows=2, cols=2,
        subplot_titles=["Effective Power Histogram", "Cost Underestimate",
                        "Cost Overestimate", "Total UCF Cost"])
    fig_anim.add_trace(go.Histogram(x=P_eff[:n_show], nbinsx=50, marker_color="#1565C0"), row=1, col=1)
    fig_anim.add_vline(x=Ps_anim, line_dash="dash", line_color="red", row=1, col=1)
    fig_anim.add_trace(go.Histogram(x=CU_a[:n_show], nbinsx=50, marker_color="#43A047"), row=1, col=2)
    fig_anim.add_trace(go.Histogram(x=CO_a[:n_show], nbinsx=50, marker_color="#C62828"), row=2, col=1)
    fig_anim.add_trace(go.Histogram(x=(CU_a+CO_a)[:n_show], nbinsx=50, marker_color="#6A1B9A"), row=2, col=2)
    fig_anim.update_layout(height=500, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.8)", font=dict(color="#1a1a2e"), showlegend=False,
        title_text=f"Monte Carlo ({n_show:,} scenarios) vs Analytical = ${COST_An:.4f}")
    st.plotly_chart(fig_anim, use_container_width=True, key="fig_ucf_anim")
    
    # Animación GIF de incertidumbre
    st.markdown("---")
    st.markdown("### 🎬 Animación: Función de Costo de Incertidumbre")
    st.markdown("""
    <div class="ucf-box">
    Animación completa mostrando: generación renovable simulada (Beta), valor programado,
    excedentes y déficits, costo asociado a cada desviación, costo esperado y escenarios.
    </div>
    """, unsafe_allow_html=True)
    try:
        st.image(os.path.join(os.path.dirname(__file__), "animacion_incertidumbre.gif"),
                 caption="UCF Animation — Generación renovable vs Programado: excedentes, déficits y costos",
                 use_container_width=True)
    except Exception:
        st.info("📽️ Animación GIF disponible al ejecutar localmente: `animacion_incertidumbre.gif`")
    
    st.markdown("---")
    st.markdown("### 🎬 Animación: UCF Solar PV + MGP")
    st.markdown("""
    <div class="ucf-box">
    Animación específica del caso Solar PV + Mercado de Generación de Potencia (MGP):
    muestra cómo la incertidumbre solar (Beta) y la incertidumbre del MGP (Normal) se combinan
    para determinar el costo esperado de incertidumbre del sistema híbrido.
    </div>
    """, unsafe_allow_html=True)
    try:
        st.image(os.path.join(os.path.dirname(__file__), "ucf_pv_mgp_small.gif"),
                 caption="UCF PV + MGP — Solar fotovoltaica + Mercado de Generación de Potencia: costo de incertidumbre combinado",
                 use_container_width=True)
    except Exception:
        st.info("📽️ Animación GIF disponible al ejecutar localmente: `ucf_pv_mgp_small.gif`")

# ═══════════════════════════════════════════════════════════
# TAB 4: UCF HIDRÓGENO VERDE
# ═══════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("# 💚 UCF para Hidrógeno Verde — Solar + Eólica + Electrolizador")
    st.markdown("""
    <div class="success-box">
    <b>Desarrollo reciente del grupo:</b> Extensión de las UCF al sistema de hidrógeno verde 
    alimentado por generación renovable híbrida. Validación analítica con error <b>1.788%</b> 
    vs Monte Carlo (10⁵ escenarios).<br>
    <b>Referencia:</b> Rincón, J.C. & Rivera, S. (2026). Boletín de Estadística e IO.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📐 Modelo Matemático")
    st.latex(r"""
    \mu_T = \mu_{PV} + \mu_w, \quad \Phi_T = \sqrt{\sigma^2_{PV} + \sigma^2_w}
    """)
    st.latex(r"""
    c_{H_2,u} = \eta_{H_2} \cdot P_{H_2}, \quad c_{H_2,o} = P_{kWh}
    """)
    st.markdown("**Subestimar:** H₂ no producido (pérdida de venta) | **Sobrestimar:** comprar energía de red")
    
    st.markdown("---")
    st.markdown("### 🎮 Simulador UCF Hidrógeno Verde")
    c1, c2, c3 = st.columns(3)
    with c1:
        h2_Wpvr = st.slider("W_PVr (MW)", 20, 150, 65, key="h2_pvr")
        h2_Wwr = st.slider("W_wr (MW)", 50, 300, 150, key="h2_wr")
        h2_Ps = st.slider("Ps electrolizador (MW)", 50, 300, 150, key="h2_ps")
    with c2:
        h2_eta = st.slider("η H₂ (kg/MWh)", 10, 30, 20, key="h2_eta")
        h2_PH2 = st.slider("Precio H₂ ($/kg)", 2, 10, 5, key="h2_ph2")
        h2_PkWh = st.slider("Precio red ($/MWh)", 10, 50, 20, key="h2_pkwh")
    with c3:
        h2_lam = st.slider("λ Lognormal", 4.0, 8.0, 6.0, 0.5, key="h2_lam")
        h2_beta = st.slider("β Lognormal", 0.1, 0.5, 0.25, 0.05, key="h2_beta")
        h2_sig_ray = st.slider("σ Rayleigh", 5.0, 25.0, 15.95, 0.5, key="h2_sigr")
    
    if st.button("▶️ Simular UCF Hidrógeno Verde", key="run_h2"):
        r_h2 = hydrogen_ucf_simulation(
            W_PVr=h2_Wpvr, W_wr=h2_Wwr, Ps=h2_Ps,
            eta_H2=h2_eta, P_H2=h2_PH2, P_kWh=h2_PkWh,
            lam=h2_lam, beta_ln=h2_beta, sigma_ray=h2_sig_ray,
            n_sim=100000, seed=42
        )
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("UCF Monte Carlo", f"${r_h2['expected_cost']:.2f}")
        c2.metric("UCF Analítico", f"${r_h2['analytical_cost']:.2f}")
        c3.metric("Error", f"{r_h2['error_pct']:.2f}%")
        c4.metric("μ_T ± Φ_T", f"{r_h2['mu_T']:.1f} ± {r_h2['sigma_T']:.1f} MW")
        
        fig_h2 = make_subplots(rows=2, cols=3,
            subplot_titles=["Potencia Solar PV", "Potencia Eólica", "Potencia Total",
                          "Costo Subestimar", "Costo Sobrestimar", "UCF Total H₂"])
        fig_h2.add_trace(go.Histogram(x=r_h2['W_PV'][:5000], nbinsx=50, marker_color="#FFB300"), row=1, col=1)
        fig_h2.add_trace(go.Histogram(x=r_h2['W_w'][:5000], nbinsx=50, marker_color="#1565C0"), row=1, col=2)
        fig_h2.add_trace(go.Histogram(x=r_h2['P_total'][:5000], nbinsx=50, marker_color="#2E7D32"), row=1, col=3)
        fig_h2.add_vline(x=h2_Ps, line_dash="dash", line_color="red", row=1, col=3)
        
        pos_h2 = np.maximum(r_h2['P_total'][:5000] - h2_Ps, 0) * r_h2['c_under']
        neg_h2 = np.maximum(h2_Ps - r_h2['P_total'][:5000], 0) * r_h2['c_over']
        fig_h2.add_trace(go.Histogram(x=pos_h2, nbinsx=50, marker_color="#43A047"), row=2, col=1)
        fig_h2.add_trace(go.Histogram(x=neg_h2, nbinsx=50, marker_color="#C62828"), row=2, col=2)
        fig_h2.add_trace(go.Histogram(x=r_h2['costs'][:5000], nbinsx=50, marker_color="#6A1B9A"), row=2, col=3)
        fig_h2.update_layout(height=500, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.8)", font=dict(color="#1a1a2e"), showlegend=False)
        st.plotly_chart(fig_h2, use_container_width=True, key="fig_ucf_h2")
        
        # Curva UCF vs Ps
        st.markdown("### 📈 UCF vs Potencia Programada — Analítico vs Monte Carlo")
        fig_h2c = go.Figure()
        fig_h2c.add_trace(go.Scatter(x=r_h2['Ps_range'], y=r_h2['ucf_mc_curve'],
            name="Monte Carlo", line=dict(color="#E65100", width=2)))
        fig_h2c.add_trace(go.Scatter(x=r_h2['Ps_range'], y=r_h2['ucf_an_curve'],
            name="Analítico", line=dict(color="#1a237e", width=2, dash="dash")))
        fig_h2c.add_vline(x=h2_Ps, line_dash="dot", line_color="red",
            annotation_text=f"Ps={h2_Ps}")
        fig_h2c.update_layout(height=400, title="UCF H₂ Verde — Validación Analítico vs MC",
            xaxis_title="Potencia Programada Ps (MW)", yaxis_title="UCF ($)",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.8)",
            font=dict(color="#1a1a2e"))
        st.plotly_chart(fig_h2c, use_container_width=True, key="fig_ucf_h2_curve")

# ═══════════════════════════════════════════════════════════
# TAB 5: CS-LCOE
# ═══════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("# 💰 CS-LCOE — Controllable System Levelized Cost of Electricity")
    st.markdown("""
    <div class="info-box">
    <b>Nuevo desarrollo del grupo:</b> Integración de las UCF dentro del marco LCOE tradicional.
    El CS-LCOE internaliza los costos de incertidumbre en la evaluación económica de plantas 
    renovables controlables (solar/eólica + BESS).<br>
    <b>Referencia:</b> Rivera, S. et al. (2026). CS-LCOE formulation and Monte Carlo validation.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📐 Formulación CS-LCOE")
    st.latex(r"""
    CS\text{-}LCOE = \frac{\sum_{t=1}^{T} \frac{CAPEX_t + OPEX_t + E[UCF_t(P_{eff,t})]}{(1+r)^t}}{\sum_{t=1}^{T} \frac{E[P_{s,t}]}{(1+r)^t}}
    """)
    st.markdown("""
    **Regiones piecewise:**
    - **Región 1** (Ps ≤ P_min): solo excedente → E[UCF] = Co·((P_max+P_min)/2 - Ps)
    - **Región 2** (P_min < Ps < P_max): dual → sub + sobre
    - **Región 3** (Ps ≥ P_max): solo déficit con BESS buffer
    """)
    
    st.markdown("---")
    st.markdown("### 🎮 Simulador CS-LCOE")
    st.markdown("""
    <div class="info-box">
    <b>Modelo BESS consistente:</b> El BESS absorbe hasta <b>P_eff MW</b> de desviación en 
    <b>ambas direcciones</b> (surplus y déficit). El costo residual de incertidumbre (después 
    del BESS) se calcula con la misma formulación cerrada en el método analítico y en Monte Carlo, 
    garantizando convergencia. El CS-LCOE debe ser cercano al LCOE tradicional cuando el BESS 
    es dimensionado correctamente.
    </div>
    """, unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        cs_capex_pv = st.slider("CAPEX Solar ($M)", 10, 200, 60, 5, key="cs_cpv")
        cs_capex_bess = st.slider("CAPEX BESS ($M)", 5, 100, 20, 5, key="cs_cbess")
        cs_opex = st.slider("OPEX anual ($M)", 0.2, 5.0, 1.2, 0.1, key="cs_opex")
    with c2:
        cs_T = st.slider("Vida útil (años)", 15, 30, 25, key="cs_T")
        cs_r = st.slider("Tasa descuento (%)", 3, 12, 6, key="cs_r") / 100
        cs_Ps = st.slider("Ps (MW)", 20, 200, 50, key="cs_Ps")
    with c3:
        cs_Pmin = st.slider("P_min (MW)", 10, 80, 30, key="cs_Pmin")
        cs_Pmax = st.slider("P_max (MW)", 40, 250, 70, key="cs_Pmax")
        cs_Cu = st.slider("Cu ($/MWh)", 10, 100, 30, key="cs_Cu")
        cs_Co = st.slider("Co ($/MWh)", 10, 100, 70, key="cs_Co")
        cs_Peff = st.slider("BESS buffer (MW)", 5, 40, 15, key="cs_Peff")
    
    if st.button("▶️ Calcular CS-LCOE", key="run_cs"):
        r_cs = cslcoe_simulation(
            capex_pv=cs_capex_pv*1e6, capex_bess=cs_capex_bess*1e6,
            opex_annual=cs_opex*1e6, T=cs_T, r=cs_r,
            P_scheduled=cs_Ps, P_min=cs_Pmin, P_max=cs_Pmax,
            Cu=cs_Cu, Co=cs_Co, P_eff=cs_Peff, n_sim=5000, seed=42
        )
        
        # Error de validación analítico vs MC
        err_pct = abs(r_cs['cslcoe_analytical'] - r_cs['cslcoe_mc_mean']) / r_cs['cslcoe_mc_mean'] * 100
        
        st.markdown("#### 📊 Resultados Comparativos")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("LCOE Tradicional", f"${r_cs['lcoe_tradicional']:.2f}/MWh", "Solar+BESS sin incertidumbre")
        c2.metric("CS-LCOE Analítico", f"${r_cs['cslcoe_analytical']:.2f}/MWh", "Fórmula cerrada + BESS")
        c3.metric("CS-LCOE Monte Carlo", f"${r_cs['cslcoe_mc_mean']:.2f}/MWh", f"Error: {err_pct:.2f}%")
        c4.metric("Premium UCF", f"${r_cs['ucf_premium']:.2f}/MWh", f"BESS reduce {r_cs['bess_reduction_pct']:.0f}%")
        
        # Validación
        if err_pct < 5:
            st.markdown(f"""
            <div class="success-box">
            <b>✅ Validación exitosa:</b> El error entre CS-LCOE analítico y Monte Carlo es 
            <b>{err_pct:.2f}%</b> (aceptable <5%). Ambos métodos usan el mismo modelo de BESS 
            (buffer P_eff={cs_Peff} MW en ambas direcciones).
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="ucf-box">
            <b>⚠️ Error de validación:</b> {err_pct:.2f}%. Considere aumentar los escenarios MC 
            o ajustar los parámetros para mejorar convergencia.
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <b>Interpretación de los 3 valores:</b><br>
        • <b>LCOE Tradicional</b> = (CAPEX Solar + CAPEX BESS + OPEX) / Energía — sin costo de incertidumbre<br>
        • <b>CS-LCOE</b> = LCOE Tradicional + Premium UCF — incluye costo residual de incertidumbre después del BESS<br>
        • <b>Premium UCF</b> = costo adicional por MWh por la incertidumbre renovable que el BESS no pudo absorber<br><br>
        <b>Si el BESS está bien dimensionado, CS-LCOE ≈ LCOE Tradicional</b> (Premium UCF pequeño).
        Esto demuestra que un sistema controlable (solar + BESS) puede competir económicamente 
        con generación convencional, internalizando el costo real de la incertidumbre.
        </div>
        """, unsafe_allow_html=True)
        
        # Comparación visual
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(
            x=["LCOE Tradicional", "CS-LCOE Analítico", "CS-LCOE MC (media)", "Premium UCF"],
            y=[r_cs['lcoe_tradicional'], r_cs['cslcoe_analytical'], r_cs['cslcoe_mc_mean'], r_cs['ucf_premium']],
            marker_color=["#1565C0", "#E65100", "#2E7D32", "#6A1B9A"],
            text=[f"${v:.2f}" for v in [r_cs['lcoe_tradicional'], r_cs['cslcoe_analytical'], r_cs['cslcoe_mc_mean'], r_cs['ucf_premium']]],
            textposition='outside'
        ))
        fig_comp.update_layout(height=400, title="Comparación: LCOE Tradicional vs CS-LCOE",
            yaxis_title="$/MWh",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.8)",
            font=dict(color="#1a1a2e"))
        st.plotly_chart(fig_comp, use_container_width=True, key="fig_cslcoe_comp")
        
        # Distribución MC
        st.markdown("#### 📊 Distribución CS-LCOE (Monte Carlo)")
        fig_cs = go.Figure()
        fig_cs.add_trace(go.Histogram(x=r_cs['mc_cslcoes'], nbinsx=50,
            name="CS-LCOE (MC)", marker_color="#1a237e"))
        fig_cs.add_vline(x=r_cs['cslcoe_analytical'], line_dash="dash", line_color="#E65100",
            annotation_text=f"Analítico: ${r_cs['cslcoe_analytical']:.2f}")
        fig_cs.add_vline(x=r_cs['lcoe_tradicional'], line_dash="dot", line_color="#1565C0",
            annotation_text=f"LCOE Trad: ${r_cs['lcoe_tradicional']:.2f}")
        fig_cs.update_layout(height=400, title="Distribución CS-LCOE con líneas de referencia",
            xaxis_title="CS-LCOE ($/MWh)", yaxis_title="Frecuencia",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.8)",
            font=dict(color="#1a1a2e"))
        st.plotly_chart(fig_cs, use_container_width=True, key="fig_cslcoe")
        
        # Detalle UCF
        st.markdown(f"""
        <div class="info-box">
        <b>Desglose del costo de incertidumbre:</b><br>
        • UCF sin BESS: <b>${r_cs['ucf_no_bess']:.2f}/h</b> (costo horario si no hubiera almacenamiento)<br>
        • UCF con BESS: <b>${r_cs['ucf_hourly']:.2f}/h</b> (después de absorber ±{cs_Peff} MW)<br>
        • Reducción BESS: <b>{r_cs['bess_reduction_pct']:.1f}%</b> del costo de incertidumbre eliminado<br>
        • Impacto anual: ${r_cs['ucf_hourly']*8760:,.0f}/año (con BESS) vs ${r_cs['ucf_no_bess']*8760:,.0f}/año (sin BESS)
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# TAB 6: SIMULACIONES (OPF, Microgrid, etc.) - de v1
# ═══════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown("# 🧪 Simulaciones de Sistemas de Potencia")
    st.markdown("""
    <div class="info-box">
    <i>Muestra representativa</i> de las capacidades del laboratorio. 
    Cada simulación implementa algoritmos del grupo. Ajuste parámetros en tiempo real.
    </div>
    """, unsafe_allow_html=True)
    
    sim_tabs = st.tabs(["⚡ OPF", "🔋 Microgrid", "🏭 Activos", "🐝 VNS-DEEPSO", "🔮 Pronóstico", "🎲 Monte Carlo"])
    
    with sim_tabs[0]:
        st.markdown("### ⚡ Flujo Óptimo de Potencia (OPF) con UCF")
        st.markdown("""
        <div class="ucf-box">
        <b>OPF con Uncertainty Cost Functions:</b> La formulación de OPF del grupo incorpora 
        directamente las UCF en la función objetivo, internalizando el costo de la incertidumbre 
        renovable en el despacho económico. La potencia programada Ps de cada generador es el 
        <b>resultado de esta optimización</b> que minimiza: costos de generación + costos de 
        incertidumbre (UCF).
        </div>
        """, unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            n_b = st.slider("Barras", 5, 118, 14, key="opf_b")
            carga = st.slider("Carga (MW)", 50, 1000, 250, 25, key="opf_c")
        with c2:
            ren = st.slider("Renovable (%)", 0, 80, 30, 5, key="opf_r")
            seed_o = st.slider("Semilla", 1, 100, 42, key="opf_s")
        with c3:
            st.markdown("**Sistemas IEEE:** 14, 30, 57, 96, 118 barras")
        
        if st.button("▶️ Ejecutar OPF", key="run_opf"):
            r = simular_opf_simplificado(n_b, carga, ren, seed_o)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Costo total", f"${r['costo_total_musd']:.0f}K")
            c2.metric("Pérdidas", f"{r['perdidas_mw']:.1f} MW")
            c3.metric("V min", f"{min(r['voltajes_pu']):.3f} pu")
            c4.metric("V max", f"{max(r['voltajes_pu']):.3f} pu")
            
            fig_o = make_subplots(rows=2, cols=1, shared_xaxes=True,
                subplot_titles=["Generación vs Demanda", "Voltajes"])
            fig_o.add_trace(go.Scatter(x=r['horas'], y=r['gen_termica_mw'], name="Térmica",
                fill="tozeroy", fillcolor="rgba(69,90,100,0.3)", line=dict(color="#455A64")), row=1, col=1)
            fig_o.add_trace(go.Scatter(x=r['horas'], y=r['gen_solar_mw'], name="Solar",
                fill="tonexty", fillcolor="rgba(255,179,0,0.3)", line=dict(color="#FFB300")), row=1, col=1)
            fig_o.add_trace(go.Scatter(x=r['horas'], y=r['demanda_mw'], name="Demanda",
                line=dict(color="#C62828", width=2, dash="dash")), row=1, col=1)
            fig_o.add_trace(go.Bar(x=list(range(1,n_b+1)), y=r['voltajes_pu'],
                marker_color=["#2E7D32" if .95<=v<=1.05 else "#C62828" for v in r['voltajes_pu']]), row=2, col=1)
            fig_o.update_layout(height=500, paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0.8)", font=dict(color="#1a1a2e"))
            st.plotly_chart(fig_o, use_container_width=True, key="fig_opf")
            
            # Animación de precios nodales (LMP)
            st.markdown("---")
            st.markdown("#### 🎬 Animación: Precios Nodales (Locational Marginal Pricing)")
            st.markdown("""
            Esta animación muestra la evolución de los precios marginales locales (LMP) en las 
            diferentes barras del sistema, resultado directo del OPF con restricciones de red. 
            Los LMP reflejan el costo de servir la próxima unidad de demanda en cada nodo, 
            incluyendo costos de congestión y pérdidas.
            """, unsafe_allow_html=True)
            st.image("10_nodal_lmp.gif", use_container_width=True)
            
            # Animación LMP académico con decisiones de refuerzo
            st.markdown("---")
            st.markdown("#### 🎬 Animación: LMP Académico y Señales de Expansión")
            st.markdown("""
            <div class="success-box">
            <b>Precios nodales como señal de inversión:</b> Esta animación muestra cómo los LMP 
            (Locational Marginal Prices) revelan congestión en la red y señalan oportunidades de 
            expansión. Cuando un nodo tiene precios consistentemente altos y una rama opera cerca 
            de su límite, es señal de que se necesita refuerzo, almacenamiento o generación local.
            </div>
            """, unsafe_allow_html=True)
            st.image(os.path.join(os.path.dirname(__file__), "04_lmp_academico_small.gif"),
                     caption="LMP Académico — Dispersión locacional, congestión y señales para decisiones de expansión de red",
                     use_container_width=True)
            st.markdown("""
            <div class="info-box">
            <b>Nota metodológica:</b> Esta es una simulación académica basada en red agregada del SIN. 
            No representa precios nodales oficiales de XM. Usos prácticos:
            <br>• Identificar nodos con precios persistentemente altos (señal de congestión)
            <br>• Detectar ramas con uso >100% (necesidad de refuerzo)
            <br>• Evaluar ubicación óptima de BESS para aliviar congestión
            <br>• Analizar viabilidad de generación distribuida en zonas caras
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="info-box">
            <b>Relación con UCF:</b> La potencia programada Ps de cada generador (resultado del OPF) 
            es el insumo principal para calcular las UCF. Si la generación real difiere de Ps debido 
            a la incertidumbre renovable, se incurre en costos de desbalance que las UCF cuantifican.
            </div>
            """, unsafe_allow_html=True)
    
    with sim_tabs[1]:
        st.markdown("### 🔋 Despacho Óptimo en Microgrid con UCF")
        c1, c2, c3 = st.columns(3)
        with c1:
            mg_s = st.slider("Solar (MW)", 1, 20, 5, key="mg_s")
            mg_w = st.slider("Eólica (MW)", 1, 15, 3, key="mg_w")
            mg_b = st.slider("BESS (kWh)", 500, 10000, 2000, 500, key="mg_b")
        with c2:
            mg_d = st.slider("Diesel (MW)", 1, 10, 4, key="mg_d")
            mg_dp = st.slider("Demanda pico (MW)", 3, 20, 8, key="mg_dp")
            mg_dr = st.slider("DR (%)", 0, 30, 15, 5, key="mg_dr")
        with c3:
            mg_seed = st.slider("Semilla", 1, 100, 42, key="mg_seed")
        
        r = simular_microgrid_despacho(mg_s, mg_w, mg_b, mg_d, mg_dp, mg_dr, mg_seed)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Costo/día", f"${r['costo_total_usd']:,.0f}")
        c2.metric("Solar", f"{r['energia_solar_mwh']:.1f} MWh")
        c3.metric("Diesel", f"{r['energia_diesel_mwh']:.1f} MWh")
        c4.metric("Curtailment", f"{np.sum(r['curtailment_mw']*0.25):.1f} MWh")
        
        fig_m = make_subplots(rows=2, cols=2, shared_xaxes=True,
            subplot_titles=["Generación y Demanda", "Despacho BESS (carga/descarga)",
                           "Estado de Carga (SOC)", "Balance Neto"])
        fig_m.add_trace(go.Scatter(x=r['horas'], y=r['solar_mw'], name="Solar",
            fill="tozeroy", fillcolor="rgba(255,179,0,0.3)", line=dict(color="#FFB300")), row=1, col=1)
        fig_m.add_trace(go.Scatter(x=r['horas'], y=r['eolica_mw'], name="Eólica",
            fill="tonexty", fillcolor="rgba(21,101,192,0.2)", line=dict(color="#1565C0")), row=1, col=1)
        fig_m.add_trace(go.Scatter(x=r['horas'], y=r['demanda_opt_mw'], name="Demanda",
            line=dict(color="#C62828", width=2, dash="dash")), row=1, col=1)
        fig_m.add_trace(go.Bar(x=r['horas'], y=r['bess_carga_mw'], name="BESS carga",
            marker_color="#43A047"), row=1, col=2)
        fig_m.add_trace(go.Bar(x=r['horas'], y=-r['bess_descarga_mw'], name="BESS descarga",
            marker_color="#1565C0"), row=1, col=2)
        fig_m.add_trace(go.Scatter(x=r['horas'], y=r['soc_pct'], name="SOC",
            line=dict(color="#FF8F00", width=2), fill="tozeroy",
            fillcolor="rgba(255,143,0,0.1)"), row=2, col=1)
        net = r['solar_mw'] + r['eolica_mw'] + r['bess_descarga_mw'] - r['bess_carga_mw'] - r['demanda_opt_mw']
        fig_m.add_trace(go.Scatter(x=r['horas'], y=net, name="Balance neto",
            fill="tozeroy", fillcolor="rgba(26,35,126,0.1)", line=dict(color="#1a237e")), row=2, col=2)
        fig_m.update_layout(height=600, barmode='relative', paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.8)", font=dict(color="#1a1a2e"))
        st.plotly_chart(fig_m, use_container_width=True, key="fig_mg")
    
    with sim_tabs[2]:
        st.markdown("### 🏭 Gestión de Activos (PywerAPM)")
        c1, c2 = st.columns(2)
        with c1:
            n_t = st.slider("Transformadores", 5, 50, 20, key="am_n")
            h_am = st.slider("Horizonte (años)", 5, 20, 10, key="am_h")
        with c2:
            s_am = st.slider("Semilla", 1, 100, 42, key="am_s")
        r = simular_gestion_activos(n_t, h_am, s_am)
        c1, c2, c3 = st.columns(3)
        c1.metric("Reemplazar", r['n_reemplazar'], "Riesgo alto")
        c2.metric("Mantener", r['n_mantener'], "Riesgo medio")
        c3.metric("Monitorear", r['n_monitorear'], "Riesgo bajo")
        fig_a = go.Figure(go.Bar(
            x=[t['id'] for t in r['trafos']],
            y=[t['riesgo'] for t in r['trafos']],
            marker_color=['#C62828' if t['decision']=='Reemplazar' else '#FF8F00' if t['decision']=='Mantener' else '#43A047' for t in r['trafos']]))
        fig_a.update_layout(height=350, title="Riesgo por Transformador",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.8)",
            font=dict(color="#1a1a2e"))
        st.plotly_chart(fig_a, use_container_width=True, key="fig_am")
        
        # Animaciones GIF de gestión de activos
        st.markdown("---")
        st.markdown("### 🎬 Animaciones de Gestión de Activos")
        
        anim_c1, anim_c2 = st.columns(2)
        with anim_c1:
            st.markdown("**Evolución del Riesgo** — Salud, probabilidad de falla y riesgo monetizado de 3 transformadores")
            try:
                st.image(os.path.join(os.path.dirname(__file__), "animacion_gestion_activos.gif"),
                         caption="Asset Risk: TR-01, TR-02, TR-03 — salud, P(falla), riesgo",
                         use_container_width=True)
            except Exception:
                st.info("📽️ GIF disponible localmente: `animacion_gestion_activos.gif`")
        with anim_c2:
            st.markdown("**Comparación de Estrategias** — No hacer nada vs Mantenimiento vs Reemplazo")
            try:
                st.image(os.path.join(os.path.dirname(__file__), "animacion_estrategias_activo.gif"),
                         caption="Estrategias: salud, riesgo y costo acumulado en 30 años",
                         use_container_width=True)
            except Exception:
                st.info("📽️ GIF disponible localmente: `animacion_estrategias_activo.gif`")
        
        st.markdown("""
        <div class="info-box">
        <b>Nota de atribución:</b> Estas animaciones son implementaciones educativas independientes,
        basadas en conceptos publicados sobre Uncertainty Cost Functions y gestión de activos eléctricos
        (riesgo de transformadores, mantenimiento y reemplazo, simulación probabilística).
        </div>
        """, unsafe_allow_html=True)
    
    with sim_tabs[3]:
        st.markdown("### 🐝 VNS-DEEPSO — Algoritmo Ganador IEEE")
        c1, c2 = st.columns(2)
        with c1:
            n_p = st.slider("Partículas", 10, 100, 30, key="de_p")
            n_i = st.slider("Iteraciones", 20, 300, 100, 10, key="de_i")
        with c2:
            s_de = st.slider("Semilla", 1, 100, 42, key="de_s")
        r = simular_deepso(n_p, n_i, seed=s_de)
        c1, c2, c3 = st.columns(3)
        c1.metric("Costo óptimo", f"${r['costo_optimo_usd_h']:.2f}/h")
        c2.metric("Gen total", f"{r['generacion_total_mw']:.1f} MW")
        c3.metric("Error balance", f"{r['error_balance_mw']:.2f} MW")
        fig_d = go.Figure()
        fig_d.add_trace(go.Scatter(x=r['iteraciones'], y=r['mejor_fitness'],
            name="Mejor", line=dict(color="#1a237e", width=2)))
        fig_d.add_trace(go.Scatter(x=r['iteraciones'], y=r['fitness_medio'],
            name="Medio", line=dict(color="#FF8F00", width=1, dash="dash")))
        fig_d.update_layout(height=350, title="Convergencia VNS-DEEPSO",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.8)",
            font=dict(color="#1a1a2e"))
        st.plotly_chart(fig_d, use_container_width=True, key="fig_de")
    
    with sim_tabs[4]:
        st.markdown("### 🔮 Pronóstico de Demanda")
        c1, c2 = st.columns(2)
        with c1:
            d_b = st.slider("Demanda base (MW)", 50, 500, 100, 25, key="fc_d")
            t_d = st.selectbox("Tipo día", ["laboral", "fin_de_semana"])
        with c2:
            s_fc = st.slider("Semilla", 1, 100, 42, key="fc_s")
        r = pronostico_demanda_24h(d_b, t_d, s_fc)
        c1, c2, c3 = st.columns(3)
        c1.metric("RMSE", f"{r['rmse_mw']:.2f} MW")
        c2.metric("MAPE", f"{r['mape_pct']:.1f}%")
        c3.metric("Banda 95%", f"±{d_b*0.05*1.96:.1f} MW")
        fig_f = make_subplots(rows=2, cols=1, shared_xaxes=True,
            subplot_titles=["Pronóstico 48h", "Error"])
        fig_f.add_trace(go.Scatter(x=r['horas'], y=r['banda_sup'], line=dict(width=0)), row=1, col=1)
        fig_f.add_trace(go.Scatter(x=r['horas'], y=r['banda_inf'], fill="tonexty",
            fillcolor="rgba(26,35,126,0.1)", line=dict(width=0)), row=1, col=1)
        fig_f.add_trace(go.Scatter(x=r['horas'], y=r['pronostico'], name="Pronóstico",
            line=dict(color="#1a237e", width=2)), row=1, col=1)
        fig_f.add_trace(go.Scatter(x=r['horas'], y=r['real'], name="Real",
            line=dict(color="#FF8F00", width=1, dash="dash")), row=1, col=1)
        fig_f.add_trace(go.Scatter(x=r['horas'], y=r['error'], name="Error",
            fill="tozeroy", fillcolor="rgba(198,40,40,0.1)", line=dict(color="#C62828")), row=2, col=1)
        fig_f.update_layout(height=500, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.8)", font=dict(color="#1a1a2e"))
        st.plotly_chart(fig_f, use_container_width=True, key="fig_fc")
    
    with sim_tabs[5]:
        st.markdown("### 🎲 Monte Carlo — ENS y Curtailment")
        c1, c2 = st.columns(2)
        with c1:
            n_s = st.slider("Simulaciones", 100, 5000, 1000, 100, key="mc_n")
            d_mc = st.slider("Demanda (MW)", 100, 1000, 300, 50, key="mc_d")
        with c2:
            r_mc = st.slider("Renovable (%)", 10, 80, 40, 5, key="mc_r")
            s_mc = st.slider("Semilla", 1, 100, 42, key="mc_s")
        r = simular_monte_carlo_ens(n_s, d_mc, r_mc, s_mc)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Costo medio", f"${r['costo_medio']:.0f}/h")
        c2.metric("CVaR 95%", f"${r['costo_cvar_95']:.0f}/h")
        c3.metric("Curtailment", f"{r['curt_medio_mw']:.1f} MW")
        c4.metric("ENS", f"{r['ens_medio_mw']:.2f} MW")
        fig_mc = make_subplots(rows=1, cols=2,
            subplot_titles=["Costos", "Curtailment vs ENS"])
        fig_mc.add_trace(go.Histogram(x=r['costos'], nbinsx=50, marker_color="#1a237e"), row=1, col=1)
        fig_mc.add_trace(go.Scatter(x=r['curtailment'], y=r['ens'], mode="markers",
            marker=dict(size=3, color="#FF8F00", opacity=0.4)), row=1, col=2)
        fig_mc.update_layout(height=400, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.8)", font=dict(color="#1a1a2e"))
        st.plotly_chart(fig_mc, use_container_width=True, key="fig_mc")

# ═══════════════════════════════════════════════════════════
# TABS 7-12: Publicaciones, Apps, Tesis, Colab, Logros, Únete - de v1
# ═══════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown("# 📊 Publicaciones Científicas")
    st.markdown(f"""
    <div class="info-box">
    <b>Esta es una pequeña muestra</b> de las +{PERFIL['publicaciones']} publicaciones del grupo 
    ({PERFIL['citas']:,}+ citas en Google Scholar, h-index {PERFIL['h_index']}).<br>
    Rankings Q basados en Scimago Journal Rank sobre los <b>168 artículos más citados</b> (últimos 10 años).
    El grupo tiene <b>muchas más publicaciones</b> no listadas aquí.
    <br>📚 Perfil completo: <a href="{PERFIL['scholar']}" target="_blank">Google Scholar</a> · 
    <a href="{PERFIL['hermes']}" target="_blank">Hermes UNAL</a>
    </div>
    """, unsafe_allow_html=True)
    
    # Métricas reales (Q_RESUMEN tiene los totales reales)
    st.markdown("### 📈 Distribución Real por Cuartil (168 artículos más citados)")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    total_q = sum(Q_RESUMEN.values())
    c1.metric("Q1", Q_RESUMEN['Q1'], f"{Q_RESUMEN['Q1']/total_q*100:.0f}%")
    c2.metric("Q2", Q_RESUMEN['Q2'], f"{Q_RESUMEN['Q2']/total_q*100:.0f}%")
    c3.metric("Q3", Q_RESUMEN['Q3'], f"{Q_RESUMEN['Q3']/total_q*100:.0f}%")
    c4.metric("Q4", Q_RESUMEN['Q4'], f"{Q_RESUMEN['Q4']/total_q*100:.0f}%")
    c5.metric("Sin Cuartil", Q_RESUMEN['NA'], f"{Q_RESUMEN['NA']/total_q*100:.0f}%")
    c6.metric("Total", total_q)
    
    # Desglose de publicaciones sin cuartil Scimago
    st.markdown("""
    <div class="info-box">
    <b>📋 Desglose de publicaciones sin cuartil Scimago (98):</b><br>
    Las publicaciones clasificadas como <b>"Sin Cuartil"</b> no carecen de valor académico — corresponden a 
    publicaciones en <b>revistas indexadas</b> (indexadas en Publindex, Latindex, WOS-ESCI, etc. pero sin 
    cuartil Scimago), <b>congresos internacionales</b> (IEEE PES, ISGT, CCA), <b>congresos nacionales</b> 
    y <b>libros/capítulos</b>.
    </div>
    """, unsafe_allow_html=True)
    
    na_c1, na_c2, na_c3, na_c4 = st.columns(4)
    na_c1.metric("Revistas Indexadas", NA_DESGLOSE["Revistas Indexadas"], f"{NA_DESGLOSE['Revistas Indexadas']/Q_RESUMEN['NA']*100:.0f}% de sin cuartil")
    na_c2.metric("Congresos Internacionales", NA_DESGLOSE["Congresos Internacionales"], f"{NA_DESGLOSE['Congresos Internacionales']/Q_RESUMEN['NA']*100:.0f}% de sin cuartil")
    na_c3.metric("Congresos Nacionales", NA_DESGLOSE["Congresos Nacionales"], f"{NA_DESGLOSE['Congresos Nacionales']/Q_RESUMEN['NA']*100:.0f}% de sin cuartil")
    na_c4.metric("Libros y Capítulos", NA_DESGLOSE["Libros y Capítulos"], f"{NA_DESGLOSE['Libros y Capítulos']/Q_RESUMEN['NA']*100:.0f}% de sin cuartil")
    
    # Gráficos interactivos
    pub_viz_tabs = st.tabs(["🥧 Cuartiles", "📅 Timeline", "📊 Citas Top", "🔍 Explorador"])
    
    with pub_viz_tabs[0]:
        st.markdown("### Distribución Completa por Tipo de Publicación (168 artículos)")
        
        pie_c1, pie_c2 = st.columns(2)
        
        with pie_c1:
            st.markdown("#### Por Cuartil Scimago")
            fig_q = go.Figure(data=[go.Pie(
                labels=[f"Q1 ({Q_RESUMEN['Q1']})", f"Q2 ({Q_RESUMEN['Q2']})", 
                        f"Q3 ({Q_RESUMEN['Q3']})", f"Q4 ({Q_RESUMEN['Q4']})", f"Sin Cuartil ({Q_RESUMEN['NA']})"],
                values=[Q_RESUMEN['Q1'], Q_RESUMEN['Q2'], Q_RESUMEN['Q3'], Q_RESUMEN['Q4'], Q_RESUMEN['NA']],
                hole=0.45,
                marker=dict(colors=["#1B5E20","#1565C0","#FF8F00","#C62828","#9E9E9E"]),
                textinfo='label+percent',
                textfont_size=12,
                hovertemplate="<b>%{label}</b><br>%{value} artículos<br>%{percent}<extra></extra>"
            )])
            fig_q.update_layout(height=420, paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#1a1a2e"),
                               title="Cuartiles Scimago (168 artículos)")
            st.plotly_chart(fig_q, use_container_width=True)
        
        with pie_c2:
            st.markdown("#### Desglose de 'Sin Cuartil' (98)")
            fig_na = go.Figure(data=[go.Pie(
                labels=[f"Rev. Indexadas ({NA_DESGLOSE['Revistas Indexadas']})", 
                        f"Congr. Internac. ({NA_DESGLOSE['Congresos Internacionales']})",
                        f"Congr. Nacionales ({NA_DESGLOSE['Congresos Nacionales']})", 
                        f"Libros/Cap. ({NA_DESGLOSE['Libros y Capítulos']})"],
                values=[NA_DESGLOSE['Revistas Indexadas'], NA_DESGLOSE['Congresos Internacionales'],
                        NA_DESGLOSE['Congresos Nacionales'], NA_DESGLOSE['Libros y Capítulos']],
                hole=0.45,
                marker=dict(colors=["#0D47A1","#1565C0","#42A5F5","#90CAF9"]),
                textinfo='label+percent',
                textfont_size=11,
                hovertemplate="<b>%{label}</b><br>%{value} artículos<br>%{percent}<extra></extra>"
            )])
            fig_na.update_layout(height=420, paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#1a1a2e"),
                               title="Detalle Sin Cuartil (98 artículos)")
            st.plotly_chart(fig_na, use_container_width=True)
        
        st.markdown("""
        <div class="success-box">
        <b>52% en Q1+Q2</b> — Más de la mitad de los artículos más citados están en revistas del 
        primer y segundo cuartil. El 31% Q1 es particularmente notable para un grupo en Latinoamérica.
        <br><br>
        <b>Las 98 publicaciones "Sin Cuartil"</b> se distribuyen entre <b>revistas indexadas</b> 
        (36%), <b>congresos internacionales</b> (36%), <b>congresos nacionales</b> (20%) y <b>libros/capítulos</b> (8%).
        Los congresos IEEE internacionales son particularmente prestigiosos en ingeniería eléctrica.
        </div>
        """, unsafe_allow_html=True)

    
    with pub_viz_tabs[1]:
        st.markdown("### 📅 Publicaciones por Año")
        st.markdown(f"""
        <div class="ucf-box">
        <b>Nota:</b> La gráfica inferior muestra <b>{len(PUBLICACIONES)} publicaciones de muestra</b> 
        de las 168+ totales. Los años 2019-2026 concentran la mayor producción del grupo, 
        coincidiendo con la expansión de las líneas de investigación en renovables y UCF.
        </div>
        """, unsafe_allow_html=True)
        
        # Datos reales de la muestra + proyección de totales
        anios = sorted(set(p['anio'] for p in PUBLICACIONES))
        counts_sample = [sum(1 for p in PUBLICACIONES if p['anio']==a) for a in anios]
        
        # Estimación de totales reales basada en la proporción de la muestra
        factor = total_q / max(len(PUBLICACIONES), 1)
        counts_estimated = [round(c * factor) for c in counts_sample]
        
        fig_t = go.Figure()
        fig_t.add_trace(go.Bar(
            x=[str(a) for a in anios], y=counts_sample,
            name="Muestra en app", marker_color="#1a237e", opacity=0.8,
            text=counts_sample, textposition='outside',
            hovertemplate="<b>%{x}</b><br>Muestra: %{y} pubs<extra></extra>"
        ))
        fig_t.add_trace(go.Bar(
            x=[str(a) for a in anios], y=counts_estimated,
            name="Estimación real", marker_color="#7C4DFF", opacity=0.4,
            text=counts_estimated, textposition='outside',
            hovertemplate="<b>%{x}</b><br>Estimado: %{y} pubs<extra></extra>"
        ))
        fig_t.update_layout(
            height=420, barmode='group',
            title=f"Publicaciones por Año — {len(PUBLICACIONES)} de muestra vs estimación real",
            xaxis_title="Año", yaxis_title="N° Publicaciones",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.8)",
            font=dict(color="#1a1a2e"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        st.plotly_chart(fig_t, use_container_width=True)
        
        # Stats por período
        c1, c2, c3 = st.columns(3)
        pre2020 = sum(1 for p in PUBLICACIONES if p['anio'] < 2020)
        post2020 = sum(1 for p in PUBLICACIONES if p['anio'] >= 2020)
        c1.metric("Antes de 2020", pre2020, "publicaciones muestra")
        c2.metric("2020 en adelante", post2020, "publicaciones muestra")
        c3.metric("Ratio", f"{post2020/max(pre2020,1):.1f}×", "más producción reciente")
    
    with pub_viz_tabs[2]:
        st.markdown("### 🏆 Top Citas — Publicaciones Más Impactantes")
        
        # Ordenar por citas
        sorted_pubs = sorted(PUBLICACIONES, key=lambda x: x['citas'], reverse=True)
        top_n = st.slider("Mostrar Top N", 5, len(sorted_pubs), 15, key="top_n_pubs")
        
        top = sorted_pubs[:top_n]
        fig_citas = go.Figure(go.Bar(
            x=[p['citas'] for p in top],
            y=[f"{p['anio']} — {p['titulo'][:50]}..." for p in top],
            orientation='h',
            marker=dict(
                color=[p['citas'] for p in top],
                colorscale=[[0, "#E8EAF6"], [0.5, "#5C6BC0"], [1, "#1a237e"]],
                showscale=True, colorbar=dict(title="Citas")
            ),
            text=[p['citas'] for p in top], textposition='outside',
            hovertemplate="<b>%{y}</b><br>Citas: %{x}<extra></extra>"
        ))
        fig_citas.update_layout(
            height=max(400, top_n * 35),
            title=f"Top {top_n} Publicaciones por Citas (de {len(PUBLICACIONES)} en muestra)",
            xaxis_title="N° Citas",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.8)",
            font=dict(color="#1a1a2e"), yaxis=dict(autorange='reversed')
        )
        st.plotly_chart(fig_citas, use_container_width=True)
        
        # Stats de citas
        all_citas = [p['citas'] for p in PUBLICACIONES]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total citas (muestra)", f"{sum(all_citas):,}")
        c2.metric("Media citas/pub", f"{np.mean(all_citas):.1f}")
        c3.metric("Mediana citas/pub", f"{np.median(all_citas):.1f}")
        c4.metric("Max citas", f"{max(all_citas):,}")
    
    with pub_viz_tabs[3]:
        st.markdown("### 🔍 Explorador Interactivo de Publicaciones")
        
        # Filtros avanzados
        c1, c2, c3 = st.columns(3)
        with c1:
            q_filter = st.multiselect("Cuartil", ["Q1","Q2","Q3","Q4","NA"], default=["Q1","Q2","Q3","Q4","NA"], key="q_explore")
        with c2:
            anios_disp = sorted(set(p['anio'] for p in PUBLICACIONES))
            year_range = st.slider("Rango de años", min(anios_disp), max(anios_disp), 
                                   (min(anios_disp), max(anios_disp)), key="year_range")
        with c3:
            # Recopilar todas las categorías
            all_cats = sorted(set(cat for p in PUBLICACIONES for cat in p['cat']))
            cat_filter = st.multiselect("Categoría", all_cats, default=[], key="cat_explore")
        
        # Búsqueda de texto
        search_term = st.text_input("🔍 Buscar en título/autores/revista", "", key="pub_search")
        
        # Orden
        sort_by = st.selectbox("Ordenar por", ["Citas (mayor a menor)", "Citas (menor a mayor)", 
                                                 "Año (reciente)", "Año (antiguo)", "Título (A-Z)"],
                               key="pub_sort")
        
        # Aplicar filtros
        filtered = [p for p in PUBLICACIONES 
                    if p['q'] in q_filter
                    and year_range[0] <= p['anio'] <= year_range[1]
                    and (not cat_filter or any(c in p['cat'] for c in cat_filter))]
        
        if search_term:
            term = search_term.lower()
            filtered = [p for p in filtered if term in p['titulo'].lower() 
                       or term in p['autores'].lower() 
                       or term in p['revista'].lower()]
        
        # Ordenar
        sort_map = {
            "Citas (mayor a menor)": lambda x: -x['citas'],
            "Citas (menor a mayor)": lambda x: x['citas'],
            "Año (reciente)": lambda x: -x['anio'],
            "Año (antiguo)": lambda x: x['anio'],
            "Título (A-Z)": lambda x: x['titulo'].lower(),
        }
        filtered.sort(key=sort_map[sort_by])
        
        # Resumen de filtros
        total_citas_filtered = sum(p['citas'] for p in filtered)
        st.markdown(f"""
        <div class="info-box">
        <b>{len(filtered)}</b> publicaciones encontradas · 
        <b>{total_citas_filtered:,}</b> citas totales · 
        Rango: {year_range[0]}-{year_range[1]}
        </div>
        """, unsafe_allow_html=True)
        
        # Mostrar resultados con paginación
        per_page = st.selectbox("Por página", [10, 20, 50], index=1, key="pub_per_page")
        n_pages = max(1, (len(filtered) + per_page - 1) // per_page)
        if n_pages > 1:
            page = st.number_input("Página", 1, n_pages, 1, key="pub_page")
        else:
            page = 1
        
        start = (page - 1) * per_page
        end = min(start + per_page, len(filtered))
        
        for p in filtered[start:end]:
            q_color = {"Q1": "#1B5E20", "Q2": "#1565C0", "Q3": "#FF8F00", "Q4": "#C62828", "NA": "#9E9E9E"}.get(p['q'], "#9E9E9E")
            with st.expander(f"🔹 [{p['q']}] [{p['anio']}] {p['titulo'][:85]}... — {p['citas']} citas"):
                st.markdown(f"""
                <div style="display:inline-block;padding:2px 10px;background:{q_color};color:white;border-radius:4px;font-weight:bold;margin-right:8px;">{p['q']}</div>
                <span style="color:#555;">{p['anio']} · {p['tipo']}</span>
                """, unsafe_allow_html=True)
                st.markdown(f"**Autores:** {p['autores']}")
                st.markdown(f"**Revista/Fuente:** *{p['revista']}*")
                st.markdown(f"**Citas:** {p['citas']:,}")
                st.markdown(f"**Categorías:** {', '.join(p['cat'])}")

with tabs[7]:
    st.markdown("# 💻 Apps Interactivas en GitHub")
    st.markdown(f"""
    <div class="info-box">
    <b>{len(APPS_GITHUB)} apps interactivas</b> desplegadas con Streamlit. Código abierto en 
    <a href="{PERFIL['github']}" target="_blank">github.com/srriverar</a>
    </div>
    """, unsafe_allow_html=True)
    for app in APPS_GITHUB:
        with st.expander(f"{app['icono']} {app['nombre']} — {app['tema']}"):
            c1, c2 = st.columns([2, 1])
            with c1:
                st.markdown(f"**{app['desc']}**\n\nLenguaje: {app['lang']}\n\n🔗 [{app['url']}]({app['url']})")
                for f in app['features']:
                    st.markdown(f"- {f}")
            with c2:
                st.markdown(f'<div style="text-align:center;padding:25px;background:#E8EAF6;border-radius:12px;"><div style="font-size:3rem;">{app["icono"]}</div><b>{app["nombre"]}</b></div>', unsafe_allow_html=True)
    
    # Animaciones de decisión para el sector eléctrico
    st.markdown("---")
    st.markdown("### 🎬 Animaciones de Decisión para el Sector Eléctrico")
    st.markdown("""
    <div class="success-box">
    <b>De dashboards a decisiones:</b> Estas animaciones muestran cómo los dashboards interactivos 
    y las herramientas de análisis apoyan la toma de decisiones en el sector eléctrico colombiano. 
    Desde el monitoreo del SIN hasta el análisis de fenómenos climáticos como ENSO, cada visualización 
    está diseñada para revelar patrones, alertar riesgos y guiar inversiones.
    </div>
    """, unsafe_allow_html=True)
    
    # Animación SIN Cockpit
    st.markdown("#### 📊 SIN Cockpit — Monitoreo Integral del Sistema")
    st.image(os.path.join(os.path.dirname(__file__), "02_sin_cockpit_small.gif"),
             caption="SIN Cockpit — Dispatch, precios, margen operativo, embalses y emisiones en tiempo real",
             use_container_width=True)
    st.markdown("""
    <div class="info-box">
    <b>Cockpit metodológicamente seguro:</b> Esta animación integra múltiples variables del SIN 
    (despacho por tecnología, precios marginales, margen operativo, nivel de embalses, emisiones) 
    en una sola vista. Las alertas automáticas señalan cuándo el sistema está cerca de límites 
    críticos (margen <150 MW, embalses <60%, precios >500 COP/kWh).
    <br><br>
    <b>Decisiones que apoya:</b>
    <br>• Operadores: anticipar necesidad de respaldo térmico o importación
    <br>• Planificadores: identificar períodos de estrés del sistema
    <br>• Inversionistas: evaluar oportunidades de BESS o generación flexible
    <br>• Reguladores: monitorear seguridad del suministro
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Animación ENSO
    st.markdown("#### 🌊 ENSO — Impacto Climático en el Sistema Eléctrico")
    st.image(os.path.join(os.path.dirname(__file__), "03_enso_frecuencias_small.gif"),
             caption="ENSO Analysis — Frecuencias mensuales vs variables operativas diarias/horarias",
             use_container_width=True)
    st.markdown("""
    <div class="info-box">
    <b>No confundir correlación con causalidad:</b> Esta animación muestra la relación entre el 
    índice ONI (El Niño/La Niña) y variables operativas del SIN (embalses, participación térmica, 
    precios). El mensaje clave: ENSO opera en escala mensual/estacional, mientras que las decisiones 
    operativas son diarias/horarias.
    <br><br>
    <b>Decisiones que apoya:</b>
    <br>• Gestión hidrológica: anticipar reducción de aportes durante El Niño
    <br>• Contratación: ajustar portafolios de generación antes de eventos extremos
    <br>• Operación: preparar respaldo térmico con anticipación
    <br>• Planificación: dimensionar almacenamiento para resiliencia climática
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    <div class="info-box">
    <b>Nota sobre el código:</b> Las animaciones presentadas son reimplementaciones visuales 
    independientes basadas en los conceptos de la app LICA-SE. No son el código original del 
    laboratorio, sino herramientas educativas que demuestran cómo la visualización interactiva 
    puede transformar datos complejos en decisiones accionables para el sector eléctrico.
    </div>
    """, unsafe_allow_html=True)

with tabs[8]:
    st.markdown("# 🌍 Colaboraciones Internacionales")
    c1, c2 = st.columns(2)
    with c1:
        df_c = pd.DataFrame(COOPERACION).sort_values('pubs', ascending=True)
        fig_c = go.Figure(go.Bar(x=df_c['pubs'], y=df_c['pais'], orientation='h', marker_color="#1a237e"))
        fig_c.update_layout(height=500, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.8)", font=dict(color="#1a1a2e"))
        st.plotly_chart(fig_c, use_container_width=True)
    with c2:
        st.dataframe(pd.DataFrame(COOPERACION).sort_values('pubs', ascending=False), use_container_width=True, hide_index=True)

with tabs[9]:
    st.markdown("# 🏆 Logros y Competencias Ganadas")
    st.markdown("""
    <div class="competition-box">
    <h3>🥇 IEEE PES Competitions — 5× Ganador</h3>
    <b>2017:</b> Competition on Modern Heuristic Optimizers for Smart Grid Operation<br>
    <b>2018:</b> Grid Optimization Competition (WCCI 2018)<br>
    <b>2020:</b> Expansion Planning considering Uncertainty<br>
    <b>2025:</b> AI-ccelerating Competition — AI aplicada al sector eléctrico<br>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🎓 Liderazgo Académico")
    st.markdown("""
    - **Líder Eje 3** — RIPCEL (Red Iberoamericana de Comunidades Energéticas, 16 países)
    - **Certificación MM SDS** — MIT (2025)
    - **Director DIEE** — UNAL (2023-2024)
    - **Coordinador Maestría y Doctorado** — Ing. Eléctrica UNAL (2017-2022)
    - **Profesor Invitado** — KIT y TU Dortmund (Alemania), Ruhr University Bochum (VIP Program), University of Florida (Fulbright Scholar)
    """)
    
    st.markdown("### 📱 Actividad Reciente (LinkedIn)")
    st.markdown("""
    - [Domando la incertidumbre renovable con UCF](https://www.linkedin.com/posts/sergio-rivera-7758ba68_domando-la-incertidumbre-renovables-con-activity-7387232905871138816-Fgiq)
    - [Almacenamiento de energía + BESS + educación continua](https://www.linkedin.com/posts/sergio-rivera-7758ba68_almacenamientodeenergaeda-educaciaejncontinua-activity-7415413506818252800-f7KU)
    - [2026 Competition on Evolutionary Computation](https://www.linkedin.com/posts/sergio-rivera-7758ba68_2026-competition-on-evolutionary-computation-activity-7410325558041026560-QtLe)
    - [Investigación UNAL](https://www.linkedin.com/posts/sergio-rivera-7758ba68_investigaciaejn-universidadnacionaldecolombia-activity-7395184230298701825-RlE1)
    - [MIT · Ingeniería · Energía](https://www.linkedin.com/posts/sergio-rivera-7758ba68_mit-ingenieraeda-energaeda-activity-7330944145684889600-6921)
    - [Diplomado en Tópicos de Sistemas de Potencia](https://www.linkedin.com/posts/sergio-rivera-7758ba68_diplomado-en-t%C3%B3picos-de-sistemas-de-potencia-activity-7341431199694823424-8UqZ)
    - [Energía Renovable + Monte Carlo + MGP](https://www.linkedin.com/posts/sergio-rivera-7758ba68_energaedarenovable-montecarlo-mgp-activity-7326665313758994432-5aSJ)
    - [Doctorado UNAL + Becas Minciencias](https://www.linkedin.com/posts/sergio-rivera-7758ba68_doctoradounal-ingenieraedaelaezctrica-becasminciencias-activity-7378851346008997888-h6ly)
    - [Video YouTube: UCF Demo](https://youtube.com/watch?v=Z--Nn_9cFHI)
    """)
    
    st.markdown("### 📋 Propuesta DNP — Mecanismo Transaccional con UCF")
    st.markdown("""
    <div class="info-box">
    <b>Mercado Local de Energía con Gestión de Incertidumbre (MLE-I)</b><br>
    Propuesta para DNP-CM-005-2026: Programa de energización rural con mecanismo transaccional 
    replicable basado en UCF. Liderado por Prof. Rivera como Líder Eje 3 de RIPCEL.
    </div>
    """, unsafe_allow_html=True)
    st.latex(r"""
    \max B_{total} = \sum_{j} U_j(d_j) - \sum_{i} C_i(g_i) - \sum_{k} UCF_k(g_k, \omega_k)
    """)

with tabs[11]:
    st.markdown("# 📝 Únete al Grupo LICA-SE")
    
    st.markdown("""
    <div class="info-box">
    <b>¿Tienes una idea de investigación o proyecto?</b><br>
    El grupo LICA-SE está abierto a colaboraciones con estudiantes de pregrado, maestría y doctorado, 
    así como con empresas e instituciones interesadas en aplicar inteligencia computacional al sector 
    eléctrico. Completa el formulario y te contactaremos para explorar sinergias.
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("### 🎓 Para Estudiantes")
        st.markdown("""
        - **Pregrado:** Proyectos de grado en optimización, renovables, sistemas de potencia
        - **Maestría:** Investigación en UCF, BESS, microredes, gestión de activos
        - **Doctorado:** Tesis innovadoras con publicación en revistas Q1/Q2
        - **Beneficios:**
          - Red internacional (15+ países)
          - Acceso a datos reales de sistemas eléctricos
          - Publicaciones en journals de alto impacto
          - Participación en competencias IEEE
          - Oportunidades de pasantías internacionales
        """)
    
    with c2:
        st.markdown("### 🏢 Para Empresas e Instituciones")
        st.markdown("""
        - **Consultoría especializada:**
          - OPF y despacho económico con UCF
          - Análisis de confiabilidad y riesgo
          - Optimización de portafolios de generación
        - **Desarrollo de software a medida:**
          - Dashboards interactivos
          - Herramientas de simulación
          - Sistemas de gestión de activos
        - **Capacitación:**
          - Cursos in-company
          - Talleres técnicos
          - Seminarios especializados
        """)
    
    st.markdown("---")
    st.markdown("### 📋 Formulario de Contacto")
    
    with st.form("contact_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        
        with col1:
            nombre = st.text_input("Nombre completo*", placeholder="Ej: María García")
            email = st.text_input("Email*", placeholder="Ej: maria@universidad.edu.co")
            institucion = st.text_input("Institución/Empresa", placeholder="Ej: Universidad Nacional")
        
        with col2:
            tipo = st.selectbox("Tipo de colaboración*", 
                ["Estudiante de Pregrado", "Estudiante de Maestría", "Estudiante de Doctorado",
                 "Empresa/Consultoría", "Investigador/Colaborador", "Otro"])
            area_interes = st.multiselect("Áreas de interés", 
                ["UCF - Uncertainty Cost Functions", "BESS - Almacenamiento de Energía",
                 "Microredes", "OPF/Despacho Económico", "Gestión de Activos",
                 "Vehículos Eléctricos", "Hidrógeno Verde", "Pronóstico de Demanda/Generación",
                 "Machine Learning aplicado", "Otro"])
            nivel_experiencia = st.select_slider("Nivel de experiencia en Python",
                options=["Ninguno", "Básico", "Intermedio", "Avanzado", "Experto"])
        
        st.markdown("#### 💡 Describe tu idea o proyecto")
        idea = st.text_area("¿Qué te gustaría desarrollar o investigar?*", 
            placeholder="Describe brevemente tu idea, problema a resolver, o área de interés...",
            height=120)
        
        requerimientos = st.text_area("¿Qué necesitas del grupo LICA-SE?",
            placeholder="Ej: Asesoría técnica, acceso a datos, coautoría, desarrollo de software, etc.",
            height=100)
        
        timeline = st.selectbox("¿Cuándo te gustaría iniciar?",
            ["Inmediatamente", "Próximo semestre", "Próximo año", "Solo explorando posibilidades"])
        
        col_submit, col_info = st.columns([1, 3])
        with col_submit:
            submitted = st.form_submit_button("📤 Enviar Solicitud", use_container_width=True)
        with col_info:
            st.markdown("*Campos obligatorios marcados con asterisco")
        
        if submitted:
            if not nombre or not email or not idea:
                st.error("❌ Por favor completa los campos obligatorios: Nombre, Email e Idea/Proyecto")
            else:
                # Guardar la solicitud en un archivo JSON
                import json
                from datetime import datetime
                
                solicitud = {
                    "timestamp": datetime.now().isoformat(),
                    "nombre": nombre,
                    "email": email,
                    "institucion": institucion,
                    "tipo": tipo,
                    "area_interes": area_interes,
                    "nivel_experiencia": nivel_experiencia,
                    "idea": idea,
                    "requerimientos": requerimientos,
                    "timeline": timeline
                }
                
                # Guardar en archivo
                try:
                    with open("solicitudes_colaboracion.json", "a", encoding="utf-8") as f:
                        f.write(json.dumps(solicitud, ensure_ascii=False) + "\n")
                    
                    st.success("""
                    ✅ **¡Solicitud enviada exitosamente!**
                    
                    Gracias por tu interés en el grupo LICA-SE. Hemos recibido tu solicitud y 
                    el Prof. Rivera o un miembro del equipo te contactará en los próximos 
                    **3-5 días hábiles** al email proporcionado.
                    
                    **Referencia:** Tu solicitud ha sido registrada con timestamp {}.
                    """.format(solicitud["timestamp"]))
                    
                    # Mostrar resumen
                    with st.expander("📄 Resumen de tu solicitud"):
                        st.json(solicitud)
                
                except Exception as e:
                    st.error(f"❌ Error al guardar la solicitud: {str(e)}")
                    st.info("Por favor contacta directamente al Prof. Rivera: srriverar@unal.edu.co")
    
    st.markdown("---")
    st.markdown("### 📧 Contacto Directo")
    st.markdown(f"""
    Si prefieres contactarnos directamente:
    - **Email:** [{PERFIL['email']}](mailto:{PERFIL['email']})
    - **GitHub:** [{PERFIL['github']}]({PERFIL['github']})
    - **Google Scholar:** [{PERFIL['scholar']}]({PERFIL['scholar']})
    - **LinkedIn:** [linkedin.com/in/sergio-rivera-7758ba68](https://www.linkedin.com/in/sergio-rivera-7758ba68)
    """)

# FOOTER
st.markdown("---")
st.markdown("""
<div style="text-align:center; padding:20px; color:#888; font-size:0.8rem;">
<b>LICA-SE v2.0</b> · Laboratorio de Inteligencia Computacional Aplicada al Sector Eléctrico<br>
Prof. Sergio Raúl Rivera Rodríguez, PhD, SMIEEE — UNAL<br>
<span style="color:#aaa;">★ UCF: Uncertainty Cost Functions — Cuantificando la Incertidumbre Renovable ★</span><br>
<span style="color:#bbb;">100% open-source · Python + Streamlit · 2026</span>
</div>
""", unsafe_allow_html=True)
