# ⚡ LICA-SE — Laboratorio de Inteligencia Computacional Aplicada al Sector Eléctrico

**Plataforma interactiva del grupo de investigación del Prof. Sergio Raúl Rivera Rodríguez (PhD, SMIEEE)**
Universidad Nacional de Colombia — Sede Bogotá

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://lica-se.streamlit.app)

---

## 🎯 Descripción

LICA-SE es una aplicación web interactiva desarrollada en **Python + Streamlit** que presenta las líneas de investigación, publicaciones, simulaciones y herramientas del grupo de investigación en Inteligencia Computacional aplicada al Sector Eléctrico.

### ★ Bandera del grupo: Uncertainty Cost Functions (UCF) ★

Las UCF son formulaciones matemáticas pioneras que **cuantifican el costo económico de la incertidumbre** en sistemas eléctricos con generación renovable (solar, eólica), vehículos eléctricos y almacenamiento de energía (BESS).

## 🚀 Características

- **🎯 UCF Laboratory**: Simulador interactivo con distribuciones Normal, Uniforme, Beta (Solar), Wind y EV
- **🔬 UCF Solar+Eólica**: Validación cruzada analítico vs Monte Carlo
- **💚 UCF Hidrógeno Verde**: Extensión a sistemas de H₂ verde híbridos
- **💰 CS-LCOE**: Controllable System LCOE con internalización de costos de incertidumbre
- **🧪 Simulaciones**: OPF, Microgrid, Gestión de Activos, VNS-DEEPSO, Pronóstico, Monte Carlo
- **📊 Publicaciones**: 168+ artículos con análisis por cuartil Scimago
- **💻 Apps GitHub**: 10+ aplicaciones interactivas desplegadas
- **🌍 Colaboraciones**: 15+ países
- **🏆 Logros**: 4× ganador de competencias IEEE PES

## 📁 Estructura del Proyecto

```
LICA-SE/
├── app.py                 # Aplicación principal Streamlit (~1650 líneas)
├── data.py                # Datos: perfil, publicaciones, apps, tesis
├── ucf_engine.py          # Motor de cálculo UCF y CS-LCOE
├── simulations.py         # Simulaciones de sistemas de potencia
├── requirements.txt       # Dependencias Python
├── .streamlit/
│   └── config.toml        # Configuración Streamlit
├── *.gif                  # Animaciones educativas (10 archivos)
└── README.md
```

## 🛠️ Instalación Local

```bash
# Clonar el repositorio
git clone https://github.com/srrriverar/LICA-SE.git
cd LICA-SE

# Crear entorno virtual (recomendado)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar la aplicación
streamlit run app.py
```

La app estará disponible en `http://localhost:8501`

## 📐 Fundamentos Matemáticos

### Uncertainty Cost Function (UCF)

$$UCF(W_s) = \mathbb{E}\left[C_u \cdot \max(W_{av} - W_s, 0) + C_o \cdot \max(W_s - W_{av}, 0)\right]$$

Donde:
- **W_s**: Potencia programada (decisión de OPF/UC)
- **W_av**: Potencia real disponible (incierta)
- **C_u**: Costo por subestimar (excedente → curtailment)
- **C_o**: Costo por sobrestimar (déficit → reserva)

### CS-LCOE (Controllable System LCOE)

$$CS\text{-}LCOE = \frac{\sum_{t=1}^{T} \frac{CAPEX_t + OPEX_t + E[UCF_t]}{(1+r)^t}}{\sum_{t=1}^{T} \frac{E[P_{s,t}]}{(1+r)^t}}$$

## 🔗 Enlaces

- **Google Scholar**: [Prof. Sergio Rivera](https://scholar.google.com/citations?user=BP8jANUAAAAJ)
- **GitHub**: [srriverar](https://github.com/srriverar)
- **Hermes UNAL**: [Perfil investigador](https://hermes.unal.edu.co/people/srriverar)
- **ResearchGate**: [Sergio Rivera](https://www.researchgate.net/profile/Sergio-Rivera-8)
- **LinkedIn**: [sergio-rivera](https://www.linkedin.com/in/sergio-rivera-7758ba68/)

## 📄 Licencia

Este proyecto es de código abierto con fines educativos y de investigación.

## 🏛️ Institución

**Universidad Nacional de Colombia** — Sede Bogotá
Departamento de Ingeniería Eléctrica y Electrónica
Grupo de Investigación en Energía y Minería Computacional (EMC-UN)

---

<p align="center">
<b>★ UCF: Uncertainty Cost Functions — Cuantificando la Incertidumbre Renovable ★</b><br>
<i>100% open-source · Python + Streamlit · 2026</i>
</p>
