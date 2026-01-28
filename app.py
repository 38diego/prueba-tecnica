import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.ticker as ticker

# Configuración de la página (Título e Icono)
st.set_page_config(
    page_title="Dashboard de Riesgo - Prueba DS",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        /*FONDO*/
        .stApp {
            background-color: #000000;
            background-image: 
                radial-gradient(circle at 90% 90%, rgba(0, 212, 72, 0.15) 0%, transparent 50%),
                radial-gradient(circle at 20% 5%, rgba(0, 212, 72, 0.12) 0%, transparent 60%);
            
            background-attachment: fixed;
            color: #FFFFFF;
        }

        /*SIDEBAR*/
        [data-testid="stSidebar"] {
            background-color: #000000 !important;
            /* Un brillo muy sutil en la parte baja del menú */
            background-image: linear-gradient(to bottom, #000000 80%, rgba(0, 212, 72, 0.05) 100%);
            border-right: 1px solid #1a1a1a;
            color: #FFFFFF;
        }
        
        /* 3. TEXTOS Y TÍTULOS CON NEÓN */
        h1, h2, h3, h4, h5, h6 {
            color: #00D448 !important;
            font-family: 'Segoe UI', sans-serif;
            text-shadow: 0 0 10px rgba(0, 212, 72, 0.4); /* Más brillo en texto */
        }
        
        p, li, label, .stMarkdown, .stRadio label {
            color: #E0E0E0 !important;
        }

        /* 4. MÉTRICAS LED */
        [data-testid="stMetricValue"] {
            color: #00D448 !important;
            text-shadow: 0 0 15px rgba(0, 212, 72, 0.6);
            font-family: 'Courier New', monospace; /* Fuente tipo código para números */
        }
        
        [data-testid="stMetricLabel"] {
            color: #AAAAAA !important;
        }

        /* 5. CÓDIGO ESTILO HACKER */
        code {
            background-color: #0a0a0a !important;
            color: #00D448 !important;
            border: 1px solid #333333;
            font-family: 'Courier New', monospace;
        }
        
        /* 6. PESTAÑAS (TABS) */
        button[data-baseweb="tab"] {
            color: #FFFFFF !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            color: #00D448 !important;
            border-bottom-color: #00D448 !important;
            background-color: rgba(0, 212, 72, 0.05) !important; /* Fondo sutil en tab activa */
        }
        
        /* Ocultar header */
        header {background-color: transparent !important}
            
        hr {
            border-color: #00D448 !important; /* Color de la línea */
            border-width: 2px !important;    /* Grosor */
            opacity: 0.6 !important;         /* Transparencia para que no sea tan agresivo */
        }
    </style>
""", unsafe_allow_html=True)

# --- 1. FUNCIÓN DE CARGA Y LIMPIEZA (Requirement 1 & Preprocesamiento) ---
@st.cache_data
def cargar_datos():
    # Intentamos leer el archivo
    try:
        df = pd.read_excel('PruebaDS.xlsx')
        
        # --- PREPROCESAMIENTO "NUCLEAR" ---
        # 1. Asegurar que 'pago' es 0 o 1 (Entero)
        df['pago'] = pd.to_numeric(df['pago'], errors='coerce').fillna(0).astype(int)
        
        return df
    except FileNotFoundError:
        return None

# Cargamos los datos
df = cargar_datos()

# --- SIDEBAR (Navegación) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=50) # Icono genérico
    st.title("Navegación")
    opcion = st.radio(
        "Ir a:",
        ["1. Introducción & Data", "2. Análisis Exploratorio (EDA)", "3. Modelado (Próximamente)", "4. SQL (Próximamente)"]
    )
    
# --- LÓGICA DE PÁGINAS ---

if df is None:
    st.error("⚠️ No se encontró el archivo 'PruebaDS.xlsx'. Por favor cárgalo en la carpeta del proyecto.")
    st.stop()

# PÁGINA 1: INTRODUCCIÓN
if opcion == "1. Introducción & Data":
    st.title("📊 Prueba Técnica: Científico de Datos")
    st.markdown("""
    **Objetivo:** Evaluar la probabilidad de que un deudor realice el pago.
    """)    

    st.markdown("""
    ### Diccionario de Datos

    **A. Perfil del Deudor (Demográfico)**
    *   **`identificacion` / `tipo_documento`**: Identificadores únicos del cliente. (C: Cédula Ciudadanía, E: Cédula Extranjería, T: Tarjeta Identidad, P: Pasaporte).
    *   **`genero`**: Sexo del cliente.
    *   **`rango_edad_probable`**: Grupo etario estimado.
    *   **`departamento`**: Ubicación geográfica.

    **B. Estado de la Deuda (Financiero)**
    *   **`mes`**: Fecha de corte de la información.
    *   **`saldo_capital`**: El monto principal que debe la persona (sin intereses de mora).
    *   **`dias_mora`**: Cuántos días han pasado desde que debió pagar.
    *   **`banco`**: La entidad dueña de la deuda original.
    *   **`antiguedad_deuda`**: Fecha en que se originó la obligación.

    **C. Comportamiento de Pago (Histórico)**
    *   **`pago_mes_anterior`**: Indica si pagó algo el mes pasado (1=Sí, 0=No).
    *   **`meses_desde_ultimo_pago`**: Recencia. Si es alto o nulo, es un cliente difícil.
    *   **`sin_pago_previo`**: Bandera que indica si nunca ha realizado un pago.

    **D. Gestión y Resultado (Operativo)**
    *   **`contacto_mes_actual` / `anterior` / `ultimos_6meses`**: Mide la intensidad de la gestión.
    *   **`duracion_llamadas_ultimos_6meses`**: Calidad del contacto.
    *   **`pago`**: **Variable Objetivo (Target)**. (1 = Recuperó, 0 = No recuperó).
    """, unsafe_allow_html=True)

    # Vista preliminar de datos
    st.header("1. Lectura y Estructura de Datos")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Registros", df.shape[0])
    col2.metric("Total Columnas", df.shape[1])
    col3.metric("Tasa Global de Pago", f"{df['pago'].mean()*100:.2f}%")
    
    st.write("Vista previa de las primeras filas:")
    st.dataframe(df.head())

    st.header("2. Limpieza y Distribución")
    
    st.subheader("Análisis de Duplicados")
    st.markdown("""
    Es importante mencionar que, para el análisis de duplicados, **no se tuvieron en cuenta** las columnas `mes` ni `antiguedad_deuda`.
    
    *   **`mes`**: Se excluye debido a la incertidumbre sobre si es un mes de registro o de corte, lo que genera diferencias en filas que describen el mismo estado de deuda.
    *   **`antiguedad_deuda`**: Se excluye por la gran cantidad de valores vacíos.

    **Ejemplo (Cliente 513810):**
    A continuación se observa cómo, para saldos y días de mora idénticos, existen valores de mes distintos y vacíos en la antigüedad.
    """)

    # Mostrar ejemplo del cliente 513810 para justificar la exclusión de variables
    st.dataframe(df[df['identificacion'].astype(str) == '513810'].sort_values(by= "saldo_capital",ascending=False))

    # --- B. ESTRATEGIA DE PRIORIZACIÓN (EL TRUCO) ---
    st.markdown("""
    **Estrategia de Limpieza:**
    Se decidió eliminar los duplicados conservando el registro con mayor información. Para esto, se ordenaron los datos priorizando aquellos que tienen fecha en `antiguedad_deuda`, asegurando que al eliminar duplicados se mantenga el registro más completo.
    """)

    # 1. Ordenamos por 'antiguedad_deuda'. 'na_position=last' empuja los vacíos al final.
    #    Así, las filas con fecha quedan ARRIBA del todo.
    df.sort_values(by='antiguedad_deuda', na_position='last', inplace=True)

    cols_modelo = [
    'tipo_documento', 'identificacion', 'genero', 'rango_edad_probable', 
    'departamento', 'saldo_capital', 'dias_mora', 'banco', 
    'pago_mes_anterior', 'meses_desde_ultimo_pago', 'sin_pago_previo', 
    'contacto_mes_actual', 'contacto_mes_anterior', 'contacto_ultimos_6meses', 
    'duracion_llamadas_ultimos_6meses', 'pago'
    ]

    # 2. Borramos duplicados quedándonos con el PRIMERO (keep='first')
    #    Como ordenamos antes, el "primero" es el que tiene fecha.
    df.drop_duplicates(subset=cols_modelo, keep='first', inplace=True)

    st.success(f"**Resultado Final:** El dataset ahora cuenta con **{df.shape[0]}** registros únicos, donde se puede encontrar un mismo cliente mas de una vez pero con deudas disntatas")
    st.write("**Valores faltantes por columna tras la limpieza:**")
    st.dataframe(df.isnull().sum().to_frame(name='Faltantes').T)

    # 1. Tu lista de variables DEFINITIVA (Sin mes, sin antiguedad)
    st.subheader("Inconsistencias y Normalización de Datos Clave")
    col1, col2 = st.columns([0.3,0.7])
    # --- PARLA (Explicación del negocio) ---
    with col1:
        st.markdown(f"""
        ### Genero
        presenta inconsistencias en la captura de datos. Para garantizar la calidad del análisis, se aplica la siguiente **lógica de normalización**:
        *   **`M`** se estandariza a **`HOMBRE`**.
        *   **`F`** se estandariza a **`MUJER`**.
        *   Los marcados como "NO APLICA" puede llegar a ser inconsistente, si se quiere respetar la diversidad de género se podria cambiar a "OTROS", pero en este caso se opta por etiquetarlos como **`NO ESPECIFICADO`**.
        *   Ademas los {df.genero.isnull().sum()} valores vacíos o nulos se etiquetan como **`No especificado`**.
        """)

        # Capturar valores antes de limpiar
        valores_sucios = df['genero'].unique()

        # Limpieza
        df['genero'] = df['genero'].replace({'M': 'HOMBRE', 'F': 'MUJER', ' ': 'No especificado', 'NO APLICA': 'No especificado'})
        df['genero'] = df['genero'].fillna('No especificado')
        
        valores_limpios = df['genero'].unique()
    
        st.warning("""
    **Alerta de Calidad de Datos**
    El análisis revela una debilidad estructural en la captura de información: **El 49.2% de la cartera (13,884 clientes) carece de identificación de género.**
    Esto representa un **'Punto Ciego Operativo'**. Al desconocer la identidad de casi la mitad de la población, cualquier segmentación tradicional por género será imprecisa.
    """)

    with col2:
        # --- GRÁFICA ESTILIZADA ---
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Fondo transparente para que luzcan los efectos CSS
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)

        sns.countplot(data=df, x='genero', hue='genero', palette='viridis', order=df['genero'].value_counts().index, ax=ax, edgecolor='white', legend=False)

        ax.set_title('Distribución de Clientes por Género', color='#00D448', fontsize=16, fontweight='bold')
        ax.set_xlabel('Género', color='white', fontsize=12)
        
        # Quitar Eje Y y Recuadro (Spines) para look minimalista
        ax.set_ylabel('')
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.tick_params(axis='x', colors='white', labelsize=10)

        # Etiquetas de datos
        for container in ax.containers:
            ax.bar_label(container, fmt='%d', padding=3, color='white', fontsize=12, fontweight='bold')

        st.pyplot(fig) 


    st.success("""
    **Perfilamiento del Segmento Identificado**
    Dentro del 51% de clientes que **sí** tienen datos, existe un claro sesgo:
    * Por cada mujer, hay **1.7 hombres** (9,101 vs 5,218).
    * El producto tiene una tracción histórica mucho mayor en el segmento masculino.
    """)

    st.divider()

    col1, col2 = st.columns([0.3, 0.7])
    
    with col1:
        st.markdown("""
        ### rango_edad_probable 
        contenía múltiples rangos superpuestos y formatos inconsistentes. 
        Se aplicó una **lógica de agrupación** para unificar estos valores:""")

        # Capturar valores sucios
        valores_edad_sucios = df['rango_edad_probable'].unique()

        # Definir mapeo
        mapa_edad = {
            '18-21': '18-25', '18-25': '18-25', '22-25': '18-25',
            '25-30': '26-35', '26-29': '26-35', '30-33': '26-35', '31-35': '26-35', '34-37': '26-35',
            '36-40': '36-45', '38-41': '36-45', '41-45': '36-45', '42-45': '36-45',
            '46-49': '46-55', '46-50': '46-55', '50-53': '46-55', '51-55': '46-55',
            '54-57': '56-65', '56-60': '56-65', '58-61': '56-65', '61-65': '56-65', '62-65': '56-65',
            '66+': 'Mayor a 65', '66-70': 'Mayor a 65', '71-75': 'Mayor a 65', 'Mas de 75': 'Mayor a 65'
        }

        # Limpieza        
        valores_edad_limpios = sorted(df['rango_edad_probable'].unique().astype(str))

        st.caption("Unificados")
        st.write(f"""
                    *   **18-25**: Jóvenes.
                    *   **26-35**: Adultos Jóvenes.
                    *   **36-45**: Adultos.
                    *   **46-55**: Adultos Maduros.
                    *   **56-65**: Mayores.
                    *   **Mayor a 65**: Tercera Edad.
                    *   **No especificado**: Los {df['rango_edad_probable'].isnull().sum()} Datos faltantes y los "NO APLICA" se marcan como no especificaods.
                     """)
            
        df['rango_edad_probable'] = df['rango_edad_probable'].replace(mapa_edad)
        df['rango_edad_probable'] = df['rango_edad_probable'].replace({'NO APLICA': 'No especificado'}).fillna('No especificado')

    with col2:
        # Gráfica
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)

        order_edad = ['18-25', '26-35', '36-45', '46-55', '56-65', 'Mayor a 65', 'No especificado']
        order_edad = [x for x in order_edad if x in df['rango_edad_probable'].unique()]

        sns.countplot(data=df, x='rango_edad_probable', hue='rango_edad_probable', palette='magma', order=order_edad, ax=ax, edgecolor='white', legend=False)

        ax.set_title('Distribución de Clientes por Edad', color='#00D448', fontsize=16, fontweight='bold')
        ax.set_xlabel('Rango de Edad', color='white', fontsize=12)
        
        ax.set_ylabel('')
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.tick_params(axis='x', colors='white', labelsize=10)

        for container in ax.containers:
            ax.bar_label(container, fmt='%d', padding=3, color='white', fontsize=12, fontweight='bold')

        st.pyplot(fig)


    # 1. Calidad de Datos (Crucial para Skip Tracing - Localización)
    st.warning("""
    **Riesgo Operativo: Datos Faltantes (25%)**
    El 25.1% de la cartera (7,071 deudores) no tiene edad registrada.
    * **Impacto en Cobranza:** Esto dificulta la segmentación de la estrategia. No es lo mismo negociar con un joven que inicia su vida crediticia que con un pensionado. Al no tener la edad, perdemos la capacidad de personalizar el guion de cobro según la etapa de vida del deudor.
    """)

    # 2. El Grueso de la Cartera (Donde está la plata)
    st.success("""
    **Foco de Gestión: Población Económicamente Activa (26-45 años)**
    El 40% de los deudores se concentra en las edades de **26 a 45 años**.
    * **Lectura de Negocio:** Es lógico, ya que es la etapa de mayor consumo y endeudamiento (hipotecas, vehículos, tarjetas).
    * **Oportunidad:** Este segmento suele estar laboralmente activo. La estrategia de recuperación aquí debe enfocarse en **acuerdos de pago basados en flujo de caja (salario)** o, en última instancia, medidas sobre ingresos laborales.
    """)

    # 3. La Anomalía del Riesgo (Jóvenes vs. Tercera Edad)
    st.info("""
    **Perfil de Riesgo Atípico:**
    * **Riesgo en Tercera Edad (>65 años):** Hay **8 veces más deudores mayores de 65 años** que jóvenes. Esto representa un riesgo de recuperación alto:
        1.  Ingresos fijos limitados (pensiones).
        2.  Protecciones legales reforzadas.
        3.  Riesgo de incobrabilidad por fallecimiento.
    """)

    st.divider()

    col1, col2 = st.columns([0.3, 0.7])

    with col1:
        st.markdown("""
        ### saldo_capital 
        corresponde al monto principal de la obligación pendiente.
        
        **Importancia:**
        Entender la distribución de los montos permite segmentar la estrategia de cobranza:
        *   **Saldos Bajos:** Gestión masiva/digital.
        *   **Saldos Altos:** Gestión personalizada.
        """)
        
        st.write("**Estadísticas Descriptivas:**")
        st.dataframe(df['saldo_capital'].describe().to_frame().style.format("${:,.0f}"))

    with col2:
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)

        sns.histplot(data=df, x='saldo_capital', kde=True, color='#00D448', ax=ax, edgecolor='#222222')

        ax.set_title('Distribución del Saldo Capital', color='#00D448', fontsize=16, fontweight='bold')
        ax.set_xlabel('Saldo Capital (COP)', color='white', fontsize=12)
        ax.set_ylabel('Frecuencia', color='white', fontsize=12)
        
        ax.tick_params(axis='x', colors='white', labelsize=9, rotation=15)
        ax.tick_params(axis='y', colors='white', labelsize=10)
        
        for spine in ax.spines.values():
            spine.set_visible(False)
            
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'${x:,.0f}'))

        st.pyplot(fig)

    # 1. El conflicto Media vs Mediana (Esencial en finanzas)
    st.info("""
    **La "Trampa del Promedio" (Media vs. Mediana)**
    Esta gráfica muestra una clásica distribución de **"Cola Larga" (Long Tail)**, típica en carteras financieras.
    * **Media Inflada 5.7M$:** El promedio es engañoso porque los grandes deudores (valores extremos) lo empujan hacia arriba.
    * **La Realidad 2.3M\$:** El dato real de gestión es que el 50% de los clientes debe menos de \$2.3 millones.
    * **Conclusión:** Diseñar metas o incentivos basados en el promedio (\$5.7M) sería un error, ya que la mayoría de la cartera no llega a ese monto.
    """)

    # 2. La Estrategia de Segmentación (El insight más valioso)
    st.success("""
    **Estrategia Sugerida por Cuartiles (Costo-Eficiencia)**
    Los cuartiles nos dictan qué canal de cobranza usar para maximizar el retorno:
    * **Masivo / Digital (Q1 < \$1.2M):** El 25% de la base debe menos de \$1.2M. Aquí, el costo de una llamada humana podría superar la ganancia esperada. **Recomendación:** SMS, Email, Bots.
    * **Gestión Híbrida (Q2 - Q3):** El grueso de la población.
    * **VIP / Especializada (Top 25% > \$6.2M):** Este grupo concentra el mayor capital en riesgo (llegando hasta \$113M). **Recomendación:** Asignar a los mejores negociadores, ya que recuperar una sola de estas cuentas equivale a recuperar 50 de las pequeñas.
    """)

    # 3. Limpieza de Datos (Anomalías)
    st.warning("""
    **Ruido Operativo: Micro-Saldos**
    Se detectó un valor mínimo de **\$500 pesos**.
    * **Diagnóstico:** Estos son probablemente "residuos de caja" (pagos mal aplicados o intereses residuales).
    * **Acción Técnica:** Se deben filtrar y excluir del modelo predictivo y de la gestión telefónica. Llamar a cobrar \$500 pesos destruye valor y genera fricción innecesaria con el cliente.
    """)
    
    st.divider()

    col1, col2 = st.columns([0.3, 0.7])

    with col1:
        st.markdown("""
        ### dias_mora 
        indica el tiempo transcurrido desde que el cliente debió realizar el pago límite.
        
        **Importancia (El Termómetro):**
        Esta variable define la etapa de gestión:
        *   **Preventiva:** Mora baja (recién vencido).
        *   **Administrativa:** Mora media.
        *   **Jurídica/Castigo:** Mora muy alta (difícil recuperación).
        """)
        
        st.write("**Estadísticas:**")
        st.dataframe(df['dias_mora'].describe().to_frame().style.format("{:,.0f}"))

    with col2:
        fig, ax = plt.subplots(figsize=(10, 7))
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)

        sns.histplot(data=df, x='dias_mora', kde=True, color='#00D448', ax=ax, edgecolor='#222222', bins=30)

        ax.set_title('Distribución de Días de Mora', color='#00D448', fontsize=16, fontweight='bold')
        ax.set_xlabel('Días de Mora', color='white', fontsize=12)
        ax.set_ylabel('Frecuencia', color='white', fontsize=12)
        
        ax.tick_params(axis='x', colors='white', labelsize=10)
        ax.tick_params(axis='y', colors='white', labelsize=10)
        
        for spine in ax.spines.values():
            spine.set_visible(False)

        st.pyplot(fig)

    # 1. El descubrimiento del Tipo de Negocio (El Insight más fuerte)
    st.warning("""
    Los datos revelan la naturaleza real de la operación:
    * **El Hallazgo:** El primer cuartil (25% más reciente) comienza en **644 días de mora** (casi 2 años).
    * **Conclusión:** No estamos gestionando créditos vigentes ni mora temprana. Estamos ante una **Cartera Castigada**.
    * **Implicación:** Las estrategias de "retención" o "preventivas" no aplican. Aquí se requiere una estrategia de **negociación de quitas y condonaciones**, ya que el cliente lleva años sin pagar.
    """)

    # 2. Análisis de la Distribución (Técnico)
    st.info("""
    La curva no es uniforme, presenta tres picos claros (~500, ~1,500 y ~2,500 días).
    * No estamos ante una población homogénea. Estos picos probablemente representan **"Cosechas" (Vintages)** específicas o compras de cartera masivas realizadas en años distintos.
    * El modelo debería incluir la variable "Antigüedad de la Deuda" como un factor de segmentación, ya que la propensión de pago de un deudor de 2 años es estructuralmente diferente a la de uno de 7 años.
    """)

    # 3. Limpieza de Datos (Legal/Outliers)
    st.error("""
    Se detectaron registros con **27 años de mora** (máx: 10,031 días).
    * Gran parte de esta deuda podría estar **prescrita legalmente**, lo que hace imposible su cobro jurídico.
    * Estos registros son ruido puro para un modelo de predicción de pago (Probabilidad $\\approx 0$). Se recomienda excluirlos del entrenamiento para no ensuciar los patrones de la deuda recuperable.
    """)

    st.divider()

    col1, col2 = st.columns([0.3, 0.7])

    with col1:
        st.markdown("""
        ### Banco
        indica la entidad financiera propietaria de la obligación. Esta variable se encuentra **totalmente limpia**: no presenta valores nulos ni categorías inconsistentes, por lo que podemos visualizar la participación de mercado directamente.
        """)

        # 1. Calidad del Dato (La buena noticia)
        st.success("""
        **✅ Integridad de Datos: Variable Limpia**
        A diferencia de los retos demográficos anteriores, la variable `banco` presenta una **completitud del 100%**.
        * Esto convierte al "Banco de Origen" en una variable pilar (Feature de alta confianza) para la segmentación y el entrenamiento del modelo.
        """)

    with col2:
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)

        # Usamos barras horizontales (y='banco') para leer mejor los nombres si son largos
        sns.countplot(data=df, y='banco', order=df['banco'].value_counts().index, hue='banco', palette='viridis', ax=ax, edgecolor='white', legend=False)

        ax.set_title('Cartera por Banco', color='#00D448', fontsize=16, fontweight='bold')
        ax.set_xlabel('Número de Clientes', color='white', fontsize=12)
        ax.set_ylabel('')
        
        ax.tick_params(axis='x', colors='white', labelsize=10)
        ax.tick_params(axis='y', colors='white', labelsize=11)
        
        for spine in ax.spines.values():
            spine.set_visible(False)
            
        for container in ax.containers:
            ax.bar_label(container, fmt='%d', padding=3, color='white', fontsize=10, fontweight='bold')

        st.pyplot(fig)

    # 2. El Principio de Pareto (El negocio)
    st.info("""
        **📊 Ley de Pareto en Acción (Concentración de Riesgo)**
        La cartera presenta una alta dependencia de dos originadores principales:
        * **Davivienda (13,463) + Colpatria (9,254)** agrupan a **22,717 clientes**.
        * **Lectura de Negocio:** El **~80% de la operación** depende de las políticas de crédito de estas dos entidades. Entender sus perfiles de riesgo explica el comportamiento macro de la cartera.
        """)
    
    st.divider()

    st.markdown("""
        ### Antiguedad de la Deuda
""")

    # 1. El Argumento Irrefutable (Calidad de Datos)
    st.error("""
    **Eliminación de Variable**
    Se ha decidido excluir esta variable del analicis y del modelo predictivo por una razón crítica de integridad:
    * **Nulidad Extrema:** Presenta **20,173 valores nulos**, lo que representa el **71.5% de los datos perdidos**.
    * **Principio de No-Invención:** imputar más del 30-40% de una variable introduce un sesgo artificial severo. Tratar de rescatar una variable con el 70% de faltantes implicaría "fabricar" la historia crediticia de la mayoría de los clientes.
    """)

    # 3. El Veredicto Final
    st.success("""
    El modelo se entrenará utilizando únicamente **`dias_mora`**.
    Esta decisión prioriza la **calidad del dato**. Estimar la antigüedad real de la deuda con tan poca información sería contraproducente y dañino para la precisión del modelo.
    """)

    st.divider()

    col1, col2 = st.columns([0.3, 0.7])

    with col1:
        st.markdown("""
        ### pago_mes_anterior 
        es una variable binaria que indica si el cliente realizó algún abono en el mes inmediatamente anterior al corte.
        """)
        
        st.error("""
        La gráfica es contundente: el **99.3% de la base no realizó pagos el mes pasado**.
        * El Reto no se trata de administrar clientes activos, se trata de **reactivar clientes inactivos**
        """)

        st.success("""
        Ese pequeño grupo del **0.7% (aprox. 197 clientes)** que sí pagó el mes pasado es el activo más valioso de la base.
        * El mejor predictor del futuro es el pasado inmediato. Si un cliente pagó hace 30 días, su probabilidad de pagar hoy es exponencialmente más alta que la del resto.
        * Estos no son deudores fríos; son clientes con **Voluntad de Pago demostrada y capacidad de caja activa**. A pesar de ser una minoría estadística, este grupo debe tener **Prioridad Absoluta** en la gestión.
        
        """)

    with col2:
        fig, ax = plt.subplots(figsize=(2,2))
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)

        # Datos
        datos = df['pago_mes_anterior'].value_counts().reindex([1, 0], fill_value=0)
        labels = ['Sí Pagó', 'No Pagó']
        colors = ['#00D448', '#FF4B4B'] # Verde y Gris

        # Gráfica de Dona
        wedges, texts, autotexts = ax.pie(
            datos, 
            labels=None,       # <--- Esto oculta los NOMBRES en la gráfica
            colors=colors, 
            autopct='%1.1f%%', # <--- Esto muestra los VALORES en la gráfica
            startangle=90, 
            pctdistance=1.15,
            wedgeprops=dict(width=0.4, edgecolor='#111111'),
            textprops=dict(color="white", fontsize=12, fontweight='bold')
        )

        # 2. Configuración de la Leyenda
        # frameon=False quita el recuadro y el fondo
        leg = ax.legend(wedges, labels,
                title="Categoría",
                loc="center left",
                bbox_to_anchor=(1, 0, 0.5, 1),
                frameon=False,      # <--- AQUÍ se hace transparente el fondo
                labelcolor='white'  # <--- Opcional: pone el texto de la leyenda en blanco
        )

        # Si quieres cambiar el color del título de la leyenda a blanco también:
        plt.setp(leg.get_title(), color='white')

        
        st.pyplot(fig)
    

    st.divider()

    col1, col2 = st.columns([0.3, 0.7])

    with col1:
        st.markdown("""
        La variable `sin_pago_previo` nos indica si el cliente ha realizado algun pago antes o si es un caso de "cero pagos" históricos, ademas de ser de gran ayuda para el analisis ya que no tiene valores vacíos.
        
        **Importancia del Hábito:**
        *   **Con Pago Previo (0):** Ya rompió la inercia. ha pagado y ha tenido voluntad antes. Es más fácil de recuperar.
        *   **Sin Pago Previo (1):** Es el perfil más riesgoso.
        """)

        
        # 1. El Hallazgo Financiero (La diferencia del 1%)
        st.warning("""
        Al cruzar los datos históricos (1.7% ha pagado alguna vez) vs. los actuales (0.7% pagó el mes pasado), encontramos una brecha crítica de 272 clientes**.
        * **¿Quiénes son?** Son **Pagadores Caídos**. Clientes que ya demostraron voluntad y capacidad de pago en el pasado, pero que recientemente se detuvieron.
        * **Oportunidad de Negocio:** Este grupo representa la **ganancia rápida**. Convencer a alguien que ya pagó es 5 veces más barato y rápido que convencer a un deudor crónico.
        """)

    with col2:
        fig, ax = plt.subplots(figsize=(2, 2))
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)

        # Datos: 1 = Sin pago previo (Malo), 0 = Con pago previo (Bueno)
        datos = df['sin_pago_previo'].value_counts().reindex([1, 0], fill_value=0)
        labels = ['Nunca ha\nPagado', 'Ha Pagado\nAntes']
        colors = ['#FF4B4B', '#00D448'] # Rojo para alerta, Verde para positivo

        wedges, texts, autotexts = ax.pie(
            datos, 
            labels=None, 
            colors=colors, 
            autopct='%1.1f%%',
            startangle=90, 
            pctdistance=1.15,
            wedgeprops=dict(width=0.4, edgecolor='#111111'),
            textprops=dict(color="white", fontsize=12, fontweight='bold')
        )
        
        # Configuración de la Leyenda (Igual a pago_mes_anterior)
        leg = ax.legend(wedges, labels, title="Categoría", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), frameon=False, labelcolor='white')
        plt.setp(leg.get_title(), color='white')
                
        st.pyplot(fig)

    st.markdown("Aqui algunos de los clientes que cumplen con esta condición de 'Pagadores Caídos':")

    st.dataframe(df[(df.pago_mes_anterior == 0) & (df.sin_pago_previo == 0)].head(5))

    st.divider()

    col1, col2 = st.columns([0.3, 0.7])

    with col1:
        st.markdown("""
        ### meses_desde_ultimo_pago 
        """)

        st.info("""
        La gran barra de valores nulos no es un error de datos, es información y significa que nunca han pagado
        * En lugar de imputar estos valores, el modelo tratará los Nulos como una categoría explícita -1. Esto por que el comportamiento de alguien que *nunca* ha pagado es estructuralmente distinto al de alguien que pagó hace 6 meses. No se deben mezclar en el análisis.
        """)

        st.error("""
        **1. El "Abismo" de Recuperación (Mes 3)**
        Los datos revelan un patrón de comportamiento dramático:
        * **Mes 1 a 2:** La retención se mantiene estable (206 $\\to$ 191 clientes). El cliente aún está "tibio".
        * **Mes 2 a 3:** Ocurre una **caída catastrófica del 75%** (de 191 bajamos a solo 47 clientes).
        * Nuestra **Ventana de Oportunidad es de exactamente 60 días**. Si un cliente interrumpe sus pagos y no logramos reactivarlo en los primeros 2 meses, se convierte en un caso dificil de reactivar.
        """)

    with col2:
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)

        # Rellenamos con -1 para visualizar
        df_viz = df.copy()
        df_viz['meses_viz'] = df_viz['meses_desde_ultimo_pago'].fillna(-1).astype(int)

        sns.countplot(data=df_viz, x='meses_viz', color='#00D448', ax=ax, edgecolor='#222222')

        ax.set_title('Distribución de Recencia (Meses)', color='#00D448', fontsize=16, fontweight='bold')
        ax.set_xlabel('Meses desde Último Pago', color='white', fontsize=12)
        ax.set_ylabel('')
        ax.set_yticks([])
        
        ax.tick_params(axis='x', colors='white', labelsize=10)
        
        # Personalizar etiquetas del eje X: Cambiamos '-1' por 'Nunca'
        labels = [item.get_text() for item in ax.get_xticklabels()]
        new_labels = ['Nunca' if x == '-1' else x for x in labels]
        ax.set_xticklabels(new_labels, rotation=45)

        for spine in ax.spines.values():
            spine.set_visible(False)
            
        for container in ax.containers:
            ax.bar_label(container, fmt='%d', padding=3, color='white', fontsize=9)
            
        st.pyplot(fig)

    # 2. Estrategia Operativa (Qué hacer)
    st.success("""
    **2. Estrategia de Intensidad Diferenciada**
    Basado en este hallazgo, la operación debe dividirse en dos fases:
    * **Fase de Choque (Días 1-60):** Gestión humana intensiva y negociación personalizada. Aquí es donde se recupera el dinero. Cada día cuenta antes de llegar al "abismo".
    * **Fase de Mantenimiento (Día 61+):** Una vez cruzada la frontera del mes 3, el costo de llamar supera la probabilidad de éxito. Estos casos deben migrar a **Canales Digitales (Low Cost)** o procesos jurídicos, liberando a los asesores para atender la Fase de Choque.
    """)

    st.divider()

    col1, col2 = st.columns([0.3, 0.7])

    with col1:
        st.markdown("""
        **Variables de Gestión (Operativo):**
        
        Estas métricas reflejan la intensidad de la cobranza realizada sobre el cliente.
        
        *   **`contacto_mes_actual` / `anterior`**: Cantidad de gestiones recientes.
        *   **`contacto_ultimos_6meses`**: Historial de insistencia.
        *   **`duracion_llamadas...`**: Calidad del contacto (tiempo acumulado).
        
        Analizamos la distribución general para entender el esfuerzo operativo promedio de la compañía.
        """)

    with col2:
        variables_gestion = [
            'contacto_mes_actual', 
            'contacto_mes_anterior', 
            'contacto_ultimos_6meses', 
            'duracion_llamadas_ultimos_6meses'
        ]

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        fig.patch.set_alpha(0.0)
        axes = axes.flatten()

        for i, col in enumerate(variables_gestion):
            ax = axes[i]
            ax.patch.set_alpha(0.0)

            # Violin Plot General (Sin discriminación)
            sns.violinplot(data=df, y=col, color='#00D448', ax=ax, linewidth=1.5, inner="quartile")
            
            # Estilizado
            titulo = col.replace('_', ' ').replace('ultimos', 'últ.').title()
            ax.set_title(titulo, color='#00D448', fontsize=12, fontweight='bold')
            ax.set_ylabel('', color='white')
            ax.set_xlabel('', color='white')
            ax.tick_params(axis='y', colors='white', labelsize=9)
            ax.set_xticks([])
            for spine in ax.spines.values(): spine.set_visible(False)

        plt.tight_layout()
        st.pyplot(fig)  


# PÁGINA 2: EDA (Aquí ponemos tu gráfica estrella)
elif opcion == "2. Análisis Exploratorio (EDA)":
    st.title("🔍 Análisis Exploratorio de Datos (EDA)")
    st.markdown("Identificación de patrones clave que diferencian a los clientes que pagan de los que no.")

    # Tabs para organizar la historia
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Drivers (Correlación)", 
        "⏱️ Recencia (Hábito)", 
        "💰 Financiero & Bancos", 
        "👥 Demográfico", 
        "📞 Gestión Operativa"
    ])

    # --- TAB 1: CORRELACIONES ---
    with tab1:
        st.subheader("¿Qué variables mueven la aguja?")
        st.markdown("""
        Utilizamos la **Correlación de Spearman** porque captura relaciones no lineales y es más robusta a valores atípicos (outliers) que la de Pearson.
        """)
        
        # Preparar datos numéricos
        df_num = df.select_dtypes(include=[np.number]).copy()
        
        if 'pago' in df_num.columns:
            # Calcular correlación con el target
            target_corr = df_num.corrwith(df['pago'], method='spearman').sort_values(ascending=False).to_frame(name='Correlación con Pago')
            
            fig, ax = plt.subplots(figsize=(8, 6))
            fig.patch.set_alpha(0.0)
            ax.patch.set_alpha(0.0)
            
            sns.heatmap(target_corr, annot=True, fmt=".2f", cmap='RdBu_r', vmin=-1, vmax=1, 
                        cbar=False, ax=ax, annot_kws={"size": 12, "weight": "bold"})
            
            ax.set_title('Impacto de Variables en la Recuperación', color='#00D448', fontsize=16, fontweight='bold')
            ax.tick_params(axis='y', colors='white', labelsize=12)
            ax.tick_params(axis='x', colors='white')
            
            st.pyplot(fig)
        else:
            st.warning("No se encontraron variables numéricas suficientes.")

    # --- TAB 2: RECENCIA ---
    with tab2:
        st.subheader("Probabilidad de Cobro según Antigüedad del Último Pago")
        st.markdown("Desglose de la tasa de éxito según cuántos meses han pasado desde el último pago del cliente.")

        # Preparación de datos para el grid 3x3
        df_plot = df.copy()
        df_plot['meses_clean'] = df_plot['meses_desde_ultimo_pago'].fillna(-1).astype(int)
        nombre_sin_pago = "Sin Pagos"
        df_plot['meses_cat'] = df_plot['meses_clean'].apply(lambda x: nombre_sin_pago if x == -1 else str(x))

        meses_target = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        meses_existentes = [x for x in meses_target if x in df_plot['meses_clean'].unique()]
        periodos_clave = [nombre_sin_pago] + [str(x) for x in meses_existentes]
        periodos_clave = periodos_clave[:9]

        # Gráfica 3x3
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        fig.patch.set_alpha(0.0)
        axes = axes.flatten()
        colores = ["#333333", '#00D448'] # Gris oscuro y Verde Neón

        for i, periodo in enumerate(periodos_clave):
            ax = axes[i]
            ax.patch.set_alpha(0.0)
            
            datos_periodo = df_plot[df_plot['meses_cat'] == periodo]
            conteo = datos_periodo['pago'].value_counts().reindex([0, 1], fill_value=0)
            
            if sum(conteo) > 0:
                wedges, texts, autotexts = ax.pie(
                    conteo.values, colors=colores, autopct=lambda p: f'{p:.1f}%' if p > 0 else '',
                    startangle=90, pctdistance=0.85,
                    wedgeprops=dict(width=0.4, edgecolor='black'),
                    textprops=dict(color="white", fontsize=10, fontweight='bold')
                )
                ax.text(0, 0, f"N={sum(conteo)}", ha='center', va='center', color='white', fontsize=10)
            
            titulo_grafica = "Nunca Pagó" if periodo == nombre_sin_pago else f"Hace {periodo} Meses"
            ax.set_title(titulo_grafica, color='white', fontsize=11, fontweight='bold')

        # Limpiar ejes vacíos
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
            
        st.pyplot(fig)
        
        st.markdown("---")
        st.subheader("Impacto del Historial Crediticio")
        
        col1, col2 = st.columns([0.6, 0.4])
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_alpha(0.0)
            ax.patch.set_alpha(0.0)
            
            sns.countplot(data=df, x='sin_pago_previo', hue='pago', palette={0: '#555555', 1: '#00D448'}, ax=ax, edgecolor='black')
            
            ax.set_title('Volumen de Clientes: Con vs Sin Historial', color='white')
            ax.set_xticklabels(['Con Historial', 'Sin Historial'], color='white')
            ax.set_ylabel('Cantidad', color='white')
            ax.tick_params(colors='white')
            ax.legend(title='Pago', labels=['No', 'Sí'], labelcolor='white')
            
            for container in ax.containers:
                ax.bar_label(container, fmt='%d', padding=3, color='white')
            
            for spine in ax.spines.values(): spine.set_visible(False)
            st.pyplot(fig)
        
        with col2:
            st.info("""
            **Interpretación:**
            La gráfica de barras muestra que, aunque el grupo "Sin Historial" es numeroso, su tasa de pago (barra verde) es extremadamente baja comparada con quienes ya tienen historial.
            """)

    # --- TAB 3: FINANCIERO & BANCOS ---
    with tab3:
        st.subheader("Perfil Financiero del Deudor")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Días de Mora vs Pago**")
            fig, ax = plt.subplots(figsize=(6, 5))
            fig.patch.set_alpha(0.0)
            ax.patch.set_alpha(0.0)
            sns.boxplot(data=df, x='pago', y='dias_mora', palette=['#555555', '#00D448'], ax=ax, showfliers=False)
            ax.set_xticklabels(['No Pagó', 'Sí Pagó'], color='white')
            ax.set_ylabel('Días de Mora', color='white')
            ax.set_xlabel('')
            ax.tick_params(colors='white')
            for spine in ax.spines.values(): spine.set_visible(False)
            st.pyplot(fig)
            
        with col2:
            st.markdown("**Saldo Capital vs Pago (Log)**")
            fig, ax = plt.subplots(figsize=(6, 5))
            fig.patch.set_alpha(0.0)
            ax.patch.set_alpha(0.0)
            sns.boxplot(data=df, x='pago', y='saldo_capital', palette=['#555555', '#00D448'], ax=ax, showfliers=False)
            ax.set_yscale('log')
            ax.set_xticklabels(['No Pagó', 'Sí Pagó'], color='white')
            ax.set_ylabel('Saldo Capital ($)', color='white')
            ax.set_xlabel('')
            ax.tick_params(colors='white')
            for spine in ax.spines.values(): spine.set_visible(False)
            st.pyplot(fig)

        st.divider()
        st.subheader("Calidad de Cartera por Banco (Top 10)")
        
        # Lógica para gráfico apilado de bancos
        top_bancos = df['banco'].value_counts().index[:10]
        df_top = df[df['banco'].isin(top_bancos)].copy()
        tabla = pd.crosstab(df_top['banco'], df_top['pago'])
        tasa_exito = tabla[1] / tabla.sum(axis=1)
        orden = tasa_exito.sort_values(ascending=False).index
        tabla_pct = tabla.div(tabla.sum(1), axis=0) * 100
        
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        
        tabla_pct.reindex(orden).plot(kind='bar', stacked=True, color=['#555555', '#00D448'], ax=ax, edgecolor='black', width=0.8)
        
        ax.set_title('Tasa de Recuperación por Banco', color='white', fontsize=14)
        ax.set_ylabel('% Recuperación', color='white')
        ax.set_xlabel('')
        ax.tick_params(colors='white', axis='x', rotation=45)
        ax.tick_params(colors='white', axis='y')
        ax.legend(title='Pago', labels=['No', 'Sí'], labelcolor='white', facecolor='black', edgecolor='white')
        
        for spine in ax.spines.values(): spine.set_visible(False)
        st.pyplot(fig)

    # --- TAB 4: DEMOGRÁFICO ---
    with tab4:
        st.subheader("Probabilidad de Pago por Edad")
        
        # Preparar datos para Barplot (Edad es categórica ordinal)
        tasa_edad = df.groupby('rango_edad_probable')['pago'].mean().reset_index()
        tasa_edad['pago_pct'] = tasa_edad['pago'] * 100
        
        # Orden lógico
        order_edad = ['18-25', '26-35', '36-45', '46-55', '56-65', 'Mayor a 65', 'No especificado']
        # Filtrar solo los que existen en los datos
        order_edad = [x for x in order_edad if x in tasa_edad['rango_edad_probable'].unique()]
        
        tasa_edad['rango_edad_probable'] = pd.Categorical(tasa_edad['rango_edad_probable'], categories=order_edad, ordered=True)
        tasa_edad = tasa_edad.sort_values('rango_edad_probable')

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        
        # Barplot en lugar de Lineplot
        sns.barplot(data=tasa_edad, x='rango_edad_probable', y='pago_pct', palette='viridis', ax=ax, edgecolor='black')

        ax.set_title('Tasa de Recuperación por Grupo Etario', color='#00D448', fontsize=16, fontweight='bold')
        ax.set_ylabel('Tasa de Recuperación (%)', color='white')
        ax.set_xlabel('Rango de Edad', color='white')
        ax.tick_params(colors='white')
        
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f%%', padding=3, color='white', fontweight='bold')

        for spine in ax.spines.values(): spine.set_visible(False)
        st.pyplot(fig)

    # --- TAB 5: GESTIÓN ---
    with tab5:
        st.subheader("Duración de Llamadas vs Pago")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        
        sns.boxplot(data=df, x='pago', y='duracion_llamadas_ultimos_6meses', palette=['#555555', '#00D448'], ax=ax, showfliers=False)
        
        ax.set_yscale('log')
        ax.set_title('Impacto de la Duración de Llamadas (Escala Log)', color='white')
        ax.set_xticklabels(['No Pagó', 'Sí Pagó'], color='white')
        ax.set_ylabel('Duración (Segundos)', color='white')
        ax.set_xlabel('')
        ax.tick_params(colors='white')
        
        for spine in ax.spines.values(): spine.set_visible(False)
        st.pyplot(fig)
        
        st.info("💡 **Interpretación:** Una mayor duración en las llamadas (cajas más altas en el grupo 'Sí Pagó') suele correlacionarse positivamente con la recuperación, indicando un contacto efectivo.")
 

elif opcion == "3. Modelado (Próximamente)":
    st.write("Espacio reservado para el modelo de ML.")

elif opcion == "4. SQL (Próximamente)":
    st.write("Espacio reservado para las consultas SQL.")