import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.ticker as ticker
import joblib
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, precision_recall_curve

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
                radial-gradient(circle at 90% 90%, rgba(0, 212, 72, 0.40) 0%, transparent 50%),
                radial-gradient(circle at 20% 5%, rgba(0, 212, 72, 0.20) 0%, transparent 60%);
            
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
        ["1. Introducción & Data", "2. Análisis Exploratorio (EDA)", "3. Modelado & Predicción", "4. SQL (Próximamente)"]
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

        st.error("""
        Observamos un fenómeno interesante en la distribución de clientes activos:
        * **Mes 1 y 2:** Mantenemos un volumen constante de clientes (~200) cuyo último pago fue reciente. Esto indica un comportamiento de pago intermitente pero activo.
        * El volumen de clientes cuyo último pago fue hace 3 meses cae drásticamente a solo **47 personas**.

        **Interpretación de Riesgo:**
        En cobranza, el mes 3 suele ser el punto de inflexión donde el hábito de pago se rompe.
        Quien deja pasar 90 días sin pagar, pierde la costumbre y la prioridad de pago. La gestión debe ser **preventiva antes del Mes 3**. Tratar de reactivar a un cliente que lleva más de 90 días "frío" es exponencialmente más costoso que gestionarlo cuando solo lleva 30 o 60 días.
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

    st.info("""
        La gran barra de valores nulos no es un error de datos, es información y significa que nunca han pagado
        * En lugar de imputar estos valores, el modelo tratará los Nulos como una categoría explícita -1. Esto por que el comportamiento de alguien que *nunca* ha pagado es estructuralmente distinto al de alguien que pagó hace 6 meses. No se deben mezclar en el análisis.
        """)

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
        ### **Variables de Gestión (Operativo):**
        """)

        # 1. EL PROBLEMA DE CAPACIDAD (Los Donas)
        st.error("""
        **1. Diagnóstico de Cobertura: El "Techo" Operativo**
        Los datos revelan una saturación crítica en la capacidad del Call Center:
        * **Capacidad Estática (~11%):** La consistencia casi robótica entre la gestión del Mes Actual (10.8%) y el Anterior (10.9%) indica que la operación ha tocado su techo físico. No importa cuánto crezca la mora, el equipo solo tiene manos para cubrir al 11% de la base.
        """)


    with col2:
        # Definimos las variables
        vars_contacto = ['contacto_mes_actual', 'contacto_mes_anterior', 'contacto_ultimos_6meses']
        var_duracion = 'duracion_llamadas_ultimos_6meses'
        
        # Unimos todas para el loop, pero las trataremos diferente
        variables_gestion = vars_contacto + [var_duracion]

        fig, axes = plt.subplots(2, 2, figsize=(10, 6))
        fig.patch.set_alpha(0.0) # Fondo transparente
        axes = axes.flatten()

        for i, col in enumerate(variables_gestion):
            ax = axes[i]
            ax.patch.set_alpha(0.0)

            # Estilizado del título
            titulo = col.replace('_', ' ').replace('ultimos', 'últ.').title()
            ax.set_title(titulo, color='#00D448', fontsize=12, fontweight='bold')

            # --- Lógica A: Variable Numérica Continua (Duración) ---
            if col == var_duracion:
                # Filtramos solo los mayores a 0
                data_filtrada = df[df[col] > 0][col]
                
                if not data_filtrada.empty:
                    sns.histplot(data_filtrada, color='#00D448', ax=ax, kde=True, bins=20, element="step", alpha=0.5)
                    # Ajustes visuales ejes
                    ax.set_ylabel('Frecuencia', color='white', fontsize=9)
                    ax.set_xlabel('Segundos', color='white', fontsize=9)
                    ax.tick_params(axis='both', colors='white', labelsize=8)
                    for spine in ax.spines.values(): 
                        spine.set_edgecolor('#444444') # Bordes sutiles
                        spine.set_visible(True)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                else:
                    ax.text(0.5, 0.5, "Sin datos > 0", color='white', ha='center')

            # --- Lógica B: Variables Binarias / Conteo (Contactos) ---
            else:
                # Creamos la lógica binaria: ¿Tiene gestión (>0) o no (0)?
                con_gestion = (df[col] == 1).sum()
                sin_gestion = (df[col] == 0).sum()
                
                datos = [con_gestion, sin_gestion]
                etiquetas = ['Con Gestión', 'Sin Gestión']
                colores = ['#00D448', '#2e2e2e'] # Verde brillante vs Gris oscuro
                
                # Gráfica de Dona
                wedges, texts, autotexts = ax.pie(
                    datos, 
                    labels=etiquetas, 
                    colors=colores, 
                    autopct='%1.1f%%', 
                    startangle=90, 
                    pctdistance=0.85, 
                    wedgeprops=dict(width=0.3, edgecolor='none') # width=0.3 hace el agujero
                )
                
                # Estilizar textos de la dona
                for text in texts:
                    text.set_color('white')
                    text.set_fontsize(9)
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                    autotext.set_fontsize(10)

        plt.tight_layout()
        st.pyplot(fig)

    # 3. LA SOLUCIÓN REFINADA (Matriz de Valor)
    st.success("""
    No basta con mirar solo el Saldo (cuánto debe) ni solo el Modelo (qué tan probable es). La estrategia ganadora cruza ambas variables:

    1.  **Prioridad 1: Los "Golden Geese" (Alto Saldo + Alta Probabilidad):**
        * Son el **Foco Absoluto** de los asesores humanos. Clientes con deuda significativa (>$2M) que el modelo marca como recuperables. Aquí está el 80% del dinero real.

    2.  **Prioridad 2: Gestión Digital (Bajo Saldo + Alta Probabilidad):**
        * Clientes que seguramente pagarán, pero deben poco. No gastar tiempo humano costoso; enviar un **Link de Pago por WhatsApp/SMS**. Se recuperan solos.

    3.  **Prioridad 3: Investigación (Alto Saldo + Baja Probabilidad):**
        * Deudores grandes que el modelo ve difíciles. No quemar llamadas; pasarlos a un equipo de **Investigación de Bienes o Cobro Jurídico**.

    *Conclusión:* El modelo no reemplaza la lógica de negocio, la **potencia** para evitar llamar a deudores grandes pero imposibles.
    """)

else:
    
    df.sort_values(by='antiguedad_deuda', na_position='last', inplace=True)

    cols_modelo = [
    'tipo_documento', 'genero', 'rango_edad_probable', 
    'departamento', 'saldo_capital', 'dias_mora', 
    'pago_mes_anterior', 'meses_desde_ultimo_pago', 'sin_pago_previo', 
    'contacto_mes_actual', 'contacto_mes_anterior', 'contacto_ultimos_6meses', 
    'duracion_llamadas_ultimos_6meses', 'pago'
    ]

    # Filtrar columnas que realmente existen
    cols_existentes = [c for c in cols_modelo if c in df.columns]
    df.drop_duplicates(subset=cols_existentes, keep='first', inplace=True)

    df['genero'] = df['genero'].replace({'M': 'HOMBRE', 'F': 'MUJER', ' ': 'No especificado', 'NO APLICA': 'No especificado'})
    df['genero'] = df['genero'].fillna('No especificado')

    mapa_edad = {
    '18-21': '18-25', '18-25': '18-25', '22-25': '18-25',
    '25-30': '26-35', '26-29': '26-35', '30-33': '26-35', '31-35': '26-35', '34-37': '26-35',
    '36-40': '36-45', '38-41': '36-45', '41-45': '36-45', '42-45': '36-45',
    '46-49': '46-55', '46-50': '46-55', '50-53': '46-55', '51-55': '46-55',
    '54-57': '56-65', '56-60': '56-65', '58-61': '56-65', '61-65': '56-65', '62-65': '56-65',
    '66+': 'Mayor a 65', '66-70': 'Mayor a 65', '71-75': 'Mayor a 65', 'Mas de 75': 'Mayor a 65'
    }
    df['rango_edad_probable'] = df['rango_edad_probable'].replace(mapa_edad)
    df['rango_edad_probable'] = df['rango_edad_probable'].replace({'NO APLICA': 'No especificado'}).fillna('No especificado')

    if opcion == "2. Análisis Exploratorio (EDA)":
        st.title("🔍 Análisis Exploratorio de Datos (EDA)")
        st.markdown("Identificación de patrones clave que diferencian a los clientes que pagan de los que no.")

        # Tabs para organizar la historia
        tab1, tab2, tab3, tab4 = st.tabs([
            "Correlación", 
            "Recencia", 
            "Financiero & Bancos", 
            "Demográfico", 
        ])

        # --- TAB 1: CORRELACIONES ---
        with tab1:
            st.subheader("¿Qué variables se relacionan con el pago?")
            st.markdown("""
            Utilizamos la **Correlación de Spearman** porque captura relaciones no lineales y es más robusta a valores atípicos (outliers) que la de Pearson.
            """)

            # Preparar datos numéricos
            df_num = df.drop(columns=["identificacion"]).select_dtypes(include=[np.number]).copy()

            # Matriz de correlación
            fig, ax = plt.subplots(figsize=(10, 4))
            fig.patch.set_alpha(0.0)
            ax.patch.set_alpha(0.0)
            
            corr_matrix = df_num.drop(columns=['pago'], errors='ignore').corr(method='spearman')
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            
            sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='RdBu_r', vmin=-1, vmax=1,
                        cbar_kws={"shrink": .8}, ax=ax)
            
            ax.set_title('Matriz de Correlación (Multicolinealidad)', color='#00D448', fontsize=16, fontweight='bold')
            ax.tick_params(axis='x', colors='white', rotation=90)
            ax.tick_params(axis='y', colors='white')
            
            st.pyplot(fig)

        
            st.markdown("""
            ### Análisis de lo que ves (Impacto en el Target)
            Esta gráfica nos dice qué variables importan, midiendo la fuerza de la relación entre cada $X$ (predictor) y $Y$ (si paga o no).

            *   **Los "Ganadores" (Predictibilidad Positiva):**
                *   `pago_mes_anterior` (0.33): Confirma la regla de oro en riesgo: "El mejor predictor del comportamiento futuro es el comportamiento pasado inmediato". Es la variable estrella.
                *   **Variables de Gestión** (~0.15): `duracion_llamadas`, `contacto_mes_anterior`, etc. Tienen una correlación positiva moderada. Esto valida la operación: gestionar sí aumenta la probabilidad de pago, pero no es determinante por sí sola.

            *   **Los "Destructores" (Predictibilidad Negativa):**
                *   `sin_pago_previo` (-0.35) y `meses_desde_ultimo_pago` (-0.32): Son correlaciones negativas fuertes. Cuanto más tiempo pasa (o si nunca ha pagado), la probabilidad de recuperar el dinero cae en picada.

            *   **La Sorpresa (Irrelevancia):**
                *   `saldo_capital` (0.02): Esto es un insight crucial. El monto de la deuda no predice la intención de pago. Un cliente puede deber $1 millón o $100 millones; su probabilidad de pagar es casi la misma. (Ojo: esto afecta el monto recuperado, pero no la probabilidad binaria).
            """)

            st.info("""
            **Decisión Técnica (Árboles de Decisión):** Aunque los modelos de árboles (XGBoost/Random Forest) son robustos a la multicolinealidad, 
            eliminamos variables con correlación extrema (>0.9) para evitar la dilución del **Feature Importance**.
            """)

            col1, col2 = st.columns(2)

            with col1:
                st.write("""
                Se detectó una correlación casi perfecta (**0.99**) entre:
                1. `duracion_llamadas_ultimos_6meses`
                2. `contacto_ultimos_6meses`
                
                **Diagnóstico:** Ambas miden esencialmente lo mismo (intensidad de gestión histórica). 
                Mantener ambas no aporta información nueva y confunde al modelo sobre cuál es la importante.
                """)

            with col2:
                st.markdown("#### ✂️ Acción Tomada")
                st.success("""
                **Se elimina:** `contacto_ultimos_6meses`
                
                **Se conserva:** `duracion_llamadas_ultimos_6meses`
                
                **Razón:** En el análisis de Spearman previo, la *duración* mostró una correlación ligeramente 
                superior con el target (0.16 vs 0.14). Preferimos la calidad del contacto sobre la cantidad.
                """)

            st.markdown("---")
            st.markdown("#### ⚠️ Nota sobre Recencia (-0.93)")
            st.warning("""
            Existe una fuerte correlación inversa (**-0.93**) entre `meses_desde_ultimo_pago` y `pago_mes_anterior`.
            * Esto es lógico: Si pagó el mes pasado, su recencia es baja.
            * **Decisión:** En este caso **CONSERVAMOS AMBAS**. 
                * `pago_mes_anterior` captura el evento inmediato (Hot Lead).
                * `meses_desde_ultimo_pago` captura la degradación del riesgo a largo plazo. 
                * Al ser árboles, el modelo aprovechará ambos matices.
            """)
                
        # --- TAB 2: RECENCIA ---
        with tab2:

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

                # 1. Los Números Duros (KPIs)
                st.markdown("##### 📊 Tasa de Conversión (Probabilidad de Éxito)")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        label="Clientes CON Historial", 
                        value="37.8%", 
                        delta="Alta Probabilidad",
                        help="De 478 clientes, 181 pagaron."
                    )

                with col2:
                    st.metric(
                        label="Clientes SIN Historial", 
                        value="1.17%", 
                        delta="- Riesgo Extremo",
                        delta_color="inverse",
                        help="De 29,135 clientes, solo 342 pagaron."
                    )

                with col3:
                    st.metric(
                        label="Factor de Multiplicación", 
                        value="32x", 
                        delta="Impacto Predictivo",
                        help="Un cliente con historial es 32 veces más probable de pagar que uno nuevo."
                    )

                # 2. Análisis de Negocio (Estrategia Diferenciada)
                st.info("""
                **🧠 Diagnóstico de Negocio: Dos Mundos Diferentes**
                Esta gráfica demuestra que mezclar clientes "vírgenes" (sin pagos) con clientes "recurrentes" en una misma lista de gestión es un error operativo grave.
                * **La Minería de Oro (Con Historial):** Tienes un grupo pequeño (~500 personas) donde **1 de cada 3 paga**.
                    * *Estrategia:* **Fidelización.** La gestión aquí debe ser de "Mantenimiento". No presionar, sino facilitar. Son tu flujo de caja seguro.
                * **La Búsqueda de Agujas (Sin Historial):** Tienes un océano masivo (~29,000 personas) donde el éxito es una anomalía (**1%**).
                    * *Estrategia:* **Machine Learning Puro.** No es rentable llamar a los 29,000. El modelo debe actuar como un "radar" para encontrar a los pocos que tienen las características de los que sí pagan, y descartar al resto.
                """)


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

            _, col, _ = st.columns([0.1, 0.7, 0.1])
    
            with col:

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

            st.markdown("### 📉 La Regla de Oro de la Recencia: Caducidad del Hábito")

            # 1. Los KPIs del "Acantilado"
            # Usamos columnas para mostrar la caída dramática
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    label="🔥 La Ventana de Oro", 
                    value="53.9%", 
                    delta="Mes 1 (Retención)",
                    help="Probabilidad de pago si el último pago fue hace 1 mes."
                )

            with col2:
                st.metric(
                    label="⚠️ El Acantilado (Drop-off)", 
                    value="14.9%", 
                    delta="-53% vs Mes 2",
                    delta_color="inverse",
                    help="En el Mes 3, la probabilidad cae a la mitad respecto al Mes 2 (31.9%)."
                )

            with col3:
                st.metric(
                    label="☠️ El Cementerio", 
                    value="0.0%", 
                    delta="Mes 5 en adelante",
                    help="Probabilidad de recuperación estadística nula después del 5to mes."
                )

            # 2. Estrategia Operativa (La Tabla de Acción)
            st.info("""
            **🧠 Estrategia de Gestión Basada en Datos (Data-Driven Strategy)**
            Los datos dictan una política de segmentación estricta para maximizar el ROI de las llamadas:

            | Perfil de Recencia | Antigüedad | Probabilidad | 📞 Acción Recomendada (Canal) |
            | :--- | :--- | :--- | :--- |
            | **HOT (Prioridad)** | 1 - 2 Meses | **32% - 54%** | **Llamada Humana Intensiva.** El hábito sigue vivo. Aquí se recupera el dinero. |
            | **RISK (Alerta)** | 3 - 4 Meses | **11% - 15%** | **Gestión Híbrida.** WhatsApp/SMS primero. Llamada humana solo si hay respuesta o saldo muy alto. |
            | **LOST (Castigo)** | 5+ Meses | **0%** | **Automatización Total.** No gastar tiempo de asesores. Enviar campañas masivas de Email/SMS. El costo de llamar supera el retorno esperado. |
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


            st.markdown("### 💰 Perfil Financiero: ¿Quiénes son los que pagan?")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Días de Mora (Factor Crítico)")
                st.info("""
                **Hallazgo:** Existe una clara distinción en la antigüedad.
                * Los clientes que pagan tienen una mediana de mora de **~800 días**.
                * Los que no pagan se sitúan sobre los **1,500 días**.
                
                **Estrategia:** La "frescura" de la deuda es un predictor fuerte de éxito. 
                Las deudas de más de 4 años (>1500 días) entran en una zona de muy difícil recuperación.
                """)

            with col2:
                st.markdown("#### Saldo Capital (Factor Neutro)")
                st.warning("""
                **Hallazgo:** Las distribuciones son casi idénticas.
                * El tamaño de la deuda **NO discrimina** entre pagadores y no pagadores.
                * Curiosamente, la mediana de los que pagan es ligeramente *superior*.
                
                **Estrategia:** No discriminar ni priorizar negativamente los montos altos. 
                Un cliente con deuda alta tiene la misma voluntad de pago que uno pequeño.
                """)

            st.divider()


            st.subheader("Calidad de Cartera por Banco")
            
            # Lógica para gráfico apilado de bancos
            top_bancos = df['banco'].value_counts().index[:10]
            df_top = df[df['banco'].isin(top_bancos)].copy()
            tabla = pd.crosstab(df_top['banco'], df_top['pago'])
            tasa_exito = tabla[1] / tabla.sum(axis=1)
            orden = tasa_exito.sort_values(ascending=False).index
            tabla_pct = tabla.div(tabla.sum(1), axis=0) * 100
            
            fig, ax = plt.subplots(figsize=(12, 4))
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
        

            st.warning("""
            **Variable de Bajo Impacto.**
            Al comparar las tasas de recuperación entre entidades, observamos un comportamiento **uniforme**.
            * La probabilidad de pago es prácticamente idéntica (~1%) sin importar si la deuda proviene de Davivienda, Citibank o BBVA.
            * **Implicación para el Modelo:** Es probable que esta variable tenga una **importancia baja (Low Feature Importance)** en el árbol de decisión, ya que no permite separar claramente a los pagadores de los deudores.
            """)

        with tab4:
            st.subheader("Perfil de Riesgo Demográfico")
            
            # Función auxiliar para graficar al estilo "Dark Mode"
            def plot_stacked_dark(df, col, titulo, palette=['#444444', '#00D448']):
                # 1. Preparar datos (Crosstab normalizado)
                cross = pd.crosstab(df[col], df['pago'], normalize='index') * 100
                
                # Ordenar por la tasa de pago (columna 1) para dar efecto de "ranking"
                cross = cross.sort_values(by=1, ascending=True)
                
                # 2. Configurar Plot
                fig, ax = plt.subplots(figsize=(10, 6))
                fig.patch.set_alpha(0.0) # Fondo transparente
                ax.patch.set_alpha(0.0)
                
                # 3. Graficar Apilado
                cross.plot(kind='barh', stacked=True, color=palette, ax=ax, edgecolor='black', width=0.7)
                
                # 4. Estilizado Dark
                ax.set_title(titulo, color='#00D448', fontsize=14, fontweight='bold')
                ax.set_xlabel('Proporción (%)', color='white')
                ax.set_ylabel('')
                ax.tick_params(colors='white', which='both')
                ax.legend(labels=['No Pagó', 'Sí Pagó'], loc='upper center', bbox_to_anchor=(0.5, -0.1), 
                        ncol=2, frameon=False, labelcolor='white')
                
                # Quitar bordes feos
                for spine in ax.spines.values(): spine.set_visible(False)
                
                # 5. Etiquetas de Datos (Solo en la parte verde para no saturar)
                for n, container in enumerate(ax.containers):
                    # Solo etiquetamos la serie 1 (Los que Pagan - Verde)
                    if n == 1: 
                        labels = [f'{v.get_width():.1f}%' if v.get_width() > 0 else '' for v in container]
                        ax.bar_label(container, labels=labels, label_type='center', 
                                    color='white', fontweight='bold', fontsize=10)
                
                return fig

            col1, col2 = st.columns(2)

            # GRÁFICA 1: EDAD
            with col1:
                # Orden lógico para edad (no por valor, sino por etapa de vida)
                orden_edad = ['18-25', '26-35', '36-45', '46-55', '56-65', 'Mayor a 65', 'No especificado']
                # Aseguramos que sea categórica ordenada
                df['rango_edad_probable'] = pd.Categorical(
                    df['rango_edad_probable'], 
                    categories=[x for x in orden_edad if x in df['rango_edad_probable'].unique()], 
                    ordered=True
                )
                
                fig_edad = plot_stacked_dark(df, 'rango_edad_probable', 'Probabilidad de Pago por Edad')
                st.pyplot(fig_edad)

            # GRÁFICA 2: GÉNERO
            with col2:
                # Limpieza rápida para agrupar vacíos
                df['genero_plot'] = df['genero'].replace({' ': 'NO ESPECIFICADO', 'NO APLICA': 'NO ESPECIFICADO'}).fillna('NO ESPECIFICADO')
                
                fig_gen = plot_stacked_dark(df, 'genero_plot', 'Probabilidad de Pago por Género')
                st.pyplot(fig_gen)

            # --- INSIGHTS DE NEGOCIO ---
            st.markdown("---")
            st.info("""
            **🧠 Lectura de las Gráficas (Stacked Bars):**
            * **La Barra Verde:** Representa tu **Tasa de Recuperación Real**. Cuanto más grande sea el segmento verde, mejor es ese grupo.
            * **Edad:** Visualmente confirmarás si la barra verde crece con la edad (tendencia típica: a mayor edad, mayor responsabilidad financiera).
            * **Género:** Te permite ver de un vistazo qué género es más rentable, ignorando el hecho de que tengas más hombres o mujeres en total.
            """)    

    elif opcion == "3. Modelado & Predicción":
        st.title("🤖 Modelado & Predicción")
        st.markdown("Predicción de probabilidad de pago y detección de anomalías utilizando los modelos entrenados.")

        # 1. Cargar artefactos
        try:
            artifacts = joblib.load('modelos_riesgo_v1.pkl')
        except FileNotFoundError:
            st.error("⚠️ Archivo 'modelos_riesgo_v1.pkl' no encontrado. Asegúrate de haber ejecutado el notebook de entrenamiento.")
            st.stop()
            
        best_tree = artifacts["arbol"]
        autoencoder = artifacts["autoencoder"]
        preprocessor = artifacts["preprocessor"]
        best_threshold = artifacts["umbral_autoencoder"]
        model_cols = artifacts["columnas_modelo"]
        
        # 2. Preparar el DF (Mismas reglas que en entrenamiento)
        df_pred = df.copy()
        
        # Filtros de negocio (replicar lo del notebook)
        df_pred = df_pred[df_pred['dias_mora'] < 3650]
        df_pred = df_pred[df_pred['saldo_capital'] > 1000]
        
        # Tratamiento de nulos
        if 'meses_desde_ultimo_pago' in df_pred.columns:
            df_pred['meses_desde_ultimo_pago'] = df_pred['meses_desde_ultimo_pago'].fillna(-1)
            
        # Validar columnas
        missing_cols = [c for c in model_cols if c not in df_pred.columns]
        if missing_cols:
            st.error(f"Faltan columnas para el modelo: {missing_cols}")
            st.stop()
            
        # Seleccionar solo las columnas del modelo para X
        X_input = df_pred[model_cols]
        
        # 3. Transformar
        try:
            X_processed = preprocessor.transform(X_input)
        except Exception as e:
            st.error(f"Error en preprocesamiento: {e}")
            st.stop()
            
        # 4. Predecir
        # Probabilidad de pago (Clase 1)
        probs = best_tree.predict_proba(X_processed)[:, 1]
        
        # Score de anomalía
        reconstruccion = autoencoder.predict(X_processed)
        mse = np.mean(np.power(X_processed - reconstruccion, 2), axis=1)
        
        # 5. Resultados
        df_pred['probabilidad_pago_arbol'] = probs
        df_pred['score_anomalia_autoencoder'] = mse
        df_pred['alerta_anomalia'] = mse > best_threshold
        
        # --- Dashboard de Resultados ---
        
        # KPIs
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Clientes Evaluados", len(df_pred))
        col2.metric("Prob. Pago Promedio", f"{probs.mean():.1%}")
        col3.metric("Clientes 'Pagadores' (>50%)", (probs > 0.5).sum())
        col4.metric("Anomalías Detectadas", df_pred['alerta_anomalia'].sum())
            
        # --- SECCIÓN DETALLADA DE MODELOS ---
        if 'pago' in df_pred.columns:
            st.markdown("---")
            st.header("🧠 Detalles y Evaluación de Modelos")
            
            y_true = df_pred['pago']
            
            st.subheader("1. Construcción del Modelo")
            st.markdown("El modelo fue optimizado usando `GridSearchCV` para maximizar el **F1-Score** de la clase minoritaria (Pagadores).")
            st.code("""
param_grid = {
    'max_depth': [3, 4, 5, 6],
    'min_samples_leaf': [30, 50, 100],
    'criterion': ['gini', 'entropy'],
    'class_weight': ['balanced']  # Clave para el desbalance
}
grid_search = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=param_grid, scoring='f1', ...)
                """, language='python')
                
            st.subheader("2. Rendimiento (Matriz de Confusión)")
            y_pred_tree = (probs > 0.5).astype(int)
                
            col1, col2 = st.columns(2)
            with col1:
                    fig_cm, ax = plt.subplots()
                    fig_cm.patch.set_alpha(0.0)
                    ax.patch.set_alpha(0.0)
                    ConfusionMatrixDisplay.from_predictions(y_true, y_pred_tree, ax=ax, cmap='Greens', colorbar=False)
                    
                    # Estilizado
                    ax.set_title("Matriz de Confusión (Árbol)", color='white')
                    ax.tick_params(colors='white')
                    ax.xaxis.label.set_color('white')
                    ax.yaxis.label.set_color('white')
                    
                    st.pyplot(fig_cm)
                
            with col2:
                    st.markdown("**Métricas Detalladas**")
                    report = classification_report(y_true, y_pred_tree, output_dict=True)
                    st.dataframe(pd.DataFrame(report).transpose().style.format("{:.2f}"))

            
            st.subheader("1. Lógica del Autoencoder")
            st.markdown("Red neuronal entrenada **solo con No Pagadores** para aprender el patrón de 'normalidad'. Los Pagadores se detectan como anomalías (alto error de reconstrucción).")
            st.code("""
autoencoder = MLPRegressor(
    hidden_layer_sizes=(64, 32, 4, 32, 64), 
    activation='relu', solver='adam', 
    alpha=1e-7, max_iter=200
)
# Entrenamiento: autoencoder.fit(X_no_paga, X_no_paga)
                """, language='python')
                
            st.subheader("2. Curva de Umbral (Precision-Recall)")
                
                # Cálculos para la curva
            precision, recall, thresholds = precision_recall_curve(y_true, mse)
            numerator = 2 * recall * precision
            denominator = recall + precision
            f1_scores = np.divide(numerator, denominator, out=np.zeros_like(denominator), where=denominator!=0)

            fig_thresh, ax_thresh = plt.subplots(figsize=(10, 4))
            fig_thresh.patch.set_alpha(0.0)
            ax_thresh.patch.set_alpha(0.0)

            ax_thresh.plot(thresholds, precision[:-1], 'b--', label='Precision', alpha=0.7)
            ax_thresh.plot(thresholds, recall[:-1], 'g-', label='Recall', alpha=0.7)
            ax_thresh.plot(thresholds, f1_scores[:-1], 'r-', label='F1 Score', linewidth=2)
            ax_thresh.axvline(best_threshold, color='#00D448', linestyle=':', label=f'Umbral Óptimo ({best_threshold:.4f})')

            ax_thresh.set_title('Optimización del Punto de Corte', color='white')
            ax_thresh.set_xlabel('Error de Reconstrucción (MSE)', color='white')
            ax_thresh.tick_params(colors='white')
            ax_thresh.legend(facecolor='black', labelcolor='white')

            st.pyplot(fig_thresh)

            st.subheader("3. Rendimiento con Umbral Óptimo")
            y_pred_ae = (mse > best_threshold).astype(int)

            # Reporte simple
            col1, col2 = st.columns(2)
            with col1:
                    fig_cm, ax = plt.subplots()
                    fig_cm.patch.set_alpha(0.0)
                    ax.patch.set_alpha(0.0)
                    ConfusionMatrixDisplay.from_predictions(y_true, y_pred_ae, ax=ax, cmap='Greens', colorbar=False)
                    
                    # Estilizado
                    ax.set_title("Matriz de Confusión (Árbol)", color='white')
                    ax.tick_params(colors='white')
                    ax.xaxis.label.set_color('white')
                    ax.yaxis.label.set_color('white')
                    
                    st.pyplot(fig_cm)
                
            with col2:
                report_ae = classification_report(y_true, y_pred_ae, output_dict=True)
                st.dataframe(pd.DataFrame(report_ae).transpose().style.format("{:.2f}"))

                st.caption(f"Umbral aplicado: {best_threshold:.6f}")

                st.markdown("---")
        
        st.subheader("🔥 Top Clientes con Mayor Probabilidad de Pago")
        st.markdown("Estos son los clientes a los que deberías llamar **YA**.")
        
        top_clients = df_pred.sort_values('probabilidad_pago_arbol', ascending=False).head(20)
        
        # Formateo
        format_dict = {
            'probabilidad_pago_arbol': '{:.1%}',
            'score_anomalia_autoencoder': '{:.4f}',
            'saldo_capital': '${:,.0f}',
            'dias_mora': '{:.0f}'
        }
        
        cols_visual = ['identificacion', 'saldo_capital', 'dias_mora', 'probabilidad_pago_arbol', 'score_anomalia_autoencoder']
        # Filtrar si alguna no existe
        cols_visual = [c for c in cols_visual if c in df_pred.columns]
        
        st.dataframe(top_clients[cols_visual].style.format(format_dict).background_gradient(subset=['probabilidad_pago_arbol'], cmap='Greens'))



    elif opcion == "4. SQL (Próximamente)":

        st.subheader("💻 Consultas SQL en Vivo")
        st.markdown("Este módulo permite ejecutar sentencias **SQL estándar** directamente sobre el DataFrame de Pandas.")
        
        # Importación necesaria (Asegúrate de tener instalada: pip install pandasql)
        from pandasql import sqldf
        
        env = {'df': df} 
        pysqldf = lambda q: sqldf(q, env)
        col1, col2 = st.columns(2)

        # --- CONSULTA 1: TOP 10 ---
        with col1:
            st.markdown("#### 🏆 Top 10 Mayores Deudores")
            st.info("Identificando a los clientes con mayor exposición de Saldo Capital.")
            
            query_top10 = """
            SELECT tipo_documento, identificacion, saldo_capital
            FROM df
            ORDER BY saldo_capital DESC
            LIMIT 10;
            """
            
            # Mostramos el código SQL para que se vea técnico
            st.code(query_top10, language='sql')
            
            try:
                resultado_top10 = pysqldf(query_top10)
                # Mostramos tabla formateada
                st.dataframe(
                    resultado_top10.style.format({'saldo_capital': '${:,.0f}'}), 
                    use_container_width=True,
                    hide_index=True
                )
            except Exception as e:
                st.error(f"Error en SQL: {e}")

        # --- CONSULTA 2: PROMEDIO POR DEPARTAMENTO ---
        with col2:
            st.markdown("#### 🗺️ Efectividad por Departamento")
            st.info("Calculando la Tasa de Pago (Promedio) agrupada geográficamente.")
            
            query_promedio = """
            SELECT departamento, AVG(pago) as tasa_pago
            FROM df
            GROUP BY departamento
            ORDER BY tasa_pago DESC;
            """
            
            st.code(query_promedio, language='sql')
            
            try:
                resultado_promedio = pysqldf(query_promedio)
                
                # Multiplicamos por 100 para que se vea como porcentaje
                resultado_promedio['tasa_pago'] = resultado_promedio['tasa_pago'] 
                
                # Usamos un gradiente de color (Heatmap) para resaltar los mejores
                st.dataframe(
                    resultado_promedio.style
                    .format({'tasa_pago': '{:.2%}'}) # Formato porcentaje
                    .background_gradient(cmap='Greens', subset=['tasa_pago']), # Mapa de calor verde
                    use_container_width=True,
                    hide_index=True
                )
            except Exception as e:
                st.error(f"Error en SQL: {e}")

