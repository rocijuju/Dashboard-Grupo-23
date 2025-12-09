import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

#"C:\Users\rpain\OneDrive\Documents\MODELACION BASE DE DATOS"

st.set_page_config(layout="wide")

# 1. CARGA DE DATOS (Simulada para el ejemplo, usa tu pd.read_csv real)
df=pd.read_csv('df_total_matricula.csv')
df_t=pd.read_csv('df_total_titulados.csv')

# --- BARRA LATERAL (Menú Oscuro) ---
with st.sidebar:
    selected = option_menu(
        menu_title="Menú",
        options=["Inicio", "Matrícula", "Titulación", "Duración de Carrera","Motivación"],
        icons=["house", "graph-up-arrow", "mortarboard", "hourglass-split", "lightbulb"], 
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"background-color": "#262730"},
            "nav-link": {"color": "white", "--hover-color": "#333333"},
            "nav-link-selected": {"background-color": "#0099ff"}
        }
    )

# ==========================================
# PÁGINA: INICIO
# ==========================================
if selected == "Inicio":
    st.title("🏠 Sistema de Alerta de Persistencia Estudiantil con Enfoque en la Brecha de Género")
    st.markdown("""
    **Bienvenido a la plataforma de visualización de datos académicos.**
    
    Esta herramienta interactiva ha sido diseñada para analizar el comportamiento estudiantil a través de cuatro dimensiones clave, permitiendo identificar patrones, brechas de género y áreas de riesgo en la trayectoria académica.

    ### 🔍 ¿Qué encontrarás en este Dashboard?

    * **📈 Análisis de Matrícula:** Evolución temporal de la admisión de estudiantes, con un enfoque específico en la participación femenina por carrera y comparativas anuales.
    * **🎓 Tasas de Titulación:** Visualización de la cantidad de titulados por período y género, permitiendo contrastar el egreso efectivo con el ingreso.
    * **⏳ Duración Real de Carrera:** Comparación entre la duración formal y el tiempo real de titulación mediante **gráficos de violín**, desglosado por género para detectar disparidades en el tiempo de permanencia.
    * **💡 Motivación y Abandono:** Un análisis crítico que cruza niveles de motivación con la intención de abandono (Heatmaps), identificando grupos de estudiantes en riesgo académico.

    ---
    👈 **Utiliza el menú lateral** para navegar entre los módulos y aplicar filtros por carrera o género.
    """)
  

# ==========================================
# PÁGINA: MATRÍCULA
# ==========================================

elif selected == "Matrícula":
    st.title("📈 Análisis de Matrícula")

    # --- FILTRO ESPECÍFICO ---
    st.sidebar.markdown("---")
    st.sidebar.header("Filtros Matrícula")
    
    lista_opciones = ["Todas las carreras"] + list(df['nomb_carrera'].unique())
    carrera_adm = st.sidebar.selectbox(
        "Selecciona Carrera:", 
        options=lista_opciones, 
        key="filtro_admision"
    )

    # --- LÓGICA DEL FILTRO ---
    
    # === OPCIÓN A: TODAS LAS CARRERAS (Comparativa) ===
    if carrera_adm == "Todas las carreras":
        st.subheader("📊 Comparativa de Carreras ")

        # 1. FILTRAR AÑOS
        anos_clave = [2013, 2019, 2025]
        df_comparativo = df[df['cat_periodo'].isin(anos_clave)]

        # 2. CALCULAR DATOS
        tabla_comp = pd.crosstab(
            index=[df_comparativo['cat_periodo'], df_comparativo['nomb_carrera']], 
            columns=df_comparativo['gen_alu']
        )
        
        # Cálculos de seguridad
        tabla_comp = tabla_comp.fillna(0) # Rellenar vacíos con 0
        tabla_comp['Total'] = tabla_comp.sum(axis=1)
        

        col_mujer = 'Femenino' # Ajusta si en tu excel se llama diferente
        if col_mujer in tabla_comp.columns:
             tabla_comp['% Mujeres'] = (tabla_comp[col_mujer] / tabla_comp['Total']) * 100
        else:
             tabla_comp['% Mujeres'] = 0 

        # 3. GRAFICAR
        df_grafico_comp = tabla_comp.reset_index()

        fig_comp = px.line(
            df_grafico_comp, 
            x='cat_periodo', 
            y='% Mujeres', 
            color='nomb_carrera', 
            markers=True,
            title="Evolución % Mujeres: Comparativa de Carreras",height=500,
            labels={'nomb_carrera': 'Carrera', 'cat_periodo': 'Año'}
        )
        
        fig_comp.update_layout(
            yaxis_range=[0, 100], 
            yaxis_title="Participación Femenina (%)",
            xaxis=dict(tickmode='array', tickvals=anos_clave)
        )
        fig_comp.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.5)
        
        st.plotly_chart(fig_comp, use_container_width=True)
        st.image("heatmapmatricula.png", caption="Heatmap evolución Matrícula", width=800)

    # === OPCIÓN B: UNA SOLA CARRERA (Detalle) ===
    else:
        # 1. FILTRAR
        df_admision = df[df['nomb_carrera'] == carrera_adm]
        
        # 2. CALCULAR TABLA Y PORCENTAJES
        tabla = pd.crosstab(df_admision['cat_periodo'], df_admision['gen_alu'])
        tabla = tabla.fillna(0)
        
        tabla.index.name = "Año"
        tabla['Total Alumnos'] = tabla.sum(axis=1)
        
        # Ajusta 'Femenino' si tu columna se llama diferente
        if 'Femenino' in tabla.columns:
            tabla['% Mujeres'] = (tabla['Femenino'] / tabla['Total Alumnos']) * 100
        else:
            tabla['% Mujeres'] = 0

        # 3. GRAFICAR (SOLO UNA VEZ AQUÍ)
        st.subheader(f"Evolución: {carrera_adm}")
        
        df_grafico = tabla.reset_index()
        fig_evolucion = px.line(
            df_grafico, 
            x='Año', 
            y='% Mujeres', 
            markers=True,
            title=f"Evolución del % de Mujeres en {carrera_adm}"
        )
        fig_evolucion.update_traces(line_color='#FF6692', line_width=3)
        fig_evolucion.update_layout(yaxis_range=[0, 100], yaxis_title="Porcentaje (%)",xaxis_title="Año")
        
        st.plotly_chart(fig_evolucion, use_container_width=True) 

        # 4. MOSTRAR TABLA
        st.subheader("📋 Tabla de Detalles por Año")
        columnas_ordenadas = ['Femenino', 'Masculino', 'Total Alumnos', '% Mujeres']
        
        # Filtramos solo las columnas que existan para que no de error
        cols_finales = [c for c in columnas_ordenadas if c in tabla.columns]
        
        st.dataframe(
            tabla[cols_finales].style.format({
                '% Mujeres': '{:.1f}%',
                'Total Alumnos': '{:.0f}',
                'Femenino': '{:.0f}',
                'Masculino': '{:.0f}'
            }),
            use_container_width=True
        )


# ==========================================
# PÁGINA: TITULACIÓN
# ==========================================
elif selected == "Titulación": # <--- Asegúrate que coincida con tu menú
    st.title("🎓 Análisis de Titulación")

    # -----------------------------------------------------------------
    # ⚠️ NOTA IMPORTANTE: 
    # Si tus titulados están mezclados con los matriculados, filtra aquí.
    # Ejemplo: df_titulados = df[df['situacion_academica'] == 'Titulado']
    # Si 'df' ya son solo titulados, deja la siguiente línea tal cual:
    df_titulados = df_t
    # -----------------------------------------------------------------

    # --- FILTRO ESPECÍFICO ---
    st.sidebar.markdown("---")
    st.sidebar.header("Filtros Titulación")
    
    lista_opciones = ["Todas las carreras"] + list(df_titulados['nomb_carrera'].unique())
    
    # IMPORTANTE: Cambiamos el 'key' para que no choque con el de admisión
    carrera_tit = st.sidebar.selectbox(
        "Selecciona Carrera:", 
        options=lista_opciones, 
        key="filtro_titulacion" 
    )

    # --- LÓGICA DEL FILTRO ---
    
    # === OPCIÓN A: TODAS LAS CARRERAS (Comparativa) ===
    if carrera_tit == "Todas las carreras":
        st.subheader("📊 Comparativa de Titulación ")

        # 1. FILTRAR AÑOS CLAVE
        anos_clave = [2018, 2019,2020,2021,2022,2023,2024]# Puedes cambiar estos años si quieres
        df_comparativo = df_titulados[df_titulados['cat_periodo'].isin(anos_clave)]

        # 2. CALCULAR DATOS
        tabla_comp = pd.crosstab(
            index=[df_comparativo['cat_periodo'], df_comparativo['nomb_carrera']], 
            columns=df_comparativo['gen_alu']
        )
        
        # Cálculos de seguridad
        tabla_comp = tabla_comp.fillna(0) 
        tabla_comp['Total'] = tabla_comp.sum(axis=1)
        
        col_mujer = 'Femenino' # Ajusta si en tu excel se llama diferente
        if col_mujer in tabla_comp.columns:
             tabla_comp['% Mujeres'] = (tabla_comp[col_mujer] / tabla_comp['Total']) * 100
        else:
             tabla_comp['% Mujeres'] = 0 

        # 3. GRAFICAR
        df_grafico_comp = tabla_comp.reset_index()

        fig_comp = px.line(
            df_grafico_comp, 
            x='cat_periodo', 
            y='% Mujeres', 
            color='nomb_carrera', 
            markers=True,
            title="Evolución % Mujeres Tituladas: Comparativa",height=500,
            labels={'nomb_carrera': 'Carrera', 'cat_periodo': 'Año'}
        )

        fig_comp.update_layout(
            yaxis_range=[0, 100], 
            yaxis_title="Participación Femenina (%)",
            xaxis=dict(tickmode='array', tickvals=anos_clave)
        )
        fig_comp.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.5)

        st.plotly_chart(fig_comp, use_container_width=True)
        st.image("heatmaptitulacion.png", width=800)

    # === OPCIÓN B: UNA SOLA CARRERA (Detalle) ===
    else:
        # 1. FILTRAR POR CARRERA
        df_carrera_tit = df_titulados[df_titulados['nomb_carrera'] == carrera_tit]
        
        # 2. CALCULAR TABLA Y PORCENTAJES
        tabla = pd.crosstab(df_carrera_tit['cat_periodo'], df_carrera_tit['gen_alu'])
        tabla = tabla.fillna(0)
        tabla.index.name = "Año"
        tabla['Total Titulados'] = tabla.sum(axis=1)
        
        if 'Femenino' in tabla.columns:
            tabla['% Mujeres'] = (tabla['Femenino'] / tabla['Total Titulados']) * 100
        else:
            tabla['% Mujeres'] = 0

        # 3. GRAFICAR (SOLO UNA VEZ AQUÍ)
        st.subheader(f"Evolución Titulación: {carrera_tit}")
        
        df_grafico = tabla.reset_index()
        fig_evolucion = px.line(
            df_grafico, 
            x='Año', 
            y='% Mujeres', 
            markers=True,
            title=f"% Mujeres Tituladas en {carrera_tit}"
        )
        # Usamos otro color (ej. Azul o Morado) para diferenciar visualmente de admisión
        fig_evolucion.update_traces(line_color='#636EFA', line_width=3)
        fig_evolucion.update_layout(yaxis_range=[0, 100], yaxis_title="Porcentaje (%)",xaxis_title="Año")
        
        st.plotly_chart(fig_evolucion, use_container_width=True) 

        # 4. MOSTRAR TABLA
        st.subheader("📋 Tabla de Titulados por Año")
        columnas_ordenadas = ['Femenino', 'Masculino', 'Total Titulados', '% Mujeres']
        
        cols_finales = [c for c in columnas_ordenadas if c in tabla.columns]

        st.dataframe(
            tabla[cols_finales].style.format({
                '% Mujeres': '{:.1f}%',
                'Total Titulados': '{:.0f}',
                'Femenino': '{:.0f}',
                'Masculino': '{:.0f}'
            }),
            use_container_width=True
        )

        # ==========================================
# PÁGINA: DURACIÓN DE CARRERA
# ==========================================
elif selected == "Duración de Carrera":
    st.title("⏳ Duración Real de la Carrera")
    st.markdown("Comparación entre la duración formal y el tiempo real que tardan los estudiantes en titularse.")

    # 1. PREPARACIÓN DE DATOS (Cálculo de Duración)
    # Hacemos una copia para no alterar el original
    # IMPORTANTE: Asegúrate de que 'df' tenga datos de titulados. 
    # Si tienes un archivo aparte para titulados, cárgalo aquí: df_duracion = pd.read_csv('titulados.csv')
    df_duracion = df_t.copy() 
    
    # Calculamos la duración (Año Titulación - Año Ingreso)
    # Ajusta los nombres de columnas si son distintos en tu excel
    if 'anio_ing_carr_act' in df_duracion.columns:
        df_duracion['Duracion_Real'] = df_duracion['cat_periodo'] - df_duracion['anio_ing_carr_act']
        
        # Filtramos datos erróneos (negativos o mayores a 15 años)
        df_limpio = df_duracion[(df_duracion['Duracion_Real'] > 0) & (df_duracion['Duracion_Real'] <= 15)]
    else:
        st.error("Falta la columna 'anio_ing_carr_act' para calcular la duración.")
        df_limpio = pd.DataFrame()

    # 2. FILTROS EN SIDEBAR
    st.sidebar.markdown("---")
    st.sidebar.header("Filtros Duración")
    
    lista_duracion = ["Todas las carreras"] + list(df_limpio['nomb_carrera'].unique())
    
    carrera_dur = st.sidebar.selectbox(
        "Selecciona Carrera:", 
        options=lista_duracion, 
        key="filtro_duracion" # Key única obligatoria
    )

    # 3. LÓGICA DEL GRÁFICO (Todas vs Individual)
    if not df_limpio.empty:
        
        # --- A. CONFIGURACIÓN SEGÚN SELECCIÓN ---
        if carrera_dur == "Todas las carreras":
            data_plot = df_limpio
            alto_grafico = 14 # Muy alto para que quepan todas
            titulo_grafico = "Distribución de Duración: Todas las Carreras"
        else:
            data_plot = df_limpio[df_limpio['nomb_carrera'] == carrera_dur]
            alto_grafico = 6  # Más bajo si es solo una
            titulo_grafico = f"Distribución de Duración: {carrera_dur}"

        # --- B. CREACIÓN DEL GRÁFICO (SEABORN) ---
        # Creamos la figura explícitamente para pasarla a Streamlit
        fig, ax = plt.subplots(figsize=(10, alto_grafico))
        
        # Dibujamos el Violín
        sns.violinplot(
            data=data_plot,
            y="nomb_carrera",       # Eje Y: Carreras
            x="Duracion_Real",      # Eje X: Años
            hue="gen_alu",          # Color por género
            split=True,             # Parte el violín en dos (Hombre/Mujer)
            inner="quart",          # Líneas de cuartiles
            palette={'Femenino': '#E74C3C', 'Masculino': '#3498DB'}, # Rojo y Azul
            gap=0.1,
            linewidth=1,
            ax=ax                   # Importante: dibujar sobre el eje que creamos
        )

        # Referencias visuales
        ax.axvline(x=6, color='green', linestyle='--', alpha=0.5, label='Duración Nominal (6 años)')
        ax.set_title(titulo_grafico, fontsize=15)
        ax.set_xlabel('Años de Duración Real', fontsize=12)
        ax.set_ylabel('')
        ax.legend(title='Género', loc='lower right')
        ax.grid(axis='x', linestyle='--', alpha=0.3)
        
        # Ajustar límites X para que se vea ordenado (ej: de 0 a 15 años)
        ax.set_xlim(0, 16)

        # --- C. MOSTRAR EN STREAMLIT ---
        # Usamos st.pyplot en lugar de plt.show()
        st.pyplot(fig)
        
        # --- D. METRICAS RÁPIDAS (Opcional) ---
        st.markdown("---")
        st.subheader("📊Promedios de duración por género")

        # 1. Creamos las columnas
        col1, col2, col3 = st.columns(3)

        # 2. Calculamos y mostramos directamente
        # Promedio General
        prom_general = data_plot['Duracion_Real'].mean()
        col1.metric("General", f"{prom_general:.1f} años")

        # Promedio Mujeres (Asegúrate que la columna sea 'Sexo' y el valor 'Mujer')
        prom_mujeres = data_plot[data_plot['gen_alu'] == 'Femenino']['Duracion_Real'].mean()
        col2.metric("Femenino", f"{prom_mujeres:.1f} años")

        # Promedio Hombres (Asegúrate que la columna sea 'Sexo' y el valor 'Hombre')
        prom_hombres = data_plot[data_plot['gen_alu'] == 'Masculino']['Duracion_Real'].mean()
        col3.metric("Masculino", f"{prom_hombres:.1f} años")

    else:
        st.warning("No hay datos suficientes para calcular la duración.")
        # ==========================================
# PÁGINA: MOTIVACIÓN
# ==========================================

elif selected == "Motivación":
    st.title("💡 Motivación vs. Intención de Abandono")
    st.markdown("Comparativa de género: ¿Cómo se relacionan la motivación y el riesgo de abandono?")

    # 1. CARGA DE DATOS (AMBOS ARCHIVOS)
    @st.cache_data
    def load_data_motivacion():
        df_m = pd.read_csv('df_nuevomujer.csv')
        df_h = pd.read_csv('df_nuevohombre.csv')
        return df_m, df_h

    df_mujer, df_hombre = load_data_motivacion()

    # 2. FILTROS LATERALES
    st.sidebar.markdown("---")
    st.sidebar.header("Filtros Motivación")

    # Unimos las carreras de ambos archivos para que no falte ninguna en la lista
    carreras_m = set(df_mujer['Carrera que estudias actualmente'].unique())
    carreras_h = set(df_hombre['Carrera que estudias actualmente'].unique())
    lista_completa = ["Todas las carreras"] + sorted(list(carreras_m.union(carreras_h)))

    carrera_mot = st.sidebar.selectbox(
        "Selecciona Carrera:", 
        options=lista_completa,
        key="filtro_motivacion"
    )

    # 3. LÓGICA DE FILTRADO
    if carrera_mot == "Todas las carreras":
        df_filt_m = df_mujer
        df_filt_h = df_hombre
        titulo_extra = "Todas las Carreras"
    else:
        df_filt_m = df_mujer[df_mujer['Carrera que estudias actualmente'] == carrera_mot]
        df_filt_h = df_hombre[df_hombre['Carrera que estudias actualmente'] == carrera_mot]
        titulo_extra = carrera_mot

    st.subheader(f"Vista: {titulo_extra}")

    # 4. GENERACIÓN DE GRÁFICOS (EN DOS COLUMNAS)
    col1, col2 = st.columns(2)

    # Definimos niveles fijos para que ambos gráficos sean idénticos (1-5)
    niveles_fijos = [1, 2, 3, 4, 5]

    # --- FUNCIÓN AUXILIAR PARA DIBUJAR HEATMAP ---
    def dibujar_heatmap(df_datos, titulo, color_map, eje):
        if df_datos.empty:
            eje.text(0.5, 0.5, "Sin Datos", ha='center', va='center')
            return
        
        # Crear matriz y rellenar huecos
        matriz = pd.crosstab(df_datos['Motivación'], df_datos['Pensando en abandonar'])
        matriz = matriz.reindex(index=niveles_fijos, columns=niveles_fijos, fill_value=0)
        
        sns.heatmap(
            matriz, annot=True, cmap=color_map, fmt="d", 
            linewidths=0.5, cbar=False, ax=eje, vmin=0
        )
        eje.set_title(titulo, fontsize=14)
        eje.set_ylabel('Nivel Motivación', fontsize=10)
        eje.set_xlabel('Nivel Abandono', fontsize=10)

    # --- COLUMNA 1: MUJERES ---
    with col1:
        st.markdown("### 👩 Mujeres")
        if not df_filt_m.empty:
            fig1, ax1 = plt.subplots(figsize=(6, 5))
            dibujar_heatmap(df_filt_m, "Matriz Femenina", "Reds", ax1) # Usamos Rojos
            st.pyplot(fig1)
            
            # Métricas
            riesgo_m = len(df_filt_m[(df_filt_m['Motivación']<=2) & (df_filt_m['Pensando en abandonar']>=4)])
            st.metric("Total Mujeres", len(df_filt_m))
            st.caption(f"⚠️ En Riesgo: {riesgo_m} estudiantes")
        else:
            st.warning("No hay datos de mujeres para esta selección.")

    # --- COLUMNA 2: HOMBRES ---
    with col2:
        st.markdown("### 👨 Hombres")
        if not df_filt_h.empty:
            fig2, ax2 = plt.subplots(figsize=(6, 5))
            dibujar_heatmap(df_filt_h, "Matriz Masculina", "Blues", ax2) # Usamos Azules
            st.pyplot(fig2)
            
            # Métricas
            riesgo_h = len(df_filt_h[(df_filt_h['Motivación']<=2) & (df_filt_h['Pensando en abandonar']>=4)])
            st.metric("Total Hombres", len(df_filt_h))
            st.caption(f"⚠️ En Riesgo: {riesgo_h} estudiantes")
        else:
            st.warning("No hay datos de hombres para esta selección.")

    # 5. NOTA AL PIE
    st.markdown("---")

    st.info("💡 **Nota:** Se utilizan paletas de color distintas (Rojos vs Azules) para facilitar la diferenciación visual rápida.")


