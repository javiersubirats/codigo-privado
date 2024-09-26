import numpy as np  # Librería para operaciones matemáticas y manejo de arrays
import matplotlib.pyplot as plt  # Librería para generar gráficos
import pandas as pd  # Librería para manipulación de datos estructurados en dataframes
import streamlit as st  # Librería para crear aplicaciones web interactivas

# Configuración de la página de Streamlit con títulos y subtítulos en HTML
st.markdown('<h1 style="color:#0072ce;">Modelo de Demanda e Ingreso Total</h1>', unsafe_allow_html=True)  # Título principal
st.markdown('<p style="color:#71c5e8;"><strong>DGI-UX by Best Practices</strong></p>', unsafe_allow_html=True)  # Subtítulo

# Entradas del usuario a través de widgets en Streamlit
lambda_value = st.number_input("Demanda máxima (λ)", min_value=0.1, value=18.0, step=0.1)  # Valor de demanda máxima, permite decimales
pmin = st.number_input("Precio mínimo a la venta (pmin)", min_value=1, value=100)  # Precio mínimo aceptado por el usuario
p0 = pmin * 0.8  # Precio inicial basado en el precio mínimo, es el 80% de pmin
max_price = st.number_input("Tarifa máxima", min_value=1, value=1600)  # Precio máximo aceptado por el usuario

# Entrada para valores de α (separados por comas), los convierte en una lista de floats
alpha_values_input = st.text_input("Valores de α (separados por comas)", "30, 130, 230")  # Entrada de texto
alpha_values = [float(alpha.strip()) for alpha in alpha_values_input.split(',')]  # Conversión a lista de floats

# Definición de la función de demanda, que depende del precio (p) y el parámetro α
def demand(p, alpha):
    return lambda_value * np.exp(-(p - p0) / alpha)  # La demanda se calcula usando una función exponencial

# Definición de la función de ingreso total, que es el precio multiplicado por la demanda
def total_revenue(p, alpha):
    return p * demand(p, alpha)

# Generación de un rango de precios desde p0 hasta la tarifa máxima, con 500 puntos
prices_extended = np.linspace(p0, max_price, 500)

# Inicializa la lista de resultados vacía
resultados = []

# Si el usuario hace clic en el botón "Generar Gráficos", se ejecuta el siguiente bloque de código
if st.button("Generar Gráficos"):
    plt.figure(figsize=(12, 12))  # Define el tamaño del gráfico con 3 subgráficos

    # 1. Gráfico de Demanda
    plt.subplot(3, 1, 1)  # Primer gráfico de demanda
    for alpha in alpha_values:
        demands = demand(prices_extended, alpha)  # Calcula la demanda para cada α
        max_demand_index = np.argmax(demands)  # Encuentra el índice donde la demanda es máxima
        max_demand_value = demands[max_demand_index]  # Valor máximo de la demanda
        max_demand_price = prices_extended[max_demand_index]  # Precio correspondiente al valor máximo de demanda
        
        # Calcula el ingreso total para trazar la línea de referencia
        it = total_revenue(prices_extended, alpha)
        max_it_index = np.argmax(it)  # Encuentra el índice donde el ingreso total es máximo
        max_it_price = prices_extended[max_it_index]  # Precio correspondiente al ingreso total máximo
        opt_demand = demands[max_it_index]  # Demanda óptima (en el punto donde IT es máximo)
        
        # Dibuja la curva de demanda y muestra en la leyenda el valor máximo de demanda
        plt.plot(prices_extended, demands, label=f'α = {alpha}, Max D = {opt_demand:.1f}')
        
        # Añade una línea vertical en el precio donde el ingreso total es máximo
        plt.axvline(x=max_it_price, linestyle='--', color='red')  # Línea roja para IT máximo
        # Texto para mostrar la demanda en ese punto
        plt.text(max_it_price, demands[max_it_index], f'D(p)\n{demands[max_it_index]:.1f}', 
                 horizontalalignment='left', fontsize=8, color='red')
    
    # Configuración de etiquetas y leyendas para el gráfico de demanda
    plt.title('Demanda', fontsize=14)
    plt.xlabel('Precio (p)', fontsize=12)
    plt.ylabel('Demanda D(p)', fontsize=12)
    plt.axvline(x=p0, color='gray', linestyle='--', label=f'p0 = {p0}')  # Línea gris en p0
    plt.axvline(x=pmin, color='gray', linestyle='-', label=f'pmin = {pmin}')  # Línea gris en pmin
    plt.legend()
    plt.grid(True)

    # 2. Gráfico de Ingreso Total
    plt.subplot(3, 1, 2)  # Segundo gráfico para el ingreso total
    for alpha in alpha_values:
        it_p0 = total_revenue(p0, alpha)  # Ingreso total en p0
        it = total_revenue(prices_extended, alpha)  # Ingreso total para todos los precios
        max_it_index = np.argmax(it)  # Índice donde IT es máximo
        max_it_price = prices_extended[max_it_index]  # Precio correspondiente al IT máximo
        max_it_value = it[max_it_index]  # Valor máximo de IT
        
        # Dibuja la curva de ingreso total y muestra el valor máximo de IT en la leyenda
        plt.plot(prices_extended, it, label=f'α = {alpha}, Max IT = {max_it_value:.2f}')
        plt.axvline(x=max_it_price, linestyle='--', color='red')  # Línea roja para IT máximo
        plt.text(max_it_price, max_it_value, f'Max IT\n={max_it_value:.1f}', 
                 horizontalalignment='left', fontsize=8, color='red')

    # Configuración de etiquetas y leyendas para el gráfico de ingreso total
    plt.title('Ingreso Total (IT)', fontsize=14)
    plt.xlabel('Precio (p)', fontsize=12)
    plt.ylabel('Ingreso Total IT(p)', fontsize=12)
    plt.axvline(x=p0, color='gray', linestyle='--', label=f'p0 = {p0}')
    plt.axvline(x=pmin, color='gray', linestyle='-', label=f'pmin = {pmin}')
    plt.legend()
    plt.grid(True)

    # 3. Gráfico de Ingreso Marginal (IT(p) - IT(p+1))
    plt.subplot(3, 1, 3)  # Tercer gráfico para la diferencia de IT
    for alpha in alpha_values:
        it = total_revenue(prices_extended, alpha)  # Calcula el ingreso total
        it_difference = it[:-1] - it[1:]  # Calcula la diferencia entre IT(p) e IT(p+1)
        
        # Elimina los valores negativos de la diferencia
        it_difference[it_difference < 0] = 0  
        
        # Encuentra el índice donde la diferencia cruza el eje X
        zero_crossing_indices = np.where(np.diff(np.sign(it_difference)))[0]  # Índices donde cruza cero
        
        # Si existen cruces, dibuja la línea vertical en el cruce
        if len(zero_crossing_indices) > 0:
            zero_crossing_index = zero_crossing_indices[0]
            zero_crossing_price = prices_extended[zero_crossing_index]  # Precio en el cruce
        
            plt.axvline(x=zero_crossing_price, linestyle='--', color='red')  # Línea roja en el cruce
            revenue_at_zero_crossing = total_revenue(zero_crossing_price, alpha)  # Ingreso en ese punto
            
            # Muestra el cruce con el texto en el gráfico
            plt.text(zero_crossing_price, 0, f'Max IM\nim={zero_crossing_price:.1f}', 
                     horizontalalignment='left', fontsize=8, color='red')
            
            # Dibuja la curva de diferencia y la marca en la leyenda
            plt.plot(prices_extended[:-1], it_difference, label=f'α = {alpha}, popt = {zero_crossing_price:.1f}')
        else:
            plt.plot(prices_extended[:-1], it_difference, label=f'α = {alpha}, popt = N/A')  # Si no hay cruce
        
    # Configuración de etiquetas y leyendas para el gráfico de ingreso marginal
    plt.title('Ingreso Marginal IT(p) - IT(p+1) (solo positivos)', fontsize=14)
    plt.xlabel('Precio (p)', fontsize=12)
    plt.ylabel('Ingreso Marginal IM(p)', fontsize=12)
    plt.axvline(x=p0, color='gray', linestyle='--', label=f'p0 = {p0}')
    plt.axvline(x=pmin, color='gray', linestyle='-', label=f'pmin = {pmin}')
    plt.legend()
    plt.grid(True)

    # Ajusta el diseño del gráfico y muestra los gráficos en la aplicación Streamlit

    plt.tight_layout()  # Ajusta automáticamente el espaciado entre los subgráficos
    st.pyplot(plt)  # Muestra el gráfico completo en la aplicación de Streamlit

    # Mostrar tablas en Streamlit con formato europeo
    for alpha in alpha_values:
        # Crea un DataFrame con el precio, la demanda y el ingreso total para cada valor de α
        df = pd.DataFrame({
            'Precio': prices_extended,  # Columna de precios
            'Demanda': demand(prices_extended, alpha),  # Columna de demanda calculada
            'Ingreso Total': total_revenue(prices_extended, alpha)  # Columna de ingreso total calculado
        })
        
        # Aplicar formato europeo a los valores numéricos (separador de miles con '.' y decimales con ',')
        df = df.applymap(lambda x: f"{x:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.') if isinstance(x, (int, float)) else x)
        
        # Muestra la tabla en Streamlit para cada α
        st.write(f"Resultados para α = {alpha}")
        st.dataframe(df)  # Muestra el DataFrame en Streamlit con los resultados
