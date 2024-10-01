import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Configuración de la página de Streamlit
st.markdown('<h1 style="color:#0072ce;">Modelo de Demanda e Ingreso Total</h1>', unsafe_allow_html=True)
st.markdown('<p style="color:#71c5e8;"><strong>DGI-UX by Best Practices</strong></p>', unsafe_allow_html=True)

# Entrada para valores de lambda (separados por comas), que se convierten en una lista de floats
lambda_values_input = st.text_input("Valores de demanda máxima λ (separados por comas)", "18, 20")
lambda_values = [float(lambda_value.strip()) for lambda_value in lambda_values_input.split(',')]

# Entrada para valores de α (separados por comas)
alpha_values_input = st.text_input("Valores de α (separados por comas)", "30, 130, 230")
alpha_values = [float(alpha.strip()) for alpha in alpha_values_input.split(',')]

# Entradas adicionales
pmin = st.number_input("Precio mínimo a la venta (pmin), DF cls Z", min_value=1, value=100)
p0 = pmin * 0.8  # Precio inicial basado en el 80% de pmin
max_price = st.number_input("Tarifa máxima (pmax), DF cls Y", min_value=1, value=1600)

# Función de demanda
def demand(p, alpha, lambda_value):
    return lambda_value * np.exp(-(p - p0) / alpha)

# Función de ingreso total
def total_revenue(p, alpha, lambda_value):
    return p * demand(p, alpha, lambda_value)

# Rango de precios
prices_extended = np.linspace(p0, max_price, 500)

# Si el usuario hace clic en el botón "Generar Gráficos"
if st.button("Generar Gráficos"):
    plt.figure(figsize=(12, 12))  # Tamaño de gráfico

    # 1. Gráfico de Demanda
    plt.subplot(3, 1, 1)
    for lambda_value in lambda_values:
        for alpha in alpha_values:
            demands = demand(prices_extended, alpha, lambda_value)
            max_demand_index = np.argmax(demands)
            max_it_index = np.argmax(total_revenue(prices_extended, alpha, lambda_value))
            opt_demand = demands[max_it_index]
            
            # Dibujar curva de demanda
            plt.plot(prices_extended, demands, label=f'λ = {lambda_value}, α = {alpha}, Max D = {opt_demand:.1f}')
            plt.axvline(x=prices_extended[max_it_index], linestyle='--', color='red')
            plt.text(prices_extended[max_it_index], demands[max_it_index], f'D(p)\n{demands[max_it_index]:.1f}', 
                     horizontalalignment='left', fontsize=8, color='red')
    
    plt.title('Demanda', fontsize=14)
    plt.xlabel('Precio (p)', fontsize=12)
    plt.ylabel('Demanda D(p)', fontsize=12)
    plt.axvline(x=p0, color='gray', linestyle='--', label=f'p0 = {p0}')
    plt.axvline(x=pmin, color='gray', linestyle='-', label=f'pmin = {pmin}')
    plt.legend()
    plt.grid(True)

    # 2. Gráfico de Ingreso Total
    plt.subplot(3, 1, 2)
    for lambda_value in lambda_values:
        for alpha in alpha_values:
            it = total_revenue(prices_extended, alpha, lambda_value)
            max_it_index = np.argmax(it)
            max_it_value = it[max_it_index]
            
            plt.plot(prices_extended, it, label=f'λ = {lambda_value}, α = {alpha}, Max IT = {max_it_value:.2f}')
            plt.axvline(x=prices_extended[max_it_index], linestyle='--', color='red')
            plt.text(prices_extended[max_it_index], max_it_value, f'Max IT\n={max_it_value:.1f}', 
                     horizontalalignment='left', fontsize=8, color='red')

    plt.title('Ingreso Total (IT)', fontsize=14)
    plt.xlabel('Precio (p)', fontsize=12)
    plt.ylabel('Ingreso Total IT(p)', fontsize=12)
    plt.axvline(x=p0, color='gray', linestyle='--', label=f'p0 = {p0}')
    plt.axvline(x=pmin, color='gray', linestyle='-', label=f'pmin = {pmin}')
    plt.legend()
    plt.grid(True)

    # 3. Gráfico de Ingreso Marginal (IT(p) - IT(p+1))
    plt.subplot(3, 1, 3)
    for lambda_value in lambda_values:
        for alpha in alpha_values:
            it = total_revenue(prices_extended, alpha, lambda_value)
            it_difference = it[:-1] - it[1:]
            it_difference[it_difference < 0] = 0
            zero_crossing_indices = np.where(np.diff(np.sign(it_difference)))[0]
            
            if len(zero_crossing_indices) > 0:
                zero_crossing_price = prices_extended[zero_crossing_indices[0]]
                plt.axvline(x=zero_crossing_price, linestyle='--', color='red')
                plt.text(zero_crossing_price, 0, f'Max IM\nim={zero_crossing_price:.1f}', 
                         horizontalalignment='left', fontsize=8, color='red')
                plt.plot(prices_extended[:-1], it_difference, label=f'λ = {lambda_value}, α = {alpha}, popt = {zero_crossing_price:.1f}')
            else:
                plt.plot(prices_extended[:-1], it_difference, label=f'λ = {lambda_value}, α = {alpha}, popt = N/A')
        
    plt.title('Ingreso Marginal IT(p) - IT(p+1) (solo positivos)', fontsize=14)
    plt.xlabel('Precio (p)', fontsize=12)
    plt.ylabel('Ingreso Marginal IM(p)', fontsize=12)
    plt.axvline(x=p0, color='gray', linestyle='--', label=f'p0 = {p0}')
    plt.axvline(x=pmin, color='gray', linestyle='-', label=f'pmin = {pmin}')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    st.pyplot(plt)

   # Cálculo del ingreso incremental y la demanda incremental
    def transformed_price(it, demands):
        # Calcula la diferencia entre ingresos totales (IT(p_i) - IT(p_{i+1}))
        delta_it = it[:-1] - it[1:]
        # Calcula la diferencia entre demandas (D(p_i) - D(p_{i+1}))
        delta_demand = demands[:-1] - demands[1:]
        # Evita divisiones por cero en la demanda incremental
        with np.errstate(divide='ignore', invalid='ignore'):
            transformed_prices = np.where(delta_demand != 0, delta_it / delta_demand, np.nan)
        return transformed_prices
    
    # Mostrar tablas en Streamlit con formato europeo, incluyendo el "Transformed Price"
    for alpha in alpha_values:
        # Cálculos de demanda e ingreso total
        demands = demand(prices_extended, alpha, lambda_value)
        it = total_revenue(prices_extended, alpha)
        
        # Cálculo del "Transformed Price"
        transformed_prices = transformed_price(it, demands)
        
        # Crea un DataFrame con el precio, la demanda, el ingreso total y el transformed price
        df = pd.DataFrame({
            'Precio': prices_extended[:-1],  # Usamos hasta el penúltimo valor debido a los cálculos incrementales
            'Demanda': demands[:-1],  # También hasta el penúltimo valor
            'Ingreso Total': it[:-1],  # Hasta el penúltimo valor
            'Transformed Price': transformed_prices  # Añadimos el transformed price
        })
        
        # Aplicar formato europeo a los valores numéricos (separador de miles con '.' y decimales con ',')
        df = df.applymap(lambda x: f"{x:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.') if isinstance(x, (int, float)) else x)
        
        # Muestra la tabla en Streamlit para cada α
        st.write(f"Resultados para α = {alpha}")
        st.dataframe(df)  # Muestra el DataFrame en Streamlit con los resultados
