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
pmin = st.number_input("Precio mínimo (p0)", min_value=1, value=100)
p0 = pmin  # Precio inicial basado en el 80% de pmin
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
    #plt.axvline(x=pmin, color='gray', linestyle='-', label=f'pmin = {pmin}')
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
    #plt.axvline(x=pmin, color='gray', linestyle='-', label=f'pmin = {pmin}')
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
    #plt.axvline(x=pmin, color='gray', linestyle='-', label=f'pmin = {pmin}')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    st.pyplot(plt)

      # Cálculo del "Transformed Price"
    def transformed_price(it, demands):
        # Inicializar una lista para los precios transformados
        transformed_prices = []
    
        # Iterar sobre los índices de las demandas y los ingresos
        for i in range(len(it) - 1):
            if demands[i+1] != 0:  # Asegurarse de no dividir por 0
                # Cálculo del transformed price como ingreso incremental / demanda incremental
                transformed_price_value = (it[i] - it[i+1]) / (demands[i] - demands[i+1])
            else:
                transformed_price_value = 0  # Para evitar división por cero
            
            # Añadir el precio transformado a la lista
            transformed_prices.append(transformed_price_value)
    
        # Añadir el valor de pmax al final (el precio máximo debe ser igual al transformed price en pmax)
        transformed_prices.append(prices_extended[-1])
    
        return np.array(transformed_prices)
    
    # Ahora llamas a esta función después de calcular 'it' y 'demands'
    for alpha in alpha_values:
        if alpha <= 0:
            st.error("Alpha debe ser mayor que 0.")
            continue
    
        # Cálculos de demanda e ingreso total
        demands = demand(prices_extended, alpha, lambda_value)  # Añadir lambda_value
        it = total_revenue(prices_extended, alpha, lambda_value)  # Añadir lambda_value
    
        # Cálculo del "Transformed Price"
        transformed_prices = transformed_price(it, demands)
    
        # Asegurarte de que todas las listas tengan la misma longitud
        # Para eso, recortamos las listas al mismo tamaño, tomando la longitud mínima entre ellas
        min_length = min(len(prices_extended), len(demands), len(it), len(transformed_prices))
        
        # Recortar las listas a la longitud mínima
        prices_extended = prices_extended[:min_length]
        demands = demands[:min_length]
        it = it[:min_length]
        transformed_prices = transformed_prices[:min_length]
        
        # Crear el DataFrame con los resultados
        df = pd.DataFrame({
            'Precio': prices_extended,
            'Demanda': demands,
            'Ingreso Total': it,
            'Transformed Price': transformed_prices
            
        # Aplicar formato a 'Transformed Price' para que tenga solo un decimal
        df['Precio'] = df['Precio'].map('{:.1f}'.format)
        df['Transformed Price'] = df['Transformed Price'].map('{:.1f}'.format)
        
        st.write(f"Resultados para α = {alpha}")
        st.dataframe(df)

