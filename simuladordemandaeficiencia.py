import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Configuración de la página de Streamlit
st.title("Modelo de Demanda e Ingreso Total")
st.markdown("**Dir.GIA by Best Practices**")  # Añadir aquí el texto

# Entradas del usuario
lambda_value = st.number_input("Demanda máxima (λ)", min_value=1, value=18)
p0 = st.number_input("Precio de referencia (p0)", min_value=1, value=100)
max_price = st.number_input("Tarifa máxima", min_value=1, value=1600)

# Entrada de valores de alpha como una lista de enteros
alpha_values_input = st.text_input("Valores de α (separados por comas)", "30, 130, 230")
alpha_values = [int(alpha.strip()) for alpha in alpha_values_input.split(',')]

# Función de demanda con diferentes valores de alpha
def demand(p, alpha):
    return lambda_value * np.exp(-(p - p0) / alpha)

# Función de ingreso total
def total_revenue(p, alpha):
    return p * demand(p, alpha)

# Rango de precios entre p0 y la tarifa máxima
prices_extended = np.linspace(p0*0.8, max_price, 500)

# Generación de gráficos
if st.button("Generar Gráficos"):
    plt.figure(figsize=(12, 12))

    # Crear contenedores de resultados
    resultados = []

    # 1. Gráfico de demanda
    plt.subplot(3, 1, 1)
    for alpha in alpha_values:
        demands = demand(prices_extended, alpha)
        plt.plot(prices_extended, demands, label=f'α = {alpha}')
        
        # Ingreso total
        it = total_revenue(prices_extended, alpha)
        max_it_index = np.argmax(it)
        max_it_price = prices_extended[max_it_index]

        # Dibujar línea vertical en el precio donde IT es máximo
        plt.axvline(x=max_it_price, linestyle='--', color='red')
        plt.text(max_it_price, demands[max_it_index], f'D(p)\n{demands[max_it_index]:.1f}', 
                 horizontalalignment='left', fontsize=8, color='red')

        # Calcular IT(p) - IT(p+1)
        it_difference = it[:-1] - it[1:]
        it_difference[it_difference < 0] = 0  
        it_difference = np.append(it_difference, 0)

        # Añadir resultados a la lista
        tabla_datos = {
            'Precio': prices_extended,
            'Demanda': demands,
            'Ingreso Total': it,
            'IT(p) - IT(p+1)': it_difference
        }
        resultados.append(pd.DataFrame(tabla_datos))

    plt.title('Demanda', fontsize=14)
    plt.xlabel('Precio (p)', fontsize=12)
    plt.ylabel('Demanda D(p)', fontsize=12)
    plt.axvline(x=p0, color='gray', linestyle='--', label=f'Precio p0 = {p0}')
    plt.legend()
    plt.grid(True)

    # 2. Gráfico de ingreso total
    plt.subplot(3, 1, 2)
    for alpha in alpha_values:
        it = total_revenue(prices_extended, alpha)
        plt.plot(prices_extended, it, label=f'α = {alpha}')

        max_it_index = np.argmax(it)
        max_it_price = prices_extended[max_it_index]
        plt.axvline(x=max_it_price, linestyle='--', color='red')
        plt.text(max_it_price, it[max_it_index], f'Max IT\np={max_it_price:.1f}', 
                 horizontalalignment='left', fontsize=8, color='red')

    plt.title('Ingreso Total (IT)', fontsize=14)
    plt.xlabel('Precio (p)', fontsize=12)
    plt.ylabel('Ingreso Total IT(p)', fontsize=12)
    plt.axvline(x=p0, color='gray', linestyle='--', label=f'Precio p0 = {p0}')
    plt.legend()
    plt.grid(True)

    # 3. Gráfico de IT(p) - IT(p+1)
    plt.subplot(3, 1, 3)
    for alpha in alpha_values:
        it = total_revenue(prices_extended, alpha)
        it_difference = it[:-1] - it[1:]
        it_difference[it_difference < 0] = 0  
        plt.plot(prices_extended[:-1], it_difference, label=f'α = {alpha}')
    
    plt.title('Ingreso Marginal IT(p) - IT(p+1) (solo positivos)', fontsize=14)
    plt.xlabel('Precio (p)', fontsize=12)
    plt.ylabel('Ingreso Marginal', fontsize=12)
    plt.axvline(x=p0, color='gray', linestyle='--', label=f'Precio p0 = {p0}')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    st.pyplot(plt)

    # Mostrar tablas en Streamlit con formato europeo
    for i, df in enumerate(resultados):
        # Formato europeo
        df = df.applymap(lambda x: f"{x:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.') if isinstance(x, (int, float)) else x)
        
        st.write(f"Resultados para α = {alpha_values[i]}")
        st.dataframe(df)
