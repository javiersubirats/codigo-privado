import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Configuración de la página de Streamlit
st.title("Modelo de Demanda e Ingreso Total")

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

# Rango de precios entre 0 y la tarifa máxima
prices_extended = np.linspace(0, max_price)

# Generación de gráficos
if st.button("Generar Gráficos"):
    plt.figure(figsize=(12, 12))

    # Gráfico de demanda
    plt.subplot(3, 1, 1)
    for alpha in alpha_values:
        demands = demand(prices_extended, alpha)
        plt.plot(prices_extended, demands, label=f'α = {alpha}')

        # Encontrar el precio donde IT es máximo
        it = total_revenue(prices_extended, alpha)
        max_it_index = np.argmax(it)
        max_it_price = prices_extended[max_it_index]

        # Dibujar línea vertical en el precio donde IT es máximo
        plt.axvline(x=max_it_price, linestyle='--', color='red')
        plt.text(max_it_price, demands[max_it_index], f'D(p)\n{demands[max_it_index]:.1f}', 
                 horizontalalignment='left', fontsize=8, color='red')

    plt.title('Demanda vs Precio para diferentes valores de α', fontsize=14)
    plt.xlabel('Precio (p)', fontsize=12)
    plt.ylabel('Demanda D(p)', fontsize=12)
    plt.axvline(x=p0, color='gray', linestyle='--', label=f'Precio p0 = {p0}')
    plt.legend()
    plt.grid(True)

    # Gráfico de ingreso total
    plt.subplot(3, 1, 2)
    for alpha in alpha_values:
        it = total_revenue(prices_extended, alpha)
        plt.plot(prices_extended, it, label=f'α = {alpha}')

        # Encontrar el precio donde IT es máximo
        max_it_index = np.argmax(it)
        max_it_price = prices_extended[max_it_index]
        plt.axvline(x=max_it_price, linestyle='--', color='red')
        plt.text(max_it_price, it[max_it_index], f'Max IT\np={max_it_price:.1f}', 
                 horizontalalignment='left', fontsize=8, color='red')

    plt.title('Ingreso Total (IT) vs Precio para diferentes valores de α', fontsize=14)
    plt.xlabel('Precio (p)', fontsize=12)
    plt.ylabel('Ingreso Total IT(p)', fontsize=12)
    plt.axvline(x=p0, color='gray', linestyle='--', label=f'Precio p0 = {p0}')
    plt.legend()
    plt.grid(True)

    # Gráfico de IT(p) - IT(p+1)
    plt.subplot(3, 1, 3)
    for alpha in alpha_values:
        it = total_revenue(prices_extended, alpha)
        
        # Calcular IT(p) - IT(p+1)
        it_difference = np.zeros_like(it)
        it_difference[:-1] = it[:-1] - it[1:]  # Diferencia de IT
        
        # Filtrar los valores negativos, reemplazándolos por 0 (opcional)
        it_difference[it_difference < 0] = 0
        
        # Graficar solo los valores positivos de IT(p) - IT(p+1)
        plt.plot(prices_extended[:-1], it_difference[:-1], label=f'α = {alpha}')  # Empezamos hasta el penúltimo índice
    
    # Añadir detalles al gráfico de diferencia de IT
    plt.title('IT(p) - IT(p+1) (solo positivos) vs Precio para diferentes valores de α', fontsize=14)
    plt.xlabel('Precio (p)', fontsize=12)
    plt.ylabel('IT(p) - IT(p+1)', fontsize=12)
    plt.axvline(x=p0, color='gray', linestyle='--', label=f'Precio p0 = {p0}')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

    
    # Mostrar gráfico en Streamlit
    st.pyplot(plt)
