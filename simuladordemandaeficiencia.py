import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Configuración de la página de Streamlit
st.markdown('<h1 style="color:#0072ce;">Modelo de Demanda e Ingreso Total</h1>', unsafe_allow_html=True)
st.markdown('<p style="color:#71c5e8;"><strong>DGI-UX by Best Practices</strong></p>', unsafe_allow_html=True)

# Entradas del usuario
lambda_value = st.number_input("Demanda máxima (λ)", min_value=0.1, value=18.0, step=0.1)  # Acepta decimales
pmin = st.number_input("Precio minimo a la venta (pmin)", min_value=1, value=100)
p0 = pmin*0.8
max_price = st.number_input("Tarifa máxima", min_value=1, value=1600)

# Entrada de valores de alpha como una lista de decimales
alpha_values_input = st.text_input("Valores de α (separados por comas)", "30, 130, 230")
alpha_values = [float(alpha.strip()) for alpha in alpha_values_input.split(',')]  # Convierte a float

# Función de demanda con diferentes valores de alpha
def demand(p, alpha):
    return lambda_value * np.exp(-(p - p0) / alpha)

# Función de ingreso total
def total_revenue(p, alpha):
    return p * demand(p, alpha)

# Rango de precios entre p0 y la tarifa máxima
prices_extended = np.linspace(p0, max_price, 500)

# Inicializa la lista de resultados
resultados = []

# Generación de gráficos
if st.button("Generar Gráficos"):
    plt.figure(figsize=(12, 12))

    # 1. Gráfico de demanda
    plt.subplot(3, 1, 1)
    for alpha in alpha_values:
        d_p0 = demand(p0, alpha)
        demands = demand(prices_extended, alpha)
        plt.plot(prices_extended, demands, label=f'α = {alpha}, D = {d_p0}')  # Añadir p0 en la leyenda
        
        # Ingreso total
        it = total_revenue(prices_extended, alpha)
        max_it_index = np.argmax(it)
        max_it_price = prices_extended[max_it_index]
    
        # Dibujar línea vertical en el precio donde IT es máximo
        plt.axvline(x=max_it_price, linestyle='--', color='red')
        plt.text(max_it_price, demands[max_it_index], f'D(p)\n{demands[max_it_index]:.1f}', 
                 horizontalalignment='left', fontsize=8, color='red')
    
    plt.title('Demanda', fontsize=14)
    plt.xlabel('Precio (p)', fontsize=12)
    plt.ylabel('Demanda D(p)', fontsize=12)
    plt.axvline(x=p0, color='gray', linestyle='--', label=f'Precio p0 = {p0}')
    plt.legend()
    plt.grid(True)
    
    # 2. Gráfico de ingreso total
    plt.subplot(3, 1, 2)
    for alpha in alpha_values:
        it_p0 = total_revenue(p0, alpha)
        it = total_revenue(prices_extended, alpha)
        plt.plot(prices_extended, it, label=f'α = {alpha}, IT = {it_p0}')  # Añadir p0 en la leyenda
    
        max_it_index = np.argmax(it)
        max_it_price = prices_extended[max_it_index]
        plt.axvline(x=max_it_price, linestyle='--', color='red')
        plt.text(max_it_price, it[max_it_index], f'Max IT\n={it[max_it_index]:.1f}', 
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
        it_difference = it[:-1] - it[1:]  # Diferencia entre IT(p) y IT(p+1)
        
        # Eliminar valores negativos
        it_difference[it_difference < 0] = 0  
    
        
        
        # Encontrar el índice donde el ingreso marginal cruza el eje X
        zero_crossing_indices = np.where(np.diff(np.sign(it_difference)))[0]
        plt.plot(prices_extended[:-1], it_difference, label=f'α = {alpha}, p0 = {p0}')  # Añadir p0 en la leyenda
        if len(zero_crossing_indices) > 0:
            zero_crossing_index = zero_crossing_indices[0]  # Toma el primer cruce
            zero_crossing_price = prices_extended[zero_crossing_index]  # Precio correspondiente
    
            # Dibujar línea roja en el cruce con el eje X
            plt.axvline(x=zero_crossing_price, linestyle='--', color='red')
    
            # Valor del ingreso total en ese punto
            revenue_at_zero_crossing = total_revenue(zero_crossing_price, alpha)
    
            plt.text(zero_crossing_price, 0, f'Max IM\nim={zero_crossing_price:.1f}', 
                     horizontalalignment='left', fontsize=8, color='red')
    
    plt.title('Ingreso Marginal IT(p) - IT(p+1) (solo positivos)', fontsize=14)
    plt.xlabel('Precio (p)', fontsize=12)
    plt.ylabel('Ingreso Marginal IM(p)', fontsize=12)
    plt.axvline(x=p0, color='gray', linestyle='--', label=f'Precio p0 = {p0}')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    st.pyplot(plt)

    # Mostrar tablas en Streamlit con formato europeo
    for alpha in alpha_values:
        df = pd.DataFrame({
            'Precio': prices_extended,
            'Demanda': demand(prices_extended, alpha),
            'Ingreso Total': total_revenue(prices_extended, alpha)
        })
        df = df.applymap(lambda x: f"{x:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.') if isinstance(x, (int, float)) else x)
        st.write(f"Resultados para α = {alpha}")
        st.dataframe(df)
