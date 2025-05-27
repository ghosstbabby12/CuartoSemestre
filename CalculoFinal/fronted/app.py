import streamlit as st
import requests
import sympy as sp
import numpy as np
import plotly.graph_objs as go

st.set_page_config(page_title="Gradiente Visual Interactivo", layout="wide")
st.title("🧠 Derivadas Parciales y Gradiente - Visualización Interactiva")

# Inputs
funcion = st.text_input("🧮 Ingresa la función f(x, y):", value="x**2 + y**2 + 3*x*y")
x0 = st.number_input("📍 Valor de x₀", value=1.0)
y0 = st.number_input("📍 Valor de y₀", value=2.0)

if st.button("Calcular"):
    with st.spinner("Procesando..."):
        response = requests.post("http://localhost:8000/resolver", json={
            "expression": funcion,
            "x0": x0,
            "y0": y0
        })

        if response.status_code == 200:
            data = response.json()

            st.success("¡Todo listo!")

            st.subheader("📘 Paso a paso (tipo Photomath):")
            for paso in data["pasos"]:
                st.markdown(f"- {paso}")

            # Convertir función a NumPy
            x, y = sp.symbols('x y')
            f_sympy = sp.sympify(funcion)
            f_lambdified = sp.lambdify((x, y), f_sympy, 'numpy')

            # Generar malla
            X = np.linspace(x0 - 5, x0 + 5, 50)
            Y = np.linspace(y0 - 5, y0 + 5, 50)
            X, Y = np.meshgrid(X, Y)
            Z = f_lambdified(X, Y)

            # Valor exacto del punto
            z0 = f_lambdified(x0, y0)
            grad = data["evaluacion"]["gradiente"]

            # Crear gráfica interactiva
            surface = go.Surface(x=X, y=Y, z=Z, colorscale='Viridis', opacity=0.8)
            point = go.Scatter3d(x=[x0], y=[y0], z=[z0], mode='markers', marker=dict(size=5, color='red'), name='Punto')
            vector = go.Cone(
                x=[x0], y=[y0], z=[z0],
                u=[grad[0]], v=[grad[1]], w=[0],
                sizemode="absolute", sizeref=1, anchor="tail", colorscale="Blues", showscale=False,
                name="Vector Gradiente"
            )

            fig = go.Figure(data=[surface, point, vector])
            fig.update_layout(
                title="🌐 Superficie f(x, y) y Vector Gradiente",
                scene=dict(
                    xaxis_title='x',
                    yaxis_title='y',
                    zaxis_title='f(x, y)',
                    aspectmode='cube'
                ),
                margin=dict(l=0, r=0, b=0, t=40)
            )

            st.plotly_chart(fig, use_container_width=True)

        else:
            st.error("❌ Error en el backend.")
