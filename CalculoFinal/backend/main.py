from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

x, y = sp.symbols('x y')
app = FastAPI()

class FunctionInput(BaseModel):
    expression: str
    x0: float
    y0: float

@app.post("/resolver")
def resolver_funcion(data: FunctionInput):
    try:
        f = sp.sympify(data.expression)
        df_dx = sp.diff(f, x)
        df_dy = sp.diff(f, y)
        grad_x = df_dx.subs({x: data.x0, y: data.y0})
        grad_y = df_dy.subs({x: data.x0, y: data.y0})

        pasos = [
            f"Se recibe la función: f(x, y) = {sp.pretty(f)}",
            f"Se calcula ∂f/∂x = {sp.pretty(df_dx)}",
            f"Se calcula ∂f/∂y = {sp.pretty(df_dy)}",
            f"Se evalúa ∂f/∂x en ({data.x0}, {data.y0}) = {grad_x}",
            f"Se evalúa ∂f/∂y en ({data.x0}, {data.y0}) = {grad_y}",
            f"El vector gradiente en ({data.x0}, {data.y0}) es: ({grad_x}, {grad_y})"
        ]

        # Gráfico 3D
        f_lambdified = sp.lambdify((x, y), f, 'numpy')
        X = np.linspace(data.x0 - 5, data.x0 + 5, 50)
        Y = np.linspace(data.y0 - 5, data.y0 + 5, 50)
        X, Y = np.meshgrid(X, Y)
        Z = f_lambdified(X, Y)
        z0 = f_lambdified(data.x0, data.y0)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
        ax.scatter(data.x0, data.y0, z0, color='red', s=50, label='Punto de Evaluación')
        ax.quiver(data.x0, data.y0, z0, float(grad_x), float(grad_y), 0, color='blue', length=1, normalize=True)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('f(x, y)')
        ax.set_title('Superficie y Vector Gradiente')
        ax.legend()

        # Convertir gráfico a base64
        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        return {
            "funcion": str(f),
            "derivada_parcial_x": str(df_dx),
            "derivada_parcial_y": str(df_dy),
            "evaluacion": {
                "x0": data.x0,
                "y0": data.y0,
                "gradiente": [float(grad_x), float(grad_y)]
            },
            "pasos": pasos,
            "grafico": img_base64
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en el análisis: {str(e)}")
