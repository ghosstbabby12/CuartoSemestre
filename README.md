# CuartoSemestre
librerías necesarias 
(backend)

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
para ejecutar:

python -m uvicorn backend.main:app --reload

(frontend)
import streamlit as st
import requests
import sympy as sp
import numpy as np
import plotly.graph_objs as go
para ejecurar:
streamlit run app.py

