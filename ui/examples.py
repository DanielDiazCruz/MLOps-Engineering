"""Payloads de ejemplo predefinidos para el formulario de inferencia.

Coinciden con el conjunto de features que produce pipeline/preprocess.py para
el problema de regresión inmobiliaria:
  - Numéricas: bed, bath, acre_lot, house_size, prev_sold_year
  - Categóricas (strings): status, city, state, zip_code

El modelo (Pipeline de sklearn) hace su propio one-hot encoding con
`handle_unknown="infrequent_if_exist"`, así que las categorías nuevas no
rompen la predicción.
"""

from __future__ import annotations

# Casa suburbana típica en venta (Nueva York)
SAMPLE_HOUSE: dict = {
    "bed": 3,
    "bath": 2,
    "acre_lot": 0.25,
    "house_size": 1800,
    "prev_sold_year": 2015,
    "status": "for_sale",
    "city": "New York",
    "state": "New York",
    "zip_code": "10001",
}

# Apartamento pequeño (Puerto Rico, datos frecuentes en el dataset)
SAMPLE_CONDO: dict = {
    "bed": 1,
    "bath": 1,
    "acre_lot": 0.05,
    "house_size": 700,
    "prev_sold_year": 2018,
    "status": "for_sale",
    "city": "San Juan",
    "state": "Puerto Rico",
    "zip_code": "00901",
}
