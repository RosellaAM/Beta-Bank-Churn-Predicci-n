# Predicción de Pérdida de Clientes en Beta Bank

Beta Bank enfrenta el desafío de la pérdida gradual de clientes mes tras mes. Este proyecto desarrolla un modelo predictivo de machine learning para identificar clientes con alta probabilidad de abandonar el banco (churn). El objetivo principal es crear un modelo de clasificación que maximice la métrica F1 (con un valor mínimo requerido de 0.59) y complementar el análisis con la métrica AUC-ROC para una evaluación integral del rendimiento.

El proyecto sigue una metodología estructurada que incluye carga y exploración de datos, preprocesamiento, análisis del desequilibrio de clases, técnicas de balanceo, ajuste de parámetros y evaluación final del modelo.

## 🎯 Resultados del Proyecto
El proyecto ha permitido desarrollar un modelo de machine learning capaz de identificar con precisión a aquellos clientes con alta probabilidad de abandonar el banco. Tras un proceso exhaustivo, se logró obtener un modelo que supera el objetivo inicial:
* Valor F1: 0.593 (superando el objetivo de 0.59)
* AUC-ROC: 0.87
* Precisión del modelo: 87% en el área bajo la curva ROC

## 🚀 Impacto en la Empresa
* Identificación temprana de clientes en riesgo de abandono
* Estrategias de retención proactivas y personalizadas
* Optimización de recursos al enfocar esfuerzos en clientes con mayor probabilidad de abandono
* Reducción tangible de la tasa de churn e incremento en ingresos por retención
* Mejora en la satisfacción del cliente mediante atención preventiva

## 🎯 Habilidades principales
* **Preprocesamiento de datos**: Manejo de valores nulos, codificación de variables categóricas, escalado de características.
* **Análisis exploratorio**: Identificación de desequilibrio de clases, análisis de distribuciones.
* **Ingeniería de características**: Transformación y preparación de datos para modelado.
* **Selección de modelos*: Comparación y evaluación de múltiples algoritmos.
* **Optimización**: Ajuste de hiperparámetros y técnicas de balanceo.
* **Evaluación de modelos**: Análisis comprehensivo mediante múltiples métricas.
* **Resolución de problemas empresariales**: Enfoque práctico para abordar desafíos de negocio.

## 🛠️ Stack Tecnológico
* **Frontend** -> Scikit-learn
* **Backend** -> Python 3.8+, Pandas, NumPy
* **Visualización** -> Matplotlib, Seaborn
* **Desarrollo** -> Jupyter Notebooks

## Ejecución Local
1. Clona el repositorio:

git clone https://github.com/RosellaAM/Megaline-Plan-Recommendation.git

2. Instala dependencias:

pip install -r requirements.txt

3. Ejecución de análisis:

  jupyter notebook notebooks/prediccion_churn_beta_bank.ipynb
