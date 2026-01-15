# Detección de rumores en Twitter mediante características sociales y lingüísticas

## 🔎 Sobre el Proyecto

Este proyecto aborda el problema de la **detección de desinformación (rumores)** en redes sociales utilizando el **dataset PHEME**.  
A diferencia de enfoques basados en análisis semántico del texto, este trabajo se centra en **características estructurales y de interacción social**, como:

- Número de retweets  
- Número de favoritos  
- Tipo de evento  
- Dinámica de difusión del hilo  
- Longitud de texto del tweet

El objetivo principal es evaluar hasta qué punto estas señales permiten distinguir entre información verdadera y falsa **sin analizar directamente el contenido textual**.

---

## 📂 Datos

### Dataset

- **Nombre:** PHEME Rumour Dataset  
- **Fuente oficial:**  
  👉 <https://figshare.com/articles/dataset/PHEME_dataset_of_rumours_and_non-rumours/4010619>  

El dataset no se incluye directamente en el repositorio debido a su tamaño y licencia.  
El usuario debe descargarlo manualmente desde el enlace oficial y colocarlo en la carpeta correspondiente.

---

## 🗂️ Estructura del Repositorio

```text
.
├── data/
│   ├── raw/                 # Dataset original PHEME en csv
│   └── processed/           # Datos procesados listos para entrenamiento
├── dataset/                 #  Dataset original
├── notebooks/
│   └── load_data.ipynb      # Análisis exploratorio y carga inicial
├── outputs/
│   ├── figures/             # Figuras y visualizaciones generadas
│   ├── metrics/             # Métricas y resultados de evaluación
│   ├── models/              # Modelos entrenados
│   └── label_mappings.json  # Mapeo de etiquetas
├── scripts/
│   ├── crossvalidation.py   # Búsqueda de hiperparámetros (CV)
│   ├── eval.py              # Evaluación final del modelo
│   ├── models.py            # Definición de modelos ML
│   ├── preprocess.py        # Preprocesamiento para entrenamiento
│   └── train.py             # Entrenamiento de modelos
├── src/
│   └── preprocess/
│       ├── build_raw.py     # Consolidación del dataset en CSV
│       └── build_features.py# Generación de features finales
├── run_all.py               # Ejecución completa del pipeline
├── requirements.txt         # Dependencias del proyecto
└── README.md

```

---

## ⚙️ Instalación

**Prerrequisitos**
- Python ≥ 3.9
- `pip` o `conda`

**Crear entorno virtual**

```bash
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows
```

**Instalar dependencias**

```bash
pip install -r requirements.txt
```

---

## Uso

### Preparación de los datos

Una vez descargado el dataset desde el enlace oficial, colócalo en la raíz del repositorio dentro de una carpera 'dataset'.

Luego ejecuta:

```bash
python src/preprocess/build_raw.py
```

Este script consolida el dataset completo en un único archivo CSV.
A continuación, genera las características finales para entrenamiento:

```bash
python src/preprocess/build_features.py
```

El dataset final se guardará en **data/processed/**.

### Ejecutar el pipeline completo de Machine Learning

```bash
python run_all.py
```

Este comando ejecuta, en orden:

- Preprocesamiento
- Entrenamiento de modelos
- Optimización de hiperparámetros
- Evaluación final

### Si se desea ejecutar solo una parte del pipeline

#### Entrenamiento de un modelo en específico

```bash
python scripts/train.py --model random_forest
```

#### Validación cruzada

```bash
python scripts/crossvalidation.py --model random_forest
```

#### Evaluación del modelo entrenado

```bash
python scripts/evaluate.py --model random_forest
````

Los resultados se almacenan automáticamente en la carpeta **outputs/**.

---

#### Modelos disponibles

- random_forest
- decision_tree
- logistic_regression
- xgboost
