# Voice Commands Classification — Robustness against Adversarial Attacks

Proyecto de curso de Deep Learning — Universidad de Antioquia

**Autores:** Michael Steven Ruiz Palacio, Marlon Giraldo

## Descripción

Clasificación de comandos de voz (35 clases) con robustez ante ataques adversariales.
Competencia Kaggle: [Voice Commands Classification 2026](https://www.kaggle.com/competitions/voice-commands-classification-2026)

## Estructura del repositorio

| Archivo | Descripción |
|---------|-------------|
| `01 - exploración de datos.ipynb` | Análisis exploratorio de datos con carga de metadatos reales |
| `02 - preprocesado.ipynb` | Extracción de características (MFCC, Mel-spectrogramas) y data augmentation |
| `03 - arquitectura de linea de base.ipynb` | CNN baseline sobre Mel-spectrogramas con BatchNorm |
| `04 - arquitectura crnn.ipynb` | CRNN: CNN 2D + BiLSTM para modelado temporal sobre mel-espectrogramas |
| `05 - arquitectura transformer.ipynb` | Spectrogram Transformer con RoPE y atención global |
| `06 - evaluación de robustez.ipynb` | Evaluación comparativa de los 3 modelos en test adversarial |
| `07 - optimización de hiperparámetros.ipynb` | LR Range Test y selección sistemática de hiperparámetros |
| `models.py` | Definiciones unificadas de los 3 modelos y transformaciones |
| `INFORME_PROYECTO.PDF` | Informe ejecutivo del proyecto |
| `ENTREGA1.PDF` | Primera entrega del proyecto |
| `train_metadata.csv` | Metadatos de entrenamiento generados por EDA |
| `test_metadata.csv` | Metadatos de test generados por EDA |
| `best_cnn_baseline.pth` | Pesos del modelo CNN baseline entrenado |

## Reproducibilidad en Google Colab

Cada notebook incluye detección automática de Colab. Al ejecutar en Colab:

1. Se autentica con kagglehub (requiere token de Kaggle: `~/.kaggle/kaggle.json`)
2. Descarga el dataset de la competencia automáticamente
3. Ejecuta el pipeline completo

## Video de presentación

[Enlace al video en YouTube](https://youtube.com) <!-- Reemplazar con el enlace real -->

## Dataset

- **Fuente:** Google Speech Commands v0.02 vía Kaggle
- **Clases:** 35 comandos de voz
- **Train:** 95,246 muestras
- **Test adversarial:** 10,576 muestras con perturbaciones
- **Formato:** Arrays NumPy (`.npy`), mono, 16kHz, float32
