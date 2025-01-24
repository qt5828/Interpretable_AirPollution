# Interpretable Deep Learning Model for Integrated Regional Air Pollution Forecast

## Overview

This repository provides the implementation of the research paper **"Proposal and Analysis of Interpretable Deep Learning Model for Integrated Regional Air Pollution Forecast"**, which was **orally presented at the Korean Computer Congress (KCC 2024)**. The study proposes an interpretable deep learning model that predicts air pollution levels across six Asian countries by integrating advanced feature engineering and LIME-based interpretability techniques.

Key highlights of the research:
- Utilizes air pollution data from six Asian countries (South Korea, China, Japan, India, Turkey, Iran).
- Compares the performance of RNN, LSTM, Transformer, and CosSquareFormer models.
- Achieves high predictive performance and localized interpretability using LIME.

## Repository Contents

- **`AirPollution_integ_country_wise.py`**: Implementation of the country-wise model.
- **`AirPollution_integ_ours_without_country.py`**: Implementation of the integrated model without country-specific features.
- **`AirPollution_integ_ours.py`**: Implementation of the integrated model with country-specific features.
- **`explain_AirPollution.ipynb`**: Jupyter Notebook demonstrating the use of LIME for model interpretability.
- **`loss_utils.py`**: Utility functions for loss calculation.
- **`models.py`**: Implementation of the RNN, LSTM, Transformer, and CosSquareFormer models.

## Experimental Results

The experimental results comparing country-specific models (`Avg.`) and the integrated model (`Integ.`) for predicting three air pollutants are summarized as follows:

| Model          | PM2.5 MSE (Avg.) | PM2.5 MSE (Integ.) | PM10 MSE (Avg.) | PM10 MSE (Integ.) | NO2 MSE (Avg.) | NO2 MSE (Integ.) |
|----------------|------------------|-------------------|-----------------|------------------|----------------|-----------------|
| **RNN**        | 21.8251          | 22.8715           | 20.9693         | 18.7686          | 6.4408         | 5.3047          |
| **LSTM**       | 21.8889          | 24.6460           | 20.3036         | 20.2910          | 6.7802         | 5.5604          |
| **Transformer**| 20.6236          | 22.0197           | 19.3164         | 17.8260          | 6.1511         | 4.9853          |
| **CosSquareFormer** | 21.8285      | 23.3749           | 20.5620         | 19.2120          | 6.0564         | 5.3064          |

- **Findings**: The integrated models demonstrate performance comparable to or better than country-specific models, with improvements in certain cases (e.g., PM10 and NO2).

## Code Usage

### Prerequisites

Ensure the following Python libraries are installed:
- NumPy
- Pandas
- Scikit-learn
- PyTorch
- Matplotlib
- LIME

### Running the Models

1. **Country-wise Model**:
   ```bash
   python AirPollution_integ_country_wise.py
   ```

2. **Integrated Model (Without Country Features)**:
   ```bash
   python AirPollution_integ_ours_without_country.py
   ```

3. **Integrated Model (With Country Features)**:
   ```bash
   python AirPollution_integ_ours.py
   ```

### Model Interpretability

- Open the Jupyter Notebook:
  ```bash
  jupyter notebook explain_AirPollution.ipynb
  ```
- Follow the steps in the notebook to analyze and visualize LIME-based explanations for the model predictions.

### Dataset

The dataset contains daily air pollution and meteorological data from 2019 to 2023 for six countries. Ensure the dataset is placed in the appropriate directory or update the file paths in the scripts accordingly.

## Research Paper

For more details, please refer to the full paper:
- **Title**: Proposal and Analysis of Interpretable Deep Learning Model for Integrated Regional Air Pollution Forecast  
- **Authors**: Wooyeon Jo, Hyunsouk Cho  
- **Emails**: qt5828@ajou.ac.kr, hyunsouk@ajou.ac.kr  
- **Conference**: Korean Computer Congress (KCC 2024)  
- **Link**: [Download Paper](file:///Users/wooyeon/Desktop/iKnow/KCC2024/KCC%E1%84%8E%E1%85%AE%E1%86%AF%E1%84%91%E1%85%A1%E1%86%AB%E1%84%8B%E1%85%AD%E1%86%BC_%E1%84%8C%E1%85%A9%E1%84%8B%E1%85%AE%E1%84%8B%E1%85%A7%E1%86%AB.pdf)

## Citation

If you use this code or find it helpful in your research, please cite the following paper:
```
@article{조우연2024지역,
  title={지역 통합 대기 오염 예측을 위한 해석 가능한 딥러닝 모델의 제안과 분석},
  author={조우연 and 조현석},
  journal={한국정보과학회 학술발표논문집},
  pages={933--935},
  year={2024}
}
```

## Acknowledgments

This work was **orally presented at the Korean Computer Congress (KCC 2024)**. Special thanks to the authors of LIME for providing the interpretability framework used in this research.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
