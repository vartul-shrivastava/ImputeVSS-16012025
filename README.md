![Impute-VSS Dashboard](images/header.png "Impute-VSS Dashboard")

# Impute-VSS: A GUI-based Visualizer and Summarizer Suite for Data Imputation with Gen-AI enabled Interpretation

## Overview
Impute-VSS represents a web-based solution engineered to address the intricate challenges of data imputation. This tool offers a unified interface combining multiple imputation techniques, generative AI-based insights, and modular pipeline capabilities. Designed for both researchers and practitioners, Impute-VSS enables an analytical exploration of missing data, comparative assessments of imputation methods, and detailed visual diagnostics to support data-driven decision-making.

---

## Features

1. **Comprehensive Imputation Techniques**:
   - **Complete Case Analysis (CCA)**: Analyze datasets by excluding incomplete records.
   - **Univariate Imputation**: Employ statistical techniques like Mean, Mode, Median, Constant Value, and Random Sample Imputation for simplicity and speed.
   - **Multivariate Imputation**: Integrate advanced methods such as K-Nearest Neighbors (KNN) and Multiple Imputation by Chained Equations (MICE) for datasets with complex interdependencies.

2. **Generative AI Support**:
   - Leverage locally installed **Ollama AI models** for automated, context-aware insights.
   - Utilize **customizable prompt templates** for tailored recommendations specific to domain requirements.

3. **Real-time Visual Feedback**:
   - **Interactive Charts**: Heatmaps, KDE Plots, Histograms, Correlation Matrices, and Box Plots to visualize imputation outcomes dynamically.
   - **Comparative Analysis**: Instant updates showcasing differences across techniques and their statistical significance.

4. **Imputation Pipeline Management**:
   - Modular workflow creation and export in the proprietary `.impvss` format for consistent replication and sharing.
   - Workflow modularity ensures efficient reuse of imputation strategies without compromising dataset privacy.

5. **Cross-Platform Compatibility**:
   - Fully web-based, compatible with desktops, tablets, and mobile devices, ensuring seamless accessibility across platforms.

6. **Statistical Metrics and Comparisons**:
   - Evaluate methods based on Kernel Density Estimation (KDE) overlap, Skewness, Kurtosis, Kolmogorov-Smirnov Statistic, and Kullback-Leibler Divergence for robust statistical insights.

---

## File Structure
```
├── static/
│   ├── css/          # Stylesheet
│   ├── images/       # Backgrounds and logos
│   ├── js/           # JavaScript file
├── templates/        # HTML template
├── app.py            # Flask application
├── requirements.txt  # Dependencies
├── README.md         # Documentation
```

## Support
- **Documentation**: [Impute-VSS Docs](https://vartul-shrivastava.github.io/ImputeVSS-documentation-github.io/)
- **Email**: 
  - Vartul Shrivastava: vartul.shrivastava@gmail.com
  - Prof. Shekhar Shukla: shekhars@iimidr.ac.in

## License
Impute-VSS is licensed under the MIT License. See the LICENSE file for more details.