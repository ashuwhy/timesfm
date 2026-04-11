The datasets used in the TimesFM and TimesFM-ICF papers, as well as the specific datasets used for your project, are outlined below. 

### 1. Domain Adaptation & Fine-Tuning Datasets (Project Specific)
These datasets are used specifically in your `CryptoFM` term project to adapt the base model to the cryptocurrency domain via LoRA. Rather than being downloaded natively as static `.csv` files, they are fetched dynamically over the internet at runtime using the `yfinance` module.
- **BTC-USD (Primary Target)**: ~4,200 daily closing price candles acting as the continuous time-series baseline.
  - Link: [Yahoo Finance API via yfinance (BTC-USD)](https://finance.yahoo.com/quote/BTC-USD/)
- **ETH-USD & SOL-USD (Auxiliary Context)**: Optional daily pricing data used to give the system more crypto variance and correlation metrics during the In-Context Fine-Tuning testing.
  - Link (ETH): [Yahoo Finance API (ETH-USD)](https://finance.yahoo.com/quote/ETH-USD/)
  - Link (SOL): [Yahoo Finance API (SOL-USD)](https://finance.yahoo.com/quote/SOL-USD/)

---

### 2. Pretraining Datasets
These datasets were used by Google to build the foundational knowledge of the pre-trained TimesFM model (O(100B) timepoints). You do not have these files locally; they are compressed into the pre-trained network weights.
- **Google Trends**: Captures search interest for millions of queries over 15 years.
  - Link: [Google Trends Datastore or Explore Trends](https://trends.google.com/)
- **Wiki Pageviews**: Hourly views for all Wikimedia pages from 2012 to 2023.
  - Link: [Wikimedia Analytics Dumps](https://dumps.wikimedia.org/other/pageviews/) or [Pageview API](https://wikitech.wikimedia.org/wiki/Analytics/AQS/Pageviews)
- **M4 Forecasting Competition**: A diverse set of 100k time-series across multiple frequencies.
  - Link: [M4 Methods GitHub Repository](https://github.com/Mcompetitions/M4-methods)
- **Synthetic Data**: Generated using statistical processes (ARMA, seasonal patterns, trends) to fill gaps in frequency coverage.
  - Link: (Not hosted; details on generation parameters are in Appendix A.8 of the first paper).
- **LibCity & Favorita Sales**: Large-scale urban and retail forecasting data.
  - Link (LibCity): [LibCity Unified Library](https://libcity.ai/)
  - Link (Favorita): [Kaggle Favorita Grocery Sales](https://www.kaggle.com/c/favorita-grocery-sales-forecasting/data)

---

### 3. Evaluation Benchmarks (Zero-Shot)
These datasets were intentionally excluded from Google's pretraining to test the original model's ability to generalize to new data.
- **Monash Time Series Forecasting Archive**: A collection of 30 datasets (18 used in these papers) covering domains like finance, health, and transport.
  - Link: [ForecastingData.org](https://forecastingdata.org/)
- **ETT (Electricity Transformer Temperature)**: Standard long-horizon forecasting benchmarks (ETTh1, ETTh2, ETTm1, ETTm2).
  - Link: [ETDataset GitHub Repository](https://github.com/zhouhaoyi/ETDataset)
- **Darts Benchmarks**: A collection of univariate datasets with varied seasonalities and trends.
  - Link: [Darts Library Datasets](https://unit8co.github.io/darts/generated_api/darts.datasets.html)

---

### 4. Additional Datasets
- **Traffic & Electricity**: Commonly used long-term forecasting data.
  - Link: [UCI Machine Learning Repository - Electricity](https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014)
  - Link: [UCI Machine Learning Repository - Traffic](https://archive.ics.uci.edu/dataset/204/pems+sf)
