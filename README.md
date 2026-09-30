# Pizza MDW & Predictive Intelligence

A four-tab Streamlit decision support dashboard for pizza sales. It creates a 2025 synthetic sales source, builds a Pandas star schema, supports OLAP drill-down, and trains regression and decision tree models for interactive prediction.

## Requirements

- Python 3.10 or newer
- Windows PowerShell, macOS Terminal, or a Linux shell

## Set up and run

From the `pizza_mdw_platform` directory:

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

Run the smoke checks with `python -m unittest discover -s tests -v`.

## Data and warehouse

On first run, the app writes 3,000 reproducible 2025 line items to `data/pizza_sales.csv`. Delete that file to regenerate it, or replace it with your own CSV containing `order_id`, `order_date`, `order_time`, `pizza_category`, `pizza_size`, `pizza_name`, `quantity`, `unit_price`, and `total_price`. Dates use `YYYY-MM-DD`; times use `HH:MM:SS`. The app validates that line totals equal quantity times unit price. Restart Streamlit after replacing the CSV so its in-memory cache reloads the file.

The Pandas warehouse contains `Fact_Sales`, `Dim_Pizza`, `Dim_Time`, and `Dim_Date`. Each fact row is one sales line. Weekend Rush selects Saturday and Sunday lunch/dinner periods; Peak Hours selects lunch and dinner across all days. Revenue, margin, and order counts respond to sidebar filters. Model labs use the filtered rows when there are enough examples and otherwise explain that they use the full dataset.

## Modeling notes

- Regression predicts line `total_price` from quantity, unit price, hour, size, and category. The benchmark uses a fixed 75/25 train/test split. Since the target is quantity times unit price, the arithmetic basket total is shown alongside model inference.
- Classification uses Budget below $20, Standard from $20 through $50, and Premium above $50. The tree uses a stratified 75/25 train/test split. ROC curves use one-versus-rest class probabilities.
- Margin rates are synthetic assumptions of 65–75%, so margin analysis is illustrative rather than accounting data.
