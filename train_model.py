# train_model.py
import pandas as pd
import numpy as np
import pickle, json, os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

CSV_PATH = "SuperMarket Analysis (2).csv"   # put your CSV in same folder or give full path
OUT_DIR = "."

def find_date_and_sales_cols(df):
    # Find likely columns for date and sales
    date_col = None
    sales_col = None
    for c in df.columns:
        lc = c.lower().strip()
        if lc == "date" or "date" in lc:
            date_col = c
        if lc == "sales" or lc == "total" or "amount" in lc:
            # prefer exact 'Sales'
            if "sales" in lc:
                sales_col = c
            elif sales_col is None:
                sales_col = c
    return date_col, sales_col

def main():
    assert os.path.exists(CSV_PATH), f"CSV not found: {CSV_PATH}"
    df = pd.read_csv(CSV_PATH)
    date_col, sales_col = find_date_and_sales_cols(df)
    if date_col is None or sales_col is None:
        print("Could not find Date or Sales columns automatically. Columns found:")
        print(df.columns.tolist())
        return

    print("Using", date_col, "as date and", sales_col, "as sales")

    # parse dates
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col, sales_col])
    df[sales_col] = pd.to_numeric(df[sales_col], errors='coerce')
    df = df.dropna(subset=[sales_col])

    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    monthly = df.groupby(['year','month'], as_index=False)[sales_col].sum().rename(columns={sales_col:'monthly_sales'})
    monthly = monthly.sort_values(['year','month']).reset_index(drop=True)
    if len(monthly) < 6:
        print("Warning: only", len(monthly), "monthly datapoints found — predictions may be unreliable.")

    # create simple time index
    monthly['ym_index'] = np.arange(len(monthly))

    X = monthly[['ym_index']]
    y = monthly['monthly_sales']

    model = LinearRegression()
    model.fit(X, y)
    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)
    print("Trained LinearRegression on", len(X), "months. MAE:", mae)

    # save
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "monthly_model.pkl"), "wb") as f:
        pickle.dump(model, f)
    monthly.to_csv(os.path.join(OUT_DIR, "monthly_aggregated.csv"), index=False)
    meta = {
        "start_year": int(monthly.loc[0,'year']),
        "start_month": int(monthly.loc[0,'month']),
        "n_months": int(len(monthly))
    }
    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump(meta, f)
    print("Saved monthly_model.pkl, monthly_aggregated.csv, meta.json to", OUT_DIR)

if __name__ == "__main__":
    main()
