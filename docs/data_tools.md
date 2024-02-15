# Data Tools for Economic Research

**[View Project on GitHub](https://github.com/Grow-Myelin/dataTools)**

## Technologies Used
- Python
- SQL

## Project Overview
Developed a suite of data analysis tools tailored for economic research, facilitating the efficient cleaning, processing, and ingestion of economic and financial API data into a SQLite database.

## Key Features
- Automated data cleaning and processing workflows.
- Ingestion of diverse economic and financial data into a unified database.
- Simplified access to clean, processed data for economic research and analysis.

## Imports
```python
import sqlite3
import requests
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import List, Optional
```

## Classes

### DataIngestor
```python
class DataIngestor:
    """
    A class to ingest and process financial data from a SQLite database.
    """

    def __init__(self, db_path: str, table_names: list, min_date: str):
        """
        Initialize the DataIngestor with database path, table names, and minimum date.

        Args:
            db_path (str): Path to the SQLite database.
            table_names (list): List of table names to process.
            min_date (str): Minimum date for filtering data.
        """
        self.db_path = db_path
        self.table_names = table_names
        self.min_date = min_date
```

### DataFetcher
```python
class DataFetcher:
    """
    A class to fetch and insert financial and economic data into a SQLite database.

    Attributes:
        db_path (str): The file path for the SQLite database.
    """

    def __init__(self, db_path: str) -> None:
        """
        Initialize the DataFetcher class.

        Args:
            db_path (str): The file path for the SQLite database.
        """
        self.db_path = db_path
```

## Methods

### DataFetcher Methods

#### create_table()
```python
def create_table(self, table_name: str, columns: List[str], is_stock: bool) -> None:
    """
    Create a table in the SQLite database.

    Args:
        table_name (str): The name of the table to create.
        columns (List[str]): A list of column names for the table.
        isStock (bool): Indicator of whether the data is stock data (True) or economic data (False).
    """
```

#### fetch_and_insert_stock_data()
```python
def fetch_and_insert_stock_data(
    self,
    stock_ticker: str,
    api_key: str,
    report_link: str,
    from_date: str,
    to_date: str,
) -> None:
    """
    Fetch stock data from an API and insert it into the database.

    Args:
        stock_ticker (str): The stock ticker symbol.
        api_key (str): The API key for authentication.
        report_link (str): The endpoint link for the API request.
        from_date (str): Start date for the data.
        to_date (str): End date for the data.
    """
```

#### fetch_and_insert_economic_data()
```python
def fetch_and_insert_economic_data(self, api_key: str, report_link: str) -> None:
    """
    Fetch economic data from an API and insert it into the database.

    Args:
        api_key (str): The API key for authentication.
        report_link (str): The endpoint link for the API request.
    """
```

#### _insert_data()
```python
def _insert_data(
    self,
    table_name: str,
    columns: List[str],
    data: List[dict],
    ticker: Optional[str] = None,
) -> None:
    """
    Insert data into a specified table in the database.

    Args:
        table_name (str): The name of the table to insert data into.
        columns (List[str]): The column names for data insertion.
        data (List[Dict]): The data to be inserted.
        ticker (Optional[str]): The stock ticker symbol. Defaults to None.
    """
```

### DataIngestor Methods

#### get_column_names()
```python
def get_column_names(self, conn: sqlite3.Connection, table_name: str) -> list:
    """
    Retrieve column names for a given table in the database.

    Args:
        conn (sqlite3.Connection): A connection object to the SQLite database.
        table_name (str): Name of the table to retrieve columns from.

    Returns:
        list: A list of column names.
    """
```

#### fetch_and_process_data()
```python
def fetch_and_process_data(self) -> list:
    """
    Fetch and process data from the database for each table.

    Returns:
        list: A list of processed pandas DataFrames.
    """
```

#### scale_data()
```python
def scale_data(self, dfs: list) -> pd.DataFrame:
    """
    Scale the data using StandardScaler.

    Args:
        dfs (list): List of pandas DataFrames to scale.

    Returns:
        pd.DataFrame: A DataFrame of scaled features.
    """
```

### utils

#### process_data()
```python
def process_data(db_path: str, table_names: list, min_date: str) -> tuple:
    """
    Process and scale data from the database.

    Args:
        db_path (str): Path to the SQLite database.
        table_names (list): List of table names to process.
        min_date (str): Minimum date for filtering data.

    Returns:
        tuple: A tuple containing list of DataFrames and scaled DataFrame.
    """
```

#### drop_table()
```python
def drop_table(db_path: str, table_name: str) -> None:
    """
    Drop a table from the database.

    Args:
        db_path (str): Path to the SQLite database.
        table_name (str): Name of the table to be dropped.
    """
```

#### fetch_stocks_data()
```python
def fetch_stocks_data(
    db_path: str, stocks: List[str], api_key: str, report_link: str
) -> None:
    """
    Fetch and insert stock data for multiple stocks into the database.

    Args:
        db_path (str): The file path for the SQLite database.
        stocks (List[str]): A list of stock ticker symbols.
        api_key (str): The API key for authentication.
        report_link (str): The endpoint link for the API request.
```

#### fetch_economic_data()
```python
def fetch_economic_data(db_path: str, table_names: List[str], api_key: str) -> None:
    """
    Fetch and insert economic data for multiple series into the database.

    Args:
        db_path (str): The file path for the SQLite database.
        table_names (list[str]): A list of economic data series identifiers.
        api_key (str): The API key for authentication.
    """
```

---
[Experience](experience.md) | [Education](education.md) | [Skills](skills.md) | [**Projects**](projects.md) | [Contact](contact.md)

---
[TI4 Combat Simulator](ti4_combat_simulator.md) | [Baseball Pitch Predictor](baseball_pitch_predictor.md) | [Probabalistic Programming Tools](prob_prog_tools.md) | [**Data Tools**](data_tools.md)