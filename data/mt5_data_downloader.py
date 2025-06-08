#!/usr/bin/env python3
"""
MT5 Data Downloader

This module provides functionality to connect to the MetaTrader 5 terminal
and download historical price data for various financial instruments.
"""
import logging
import os
from datetime import datetime

import MetaTrader5 as mt5
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class MT5DataDownloader:
    """
    A class to handle the connection and data downloading from MetaTrader 5.
    """

    def __init__(
        self,
        login: int | None = None,
        password: str | None = None,
        server: str | None = None,
    ):
        """
        Initializes the connection to the MetaTrader 5 terminal.

        Args:
            login (Optional[int]): The account number.
            password (Optional[str]): The password.
            server (Optional[str]): The server name.
        """
        initialize_params = {}
        if login:
            initialize_params["login"] = int(login)
        if password:
            initialize_params["password"] = password
        if server:
            initialize_params["server"] = server

        if not mt5.initialize(**initialize_params):
            logging.error("initialize() failed, error code =", mt5.last_error())
            raise ConnectionError("Failed to initialize MetaTrader 5 connection.")

        logging.info("MetaTrader 5 connection initialized successfully.")
        terminal_info = mt5.terminal_info()
        if terminal_info:
            logging.info(
                f"Connected to {terminal_info.name} on account {mt5.account_info().login}"
            )
        logging.info(f"MetaTrader 5 version: {mt5.version()}")

    def disconnect(self):
        """
        Shuts down the connection to the MetaTrader 5 terminal.
        """
        mt5.shutdown()
        logging.info("MetaTrader 5 connection shut down.")

    def get_symbol_info(self, symbol: str):
        """
        Gets information about a specific symbol.

        Args:
            symbol (str): The symbol to get information for.

        Returns:
            A structure with symbol properties or None if the symbol is not found.
        """
        info = mt5.symbol_info(symbol)
        if info is None:
            logging.error(f"Symbol {symbol} not found.")
            return None

        logging.info(f"Symbol {symbol} info: {info}")
        return info

    def download_data(
        self, symbol: str, timeframe, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """
        Downloads historical data for a given symbol and timeframe, handling large date ranges by chunking.

        Args:
            symbol (str): The financial instrument to download data for (e.g., "EURUSD").
            timeframe: The timeframe to use (e.g., mt5.TIMEFRAME_M1, mt5.TIMEFRAME_H1).
            start_date (datetime): The start date for the data download.
            end_date (datetime): The end date for the data download.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the historical data, or an empty DataFrame if failed.
        """
        logging.info(
            f"Attempting to download data for {symbol} from {start_date} to {end_date} with timeframe {self._timeframe_to_str(timeframe)}."
        )

        all_dataframes = []
        current_start = start_date

        while current_start < end_date:
            # Set the end of the chunk to one year later or the final end_date, whichever is smaller
            chunk_end = min(
                current_start.replace(year=current_start.year + 1), end_date
            )
            logging.info(f"  Downloading chunk from {current_start} to {chunk_end}")

            try:
                rates = mt5.copy_rates_range(
                    symbol, timeframe, current_start, chunk_end
                )

                if rates is None or len(rates) == 0:
                    logging.warning(
                        f"  No rates received for chunk. Error code: {mt5.last_error()}"
                    )
                else:
                    # Convert the structured array directly to DataFrame
                    chunk_df = pd.DataFrame(rates)
                    all_dataframes.append(chunk_df)
                    logging.info(f"  Received {len(rates)} records for this chunk.")

            except Exception as e:
                logging.error(
                    f"  An error occurred during chunk download for {symbol}: {e}"
                )

            # Move to the next chunk
            current_start = chunk_end

        if not all_dataframes:
            logging.error(
                f"Failed to download any data for {symbol} in the given range."
            )
            return pd.DataFrame()

        # Concatenate all chunk DataFrames
        df = pd.concat(all_dataframes, ignore_index=True)

        # Convert time in seconds into a datetime object
        df["time"] = pd.to_datetime(df["time"], unit="s")
        # Remove duplicates that might occur at chunk boundaries
        df = (
            df.drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)
        )

        logging.info(
            f"Successfully downloaded a total of {len(df)} records for {symbol}."
        )
        return df

    def download_mtf_data(
        self, symbol: str, timeframes: list, start_date: datetime, end_date: datetime
    ) -> dict[str, pd.DataFrame]:
        """
        Downloads historical data for a given symbol across multiple timeframes.

        Args:
            symbol (str): The financial instrument to download data for.
            timeframes (list): A list of MT5 timeframe constants (e.g., [mt5.TIMEFRAME_M1, mt5.TIMEFRAME_H1]).
            start_date (datetime): The start date for the data download.
            end_date (datetime): The end date for the data download.

        Returns:
            dict[str, pd.DataFrame]: A dictionary where keys are timeframe strings
                                     (e.g., "M1", "H1") and values are the corresponding
                                     DataFrames.
        """
        mtf_data = {}
        for tf in timeframes:
            tf_str = self._timeframe_to_str(tf)
            logging.info(f"--- Downloading {tf_str} data for {symbol} ---")
            df = self.download_data(symbol, tf, start_date, end_date)
            if not df.empty:
                mtf_data[tf_str] = df
            else:
                logging.warning(
                    f"No data downloaded for {symbol} on {tf_str} timeframe."
                )
        return mtf_data

    def _timeframe_to_str(self, timeframe: int) -> str:
        """Converts an MT5 timeframe constant to a string."""
        mapping = {
            mt5.TIMEFRAME_M1: "M1",
            mt5.TIMEFRAME_M5: "M5",
            mt5.TIMEFRAME_M15: "M15",
            mt5.TIMEFRAME_M30: "M30",
            mt5.TIMEFRAME_H1: "H1",
            mt5.TIMEFRAME_H4: "H4",
            mt5.TIMEFRAME_D1: "D1",
            mt5.TIMEFRAME_W1: "W1",
            mt5.TIMEFRAME_MN1: "MN1",
        }
        return mapping.get(timeframe, f"TF_{timeframe}")

    def save_to_file(
        self, df: pd.DataFrame, symbol: str, timeframe_str: str, path: str = "data/raw"
    ):
        """
        Saves the DataFrame to a file (Parquet format recommended).

        Args:
            df (pd.DataFrame): The DataFrame to save.
            symbol (str): The symbol of the instrument.
            timeframe_str (str): A string representation of the timeframe (e.g., "M1", "H1").
            path (str, optional): The directory to save the file in. Defaults to 'data/raw'.
        """
        if df.empty:
            logging.warning("DataFrame is empty. Nothing to save.")
            return

        if not os.path.exists(path):
            os.makedirs(path)
            logging.info(f"Created directory: {path}")

        filename = f"{symbol}_{timeframe_str}.parquet"
        filepath = os.path.join(path, filename)

        try:
            df.to_parquet(filepath, index=False)
            logging.info(f"Data successfully saved to {filepath}")
        except Exception as e:
            logging.error(f"Failed to save data to {filepath}: {e}")


if __name__ == "__main__":
    # Example usage:
    # It's recommended to use environment variables for credentials
    mt5_login = os.getenv("MT5_LOGIN")
    mt5_password = os.getenv("MT5_PASSWORD")
    mt5_server = os.getenv("MT5_SERVER")

    # Ensure login is an int if it exists
    if mt5_login:
        mt5_login = int(mt5_login)

    downloader = MT5DataDownloader(
        login=mt5_login, password=mt5_password, server=mt5_server
    )

    # Define parameters
    symbol_to_download = "EURUSD"
    timeframe_to_use = mt5.TIMEFRAME_M1
    start_dt = datetime(2023, 1, 1)
    end_dt = datetime(2023, 1, 31)

    # Check if symbol is available
    if downloader.get_symbol_info(symbol_to_download):
        # Example of new MTF download
        mtf_timeframes = [mt5.TIMEFRAME_M5, mt5.TIMEFRAME_H1]
        mtf_data_dict = downloader.download_mtf_data(
            symbol_to_download, mtf_timeframes, start_dt, end_dt
        )

        for tf_str, data_df in mtf_data_dict.items():
            if not data_df.empty:
                downloader.save_to_file(data_df, symbol_to_download, tf_str)

    # Disconnect
    downloader.disconnect()
