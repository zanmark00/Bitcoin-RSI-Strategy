# Bitcoin RSI Strategy

This repository contains the implementation of a Bitcoin trading strategy based on the Relative Strength Index (RSI). The strategy uses historical Bitcoin price data to calculate RSI and generate buy or sell signals based on specific thresholds.

## Project Description

The RSI strategy aims to identify potential buy and sell opportunities by calculating the relative strength index of Bitcoin's price over time. It signals a buy when the RSI is below 30 (indicating potentially oversold conditions) and a sell when the RSI is above 70 (indicating potentially overbought conditions).

## Libraries Required

To run the scripts in this repository, you need to have Python installed along with the following libraries:
- **yfinance**: To fetch historical market data.
- **pandas**: For data manipulation and analysis.
- **numpy**: For numerical operations.
- **matplotlib**: For plotting data and RSI indicators.

You can install these libraries using pip:
```bash
pip install yfinance pandas numpy matplotlib
