# Stock Trend Monitoring Bot

This project consists of a Python script for performing technical analysis on stock data and a Discord bot that uses this analysis to provide insights, alerts, and respond to commands.

## Overview

The `trends.py` script analyzes stock market trends using various technical indicators like Simple Moving Averages (SMAs), Relative Strength Index (RSI), Bollinger Bands, MACD, Stochastic Oscillator, and Williams %R. The Discord bot (`bot.py`) leverages this analysis to:

* Send automatic alerts to a Discord channel when significant SMA crossover signals (Golden Cross, Death Cross) are detected.
* Respond to commands in Discord for:
    * Manually checking all monitored tickers (`!run`).
    * Getting detailed technical analysis for a specific ticker (`!analyze [TICKER]`).
    * Adding a ticker to the watchlist (`!add [TICKER]`).
    * Removing a ticker from the watchlist (`!remove [TICKER]`).
    * Listing the current watchlist (`!list`).

## Key Features

**`trends.py` (Technical Analysis):**

* Configurable technical indicator parameters.
* Fetches stock data from Yahoo Finance (`yfinance`).
* Calculates SMAs, RSI, Bollinger Bands, MACD, Stochastic Oscillator, and Williams %R.
* Detects Golden Cross and Death Cross SMA crossover signals.
* Provides comprehensive technical analysis reports.
* Offers volume and momentum analysis functions.

**`bot.py` (Discord Bot):**

* Uses `discord.py` for Discord integration.
* Reads bot token and channel ID from a `.env` file.
* Monitors a configurable list of stock tickers.
* Sends scheduled alerts for SMA crossover signals.
* Responds to user commands for manual analysis and watchlist management.

## Prerequisites

* **Python 3.7+**
* **pip** (Python package installer)
* **Discord Bot Token:** You'll need to create a Discord bot application and obtain its token from the [Discord Developer Portal](https://discord.com/developers/applications).
* **Discord Server and Channel:** You need a Discord server where you have administrative privileges to invite the bot, and a specific channel where you want the bot to send alerts and receive commands.
* **`.env` File:** You will need to create a `.env` file to store your bot token and channel ID securely.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [repository_url]
    cd [repository_name]
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On macOS/Linux
    # venv\Scripts\activate   # On Windows
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Create a `.env` file:**
    In the root of your project directory, create a file named `.env` and add the following lines, replacing the placeholders with your actual values:
    ```env
    TOKEN=YOUR_DISCORD_BOT_TOKEN
    CHANNEL_ID=YOUR_DISCORD_CHANNEL_ID
    ```
5.  **Invite the Discord bot to your server:**
    Follow the instructions in the [Discord Developer Portal](https://discord.com/developers/applications) to generate an invite link for your bot and add it to your Discord server. Ensure the bot has the necessary permissions (e.g., "View Channels," "Send Messages").

## Usage

1.  **Run the Discord bot:**
    ```bash
    python bot.py
    ```
    The bot will log in to Discord and start monitoring the tickers in the `TICKERS` list defined in `bot.py`.

2.  **Interact with the bot in your Discord channel using the following commands:**
    * `!test`: Triggers an immediate check for SMA crossover trends for all tickers in the watchlist.
    * `!analyze [TICKER]`: Provides a detailed technical analysis for the specified ticker symbol.
    * `!add [TICKER]`: Adds the given ticker symbol to the watchlist.
    * `!remove [TICKER]`: Removes the given ticker symbol from the watchlist.
    * `!list`: Displays the current list of tickers being monitored.

The bot will also automatically run a trend check every hour (60 minutes) and post alerts for significant SMA crossover signals.

## Configuration

### `trends.py`

You can customize the technical analysis parameters by modifying the `TechnicalAnalysisConfig` class in `trends.py`:

* `sma_short`: Period for the short-term Simple Moving Average (default: 20).
* `sma_long`: Period for the long-term Simple Moving Average (default: 50).
* `rsi_period`: Period for the Relative Strength Index (default: 14).
* `rsi_overbought`: RSI level considered overbought (default: 70).
* `rsi_oversold`: RSI level considered oversold (default: 30).
* `bb_period`: Period for the Bollinger Bands (default: 20).
* `bb_std`: Number of standard deviations for the Bollinger Bands (default: 2).
* `volume_threshold`: Threshold for considering volume as high (default: 1.5 times the average).
* `data_period`: Time period for fetching historical data (default: "3mo").
* `cache_minutes`: Time in minutes to cache fetched data (default: 5).
* `macd_fast`, `macd_slow`, `macd_signal`: Parameters for MACD calculation.
* `stoch_k`, `stoch_d`: Parameters for Stochastic Oscillator calculation.
* `williams_period`: Period for Williams %R calculation.

### `bot.py`

* `TOKEN`: Your Discord bot token (set in the `.env` file).
* `CHANNEL_ID`: The ID of the Discord channel where the bot should send alerts and receive commands (set in the `.env` file).
* `TICKERS`: A list of stock ticker symbols that the bot will monitor for automatic alerts. You can modify this list directly in the `bot.py` file.
* `@tasks.loop(minutes=60)`: This decorator controls the interval at which the automatic trend check runs. You can adjust the `minutes` value to change the frequency.

## Contributing

Bug Reports:

If you encounter any bugs, please help us by reporting them! When submitting a bug report, please include as much information as possible. Useful details include:

A clear and concise description of the bug. What were you trying to do, and what happened instead?

Steps to reproduce the bug. If possible, provide a step-by-step guide on how to trigger the issue.

The version of Python you are using.

The version of the bot (if you've made any changes).

Any relevant error messages or traceback. Please use code blocks (```) to format error messages for readability.

Your operating system.

You can submit bug reports by opening a new issue on [GitHub Issues] (https://github.com/Str8Rogue16/StockBot/issues) with the label "bug".

Feature Requests:

We welcome suggestions for new features and improvements! If you have an idea that you think would enhance the Stock Trend Monitoring Bot, please let us know. When submitting a feature request, please:

Provide a clear and descriptive title for your suggestion.

Explain the problem or need that your feature request addresses. Why do you think this feature would be valuable?

Describe the feature in detail. How would it work? What would the user experience be like?

Include any use cases or scenarios where this feature would be beneficial.

Feel free to suggest multiple small, related features as separate requests.

You can submit feature requests by opening a new issue on [GitHub Issues] (https://github.com/Str8Rogue16/StockBot/issues) with the label "enhancement".

Documentation:

Clear and comprehensive documentation is crucial for any project. We appreciate any help in improving the documentation for the Stock Trend Monitoring Bot. You can contribute to documentation by:

Improving the README.md file. Are there any sections that are unclear or could be expanded?

Adding more detailed explanations of the bot's features and commands.

Creating tutorials or guides for specific use cases.

Adding comments to the code to clarify its functionality.

Translating the documentation into other languages.

If you'd like to contribute to the documentation, you can either submit a pull request with your changes or open an issue on GitHub Issues with the label "documentation" to discuss potential improvements.

Testing:

Testing is essential to ensure the stability and reliability of the Stock Trend Monitoring Bot. You can help us by:

Running the bot and using its features. Do you encounter any unexpected behavior?

Trying different stock tickers and market conditions. Does the analysis work as expected?

Testing new features or bug fixes that are being developed (often mentioned in issues or pull requests).

Writing unit tests for the trends.py or bot.py code (if you are familiar with Python testing frameworks like pytest or unittest).

If you find any issues during testing, please report them as a bug report as described above. If you are interested in contributing more actively to testing, feel free to reach out by opening an issue on [GitHub Issues] (https://github.com/Str8Rogue16/StockBot/issues) with the label "testing".

## License

MIT License

## Support

[Optional: Add information on how users can get support or contact you.]
