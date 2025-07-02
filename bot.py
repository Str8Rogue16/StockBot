import discord
from discord.ext import tasks, commands
from trends import check_sma_crossover_fixed, get_comprehensive_analysis
import asyncio
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

TOKEN = os.getenv('TOKEN')
CHANNEL_ID_STR = int(os.getenv('CHANNEL_ID'))
TICKERS = ["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN", "CRWD", "PLTR", "DFDV", "SUPX", "CSCO"]

# Validate required environment variables
if not TOKEN:
    print("❌ Error: DISCORD_TOKEN not found in .env file")
    print("Make sure your .env file exists and contains: DISCORD_TOKEN=your_token_here")
    exit(1)

if not CHANNEL_ID_STR:
    print("❌ Error: CHANNEL_ID not found in .env file")
    print("Make sure your .env file contains: CHANNEL_ID=859570733486571553")
    exit(1)

try:
    CHANNEL_ID = int(CHANNEL_ID_STR)
except ValueError:
    print(f"❌ Error: CHANNEL_ID must be a number, got: {CHANNEL_ID_STR}")
    exit(1)

# Use commands.Bot for better command handling
intents = discord.Intents.default()
intents.message_content = True  # Enable message content intent for commands
bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def on_ready():
    print(f"✅ Logged in as {bot.user}")
    
    # Test channel access immediately
    try:
        channel = await bot.fetch_channel(CHANNEL_ID)
        print(f"✅ Successfully accessed channel: {channel.name}")
        
        # Send startup message with current time
        startup_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        embed = discord.Embed(
            title="🤖 Stock Bot Online",
            description=f"Monitoring {len(TICKERS)} tickers",
            color=0x00ff00,
            timestamp=datetime.now()
        )
        embed.add_field(name="Tickers", value=", ".join(TICKERS), inline=False)
        embed.add_field(name="Check Interval", value="60 minutes", inline=True)
        embed.add_field(name="Commands", value="`!run` - Manual check\n`!analyze [TICKER]` - Detailed analysis", inline=True)
        
        await channel.send(embed=embed)
        
        # Start the loop only if channel access works
        trend_check.start()
        
    except discord.errors.Forbidden as e:
        print(f"❌ Permission Error: {e}")
        print("Check if bot has 'View Channels' and 'Send Messages' permissions")
        return
    except discord.errors.NotFound as e:
        print(f"❌ Channel not found: {e}")
        print("Check if CHANNEL_ID is correct")
        return
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return

@tasks.loop(minutes=60)
async def trend_check():
    try:
        channel = bot.get_channel(CHANNEL_ID)
        if not channel:
            channel = await bot.fetch_channel(CHANNEL_ID)
        
        print(f"📊 Running trend check at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Track signals to send
        signals_to_send = []
        
        # Process tickers with delay to avoid rate limiting
        for i, ticker in enumerate(TICKERS):
            try:
                print(f"Checking {ticker} ({i+1}/{len(TICKERS)})...")
                trend = check_sma_crossover_fixed(ticker)
                
                # Only send significant signals (crossovers, not regular trends)
                if trend and ("GOLDEN CROSS" in trend or "Death Cross" in trend):
                    signals_to_send.append(trend)
                    print(f"🚨 SIGNAL: {trend}")
                elif trend:
                    print(f"ℹ️ {trend}")
                
                # Small delay to avoid overwhelming the API
                if i < len(TICKERS) - 1:
                    await asyncio.sleep(1)
                    
            except Exception as e:
                print(f"Error checking trend for {ticker}: {e}")
        
        # Send all signals in one message if any found
        if signals_to_send:
            signal_embed = discord.Embed(
                title="🚨 Trading Signals Detected!",
                color=0xff6b00,
                timestamp=datetime.now()
            )
            
            for signal in signals_to_send:
                # Extract ticker from signal
                ticker = signal.split(":")[0].replace("🚀", "").replace("⚠️", "").strip()
                signal_embed.add_field(
                    name=ticker,
                    value=signal.split(":", 1)[1].strip(),
                    inline=False
                )
            
            await channel.send(embed=signal_embed)
            print(f"📤 Sent {len(signals_to_send)} signals to Discord")
        else:
            print("📭 No significant signals to report")
                
    except discord.errors.Forbidden:
        print("❌ Lost channel permissions during runtime")
        trend_check.stop()
    except Exception as e:
        print(f"❌ Error in trend_check: {e}")

@trend_check.before_loop
async def before_trend_check():
    await bot.wait_until_ready()

# Command: Manual test of all tickers
@bot.command(name='run')
async def manual_run(ctx):
    """Manually check all tickers for trends"""
    await ctx.send("🔄 Running manual trend check...")
    
    # Create embed for results
    embed = discord.Embed(
        title="📊 Manual Trend Check",
        color=0x0099ff,
        timestamp=datetime.now()
    )
    
    results = []
    for ticker in TICKERS:
        try:
            trend = check_sma_crossover_fixed(ticker)
            if trend:
                # Truncate long messages for embed
                trend_short = trend[:100] + "..." if len(trend) > 100 else trend
                results.append(f"**{ticker}**: {trend_short}")
        except Exception as e:
            results.append(f"**{ticker}**: ❌ Error - {str(e)[:50]}")
    
    # Split results into chunks if too long
    chunk_size = 10
    for i in range(0, len(results), chunk_size):
        chunk = results[i:i+chunk_size]
        embed.add_field(
            name=f"Results ({i+1}-{min(i+chunk_size, len(results))})",
            value="\n".join(chunk),
            inline=False
        )
    
    await ctx.send(embed=embed)

# Command: Detailed analysis of specific ticker
@bot.command(name='analyze')
async def analyze_ticker(ctx, ticker: str = None):
    """Get detailed technical analysis for a specific ticker"""
    if not ticker:
        await ctx.send("❌ Please provide a ticker symbol. Example: `!analyze AAPL`")
        return
    
    ticker = ticker.upper()
    await ctx.send(f"🔍 Analyzing {ticker}...")
    
    try:
        print(f"📊 Calling get_comprehensive_analysis for {ticker}")
        analysis = get_comprehensive_analysis(ticker)
        print(f"✅ Analysis received: {analysis[:100]}...")

    # Check if analysis is too long for Discord embed
        if len(analysis) > 4096:
            analysis = analysis[:4090] + "..."   
        
        # Create embed for analysis
        embed = discord.Embed(
            title=f"📈 Technical Analysis: {ticker}",
            description=analysis,
            color=0x00ff00,
            timestamp=datetime.now()
        )
        
        await ctx.send(embed=embed)
        
    except Exception as e:
        error_message=f"❌ Error analyzing {ticker}: {str(e)}"
        print(f"❌ {error_message}")
        await ctx.send(error_message)
        

# Command: Add ticker to watchlist
@bot.command(name='add')
async def add_ticker(ctx, ticker: str = None):
    """Add a ticker to the watchlist"""
    if not ticker:
        await ctx.send("❌ Please provide a ticker symbol. Example: `!add NVDA`")
        return
    
    ticker = ticker.upper()
    if ticker not in TICKERS:
        TICKERS.append(ticker)
        await ctx.send(f"✅ Added {ticker} to watchlist. Now monitoring {len(TICKERS)} tickers.")
    else:
        await ctx.send(f"ℹ️ {ticker} is already in the watchlist.")

# Command: Remove ticker from watchlist
@bot.command(name='remove')
async def remove_ticker(ctx, ticker: str = None):
    """Remove a ticker from the watchlist"""
    if not ticker:
        await ctx.send("❌ Please provide a ticker symbol. Example: `!remove NVDA`")
        return
    
    ticker = ticker.upper()
    if ticker in TICKERS:
        TICKERS.remove(ticker)
        await ctx.send(f"✅ Removed {ticker} from watchlist. Now monitoring {len(TICKERS)} tickers.")
    else:
        await ctx.send(f"ℹ️ {ticker} is not in the watchlist.")

# Command: Show current watchlist
@bot.command(name='list')
async def show_watchlist(ctx):
    """Show current ticker watchlist"""
    embed = discord.Embed(
        title="📋 Current Watchlist",
        description=f"Monitoring {len(TICKERS)} tickers",
        color=0x9932cc
    )
    
    # Group tickers for better display
    ticker_chunks = [TICKERS[i:i+5] for i in range(0, len(TICKERS), 5)]
    for i, chunk in enumerate(ticker_chunks):
        embed.add_field(
            name=f"Group {i+1}",
            value=" • ".join(chunk),
            inline=True
        )
    
    await ctx.send(embed=embed)

# Error handling for commands
@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.CommandNotFound):
        print(f"❓ Unknown command attempted: {ctx.message.content}")
        return  # Ignore unknown commands
    elif isinstance(error, commands.MissingRequiredArgument):
        await ctx.send(f"❌ Missing required argument for `{ctx.command.name}` command")
        print(f"❌ Missing argument error: {error}")
    else:
        await ctx.send(f"❌ An error occurred: {str(error)}")
        print(f"❌ Command error in {ctx.command.name}: {error}")

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
    
    # Debug: Print all messages that start with !
    if message.content.startswith('!'):
        print(f"📝 Command received: '{message.content}' from {message.author}")
    
    # This is CRITICAL - without this, commands won't work!
    await bot.process_commands(message)


bot.run(TOKEN)
