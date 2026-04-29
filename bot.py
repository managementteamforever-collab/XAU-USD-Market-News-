"""
Gold Market AI News Agent - Telegram Bot
Monitors high-impact gold market news and alerts via Telegram
"""

import os
import asyncio
import json
import time
import logging
from datetime import datetime, timezone
import httpx
from telegram import Bot
from telegram.constants import ParseMode

# ─── CONFIG ────────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "YOUR_BOT_TOKEN_HERE")
TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID",   "YOUR_CHAT_ID_HERE")
ANTHROPIC_API_KEY  = os.environ.get("ANTHROPIC_API_KEY",  "YOUR_ANTHROPIC_KEY_HERE")
CHECK_INTERVAL     = 900   # seconds (15 minutes between news sweeps)
SENT_CACHE_FILE    = "sent_news_cache.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ─── NEWS SOURCES (Free RSS feeds) ─────────────────────────────────────────────
RSS_FEEDS = [
    # Forex / Macro
    "https://www.forexlive.com/feed/news",
    "https://www.fxstreet.com/rss/news",
    # Commodities
    "https://feeds.bloomberg.com/markets/news.rss",
    "https://www.kitco.com/rss/KitcoNews.xml",
    # Central Bank / Economic
    "https://www.investing.com/rss/news_25.rss",   # Gold news
    "https://www.marketwatch.com/rss/realtimeheadlines",
]

# Keywords that could move gold — if headline has NONE of these, skip immediately
GOLD_RELEVANT_KEYWORDS = [
    # Central Banks
    "fed", "federal reserve", "fomc", "powell", "rate cut", "rate hike",
    "interest rate", "ecb", "boe", "bank of japan", "pboc", "central bank",
    "warsh", "monetary policy", "dot plot", "hawkish", "dovish",
    # Inflation / Economy
    "cpi", "inflation", "pce", "gdp", "recession", "stagflation",
    "nonfarm", "nfp", "jobs report", "unemployment", "payroll",
    # Geopolitics
    "war", "iran", "strait of hormuz", "russia", "ukraine", "conflict",
    "sanction", "nuclear", "middle east", "attack", "missile",
    # Dollar / DXY
    "dollar", "dxy", "usd", "treasury", "yield", "10-year",
    # Gold specific
    "gold", "xau", "bullion", "precious metal", "safe haven",
    "comex", "spot gold", "gold price",
    # Market stress
    "crash", "collapse", "crisis", "default", "banking",
    "emergency", "black swan",
]

# ─── CACHE ─────────────────────────────────────────────────────────────────────
def load_cache() -> set:
    try:
        with open(SENT_CACHE_FILE, "r") as f:
            data = json.load(f)
            return set(data.get("sent", []))
    except (FileNotFoundError, json.JSONDecodeError):
        return set()

def save_cache(cache: set):
    # Keep only last 500 to avoid unbounded growth
    recent = list(cache)[-500:]
    with open(SENT_CACHE_FILE, "w") as f:
        json.dump({"sent": recent}, f)

# ─── RSS FETCHER ────────────────────────────────────────────────────────────────
async def fetch_rss(url: str, client: httpx.AsyncClient) -> list[dict]:
    """Fetch RSS feed and return list of {title, link, published} dicts."""
    try:
        resp = await client.get(url, timeout=10, follow_redirects=True)
        resp.raise_for_status()
        text = resp.text

        items = []
        # Simple XML parsing without external deps
        import re
        entries = re.findall(r"<item>(.*?)</item>", text, re.DOTALL)
        for entry in entries[:10]:  # Latest 10 per feed
            title_m = re.search(r"<title>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</title>", entry, re.DOTALL)
            link_m  = re.search(r"<link>(.*?)</link>",   entry, re.DOTALL)
            pub_m   = re.search(r"<pubDate>(.*?)</pubDate>", entry, re.DOTALL)

            if title_m:
                items.append({
                    "title":     title_m.group(1).strip(),
                    "link":      link_m.group(1).strip()  if link_m  else url,
                    "published": pub_m.group(1).strip()   if pub_m   else "",
                    "source":    url.split("/")[2],
                })
        return items
    except Exception as e:
        log.warning(f"RSS fetch failed for {url}: {e}")
        return []

def quick_filter(headline: str) -> bool:
    """Return True if headline might be gold-relevant."""
    h = headline.lower()
    return any(kw in h for kw in GOLD_RELEVANT_KEYWORDS)

# ─── AI ANALYZER ────────────────────────────────────────────────────────────────
async def analyze_news_with_claude(headlines: list[dict]) -> list[dict] | None:
    """
    Send headlines to Claude. Returns only HIGH-IMPACT items with analysis.
    """
    if not headlines:
        return []

    headlines_text = "\n".join(
        f"{i+1}. [{item['source']}] {item['title']}"
        for i, item in enumerate(headlines)
    )

    prompt = f"""You are a professional gold market analyst. Analyze these news headlines and identify ONLY those that could cause SIGNIFICANT movement in gold prices (XAU/USD).

Headlines:
{headlines_text}

STRICT CRITERIA — only include if it's likely to cause >0.5% gold price movement:
- Fed rate decision/surprise/unexpected comments
- Major geopolitical escalation (war, nuclear threat, sanctions)
- Significant inflation data surprise (CPI, PCE, NFP big miss/beat)
- Central bank emergency action
- Major financial crisis or banking stress
- US Dollar sharp move drivers
- Large central bank gold buying/selling announcement

For EACH qualifying headline, provide a JSON object. Return ONLY a JSON array, no other text:
[
  {{
    "headline": "original headline text",
    "impact": "BULLISH" or "BEARISH" or "VOLATILE",
    "impact_score": 1-10,
    "reason": "1-2 sentence explanation why gold will move",
    "expected_move": "e.g. Gold could rally $30-50 because...",
    "urgency": "HIGH" or "MEDIUM"
  }}
]

If NO headlines qualify, return exactly: []
Only return valid JSON. No markdown, no explanation outside JSON."""

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 1500,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            raw = data["content"][0]["text"].strip()

            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            result = json.loads(raw)
            return result if isinstance(result, list) else []
    except Exception as e:
        log.error(f"Claude API error: {e}")
        return None

# ─── TELEGRAM SENDER ────────────────────────────────────────────────────────────
def format_telegram_message(item: dict, link: str) -> str:
    impact = item.get("impact", "VOLATILE")
    score  = item.get("impact_score", 5)
    urgency = item.get("urgency", "MEDIUM")

    emoji_map = {"BULLISH": "🟢📈", "BEARISH": "🔴📉", "VOLATILE": "⚡️📊"}
    emoji = emoji_map.get(impact, "⚡️")

    urgency_badge = "🚨 HIGH URGENCY" if urgency == "HIGH" else "⚠️ MEDIUM URGENCY"
    stars = "⭐" * min(score, 10)

    now = datetime.now(timezone.utc).strftime("%d %b %Y • %H:%M UTC")

    msg = f"""{emoji} *GOLD MARKET ALERT*
{urgency_badge}

📰 *Headline:*
{item['headline']}

📊 *Impact:* {impact} | Score: {stars} ({score}/10)

🧠 *Analysis:*
{item['reason']}

💰 *Expected Gold Move:*
{item['expected_move']}

🔗 [Full Story]({link})

🕐 _{now}_
━━━━━━━━━━━━━━━━━━"""
    return msg

async def send_telegram(bot: Bot, message: str):
    await bot.send_message(
        chat_id=TELEGRAM_CHAT_ID,
        text=message,
        parse_mode=ParseMode.MARKDOWN,
        disable_web_page_preview=False,
    )

# ─── MAIN LOOP ──────────────────────────────────────────────────────────────────
async def run_agent():
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    sent_cache = load_cache()

    # Startup message
    await send_telegram(bot,
        "🤖 *Gold News Agent Started!*\n\n"
        "Monitoring major gold-moving news...\n"
        "You'll only get alerts for HIGH-IMPACT events. 🥇"
    )
    log.info("Agent started. Monitoring gold news...")

    while True:
        try:
            log.info("Fetching news feeds...")
            all_items = []

            async with httpx.AsyncClient() as client:
                tasks = [fetch_rss(url, client) for url in RSS_FEEDS]
                results = await asyncio.gather(*tasks)
                for r in results:
                    all_items.extend(r)

            # Quick keyword filter
            filtered = [
                item for item in all_items
                if quick_filter(item["title"])
                   and item["title"] not in sent_cache
            ]
            log.info(f"Fetched {len(all_items)} headlines, {len(filtered)} passed keyword filter")

            if filtered:
                # Send to Claude for deep analysis
                analyzed = await analyze_news_with_claude(filtered)

                if analyzed is None:
                    log.error("Claude analysis failed, skipping cycle")
                elif analyzed:
                    log.info(f"Claude flagged {len(analyzed)} HIGH-IMPACT items")
                    for item in analyzed:
                        headline = item.get("headline", "")
                        # Find matching link from filtered items
                        link = next(
                            (f["link"] for f in filtered if f["title"] in headline or headline in f["title"]),
                            "https://www.kitco.com"
                        )
                        msg = format_telegram_message(item, link)
                        await send_telegram(bot, msg)
                        sent_cache.add(headline[:200])
                        await asyncio.sleep(2)  # Avoid Telegram rate limit

                    save_cache(sent_cache)
                else:
                    log.info("No high-impact items found this cycle")
            else:
                log.info("No relevant headlines after keyword filter")

        except Exception as e:
            log.error(f"Main loop error: {e}")
            await asyncio.sleep(60)
            continue

        log.info(f"Sleeping {CHECK_INTERVAL}s until next check...")
        await asyncio.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    asyncio.run(run_agent())
