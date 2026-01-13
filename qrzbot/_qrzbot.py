#!/usr/bin/env python
#
import asyncio
import functools
import io
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from datetime import time as dttime
from datetime import timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, ParamSpec, TypeVar

import aiosqlite
import httpx
import toml
from qrzlib import QRZ
from telegram import BotCommand, Update
from telegram.constants import ParseMode
from telegram.ext import (Application, CallbackContext, CommandHandler,
                          ContextTypes, ConversationHandler, ExtBot,
                          MessageHandler, filters)

from .quiz import quiz_status, reset_quiz, send_quiz
from .tools import get_effective_chat, get_effective_user, get_message

__version__ = '0.2.2'

P = ParamSpec("P")
R = TypeVar("R")


CONFIG_FILES = ("qrzbot.toml", "~/.local/qrzbot.toml", "/etc/qrzbot.toml")

POTA_URL = 'https://api.pota.app/spot/activator'
SOTA_URL = 'https://api-db2.sota.org.uk/api/spots/{length}/{cond}'
# SOTA_URL = 'https://api2.sota.org.uk/api/alerts/'

R_QRZ_URL = re.compile(r'https?://(?:www.|)qrz.com/db/(\w+)', re.IGNORECASE)

# States for the ConversationHandler
ASK_CALLSIGN = 1

logging.basicConfig(
  format="%(asctime)s - %(name)s[%(process)d]:%(lineno)d - %(levelname)s - %(message)s",
  datefmt='%H:%M:%S',
  level=logging.INFO
)
# set higher logging level for httpx to avoid all GET and POST requests being logged
logging.getLogger("httpx").setLevel(logging.WARNING)
LOG = logging.getLogger(__name__)

HELP = {
  "start": "Start the bot",
  "help": "Get help information",
  "ohmslaw": "Show formulas for ohms law",
  "pota": "Latest POTA activation. Arguments: [number filter= mode=]",
  "park": "Show park information. Argument:  park_id",
  "quiz": "Send a new quiz",
  "quizstatus": "Show status statistics",
  "resetquiz": "Reset the quiz to the beginning",
  "whois": "Displays call sign information. Argument: [call_sign]",
}


def async_lru_cache(
  maxsize: int = 128, typed: bool = False
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]:
  """LRU Cache decorator for async functions."""
  def decorator(async_func: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
    # Create a cache using functools.lru_cache for a helper function
    # This gives us the LRU behavior
    cached_key_mapper = functools.lru_cache(
      maxsize=maxsize,
      typed=typed)(lambda *args, **kwargs: None)

    # Store actual results separately
    result_cache: dict[str, str] = {}

    @functools.wraps(async_func)
    async def wrapper(*args, **kwargs):
      # Create a hashable key using the same method as functools.lru_cache
      key = functools._make_key(args, kwargs, typed)  # pylint: disable=protected-access

      # Update the LRU order by calling the cached function
      cached_key_mapper(*args, **kwargs)

      # Check if result is in cache
      if key in result_cache:
        return result_cache[key]

      # Not in cache, call the function and cache the result
      result = await async_func(*args, **kwargs)
      result_cache[key] = result
      return result

    return wrapper

  return decorator


def ttl_hash(seconds: int = 86400) -> int:
  # Quick and dirty way to add a ttl to lru_cache
  """Return the same value withing `seconds` time period"""
  return int(round(time.time() / seconds))


@async_lru_cache(maxsize=64)
async def get_state_name(country_code: str, state_code: str,
                         ttl: int = ttl_hash()) -> str | None:
  del ttl
  request = "SELECT name FROM states WHERE country_code=? AND state_code=?"
  async with aiosqlite.connect(Config.dbname) as db:
    db.row_factory = aiosqlite.Row
    cursor = await db.execute(request, (country_code, state_code))  # Corrected parameter names
    row = await cursor.fetchone()
  return str(row["name"]) if row else None  # Handle case when no result is found


@dataclass(slots=True)
class CallInfo:
  # pylint: disable=too-many-instance-attributes
  call: str
  full_name: str
  grid: str
  email: str
  country: str
  state: str
  latlon: tuple[float, float] | None
  expdate: datetime | None
  expired: bool | None

  @property
  def lat(self) -> float:
    if self.latlon:
      return self.latlon[0]
    return 0.0

  @property
  def lon(self) -> float:
    if self.latlon:
      return self.latlon[1]
    return 0.0


@dataclass()
class Config:
  # pylint: disable=too-few-public-methods
  """Holds configuration informations"""

  class Error(Exception):
    pass

  token: str = ''
  developer_id: int = 0
  qrz_call: str = ''
  qrz_key: str = ''
  dbname: str = ''

  @classmethod
  def load(cls) -> None:
    """load token and developer_id from the config file"""
    for config_file in CONFIG_FILES:
      config_path = Path(config_file).expanduser()
      if config_path.exists():
        break
    else:
      raise FileNotFoundError('Configuration file missing')

    try:
      with config_path.open('r', encoding="utf-8") as cfd:
        _config = toml.load(cfd)
    except ValueError as err:
      raise Config.Error(f'Configuration error {err}')

    # Update instance attributes
    for key, value in _config.items():
      if hasattr(cls, key):
        setattr(cls, key, value)  # Assign values dynamically
      else:
        LOG.warning('Unknown config attribute: "%s"', key)


def esc_md(text: str) -> str:
  # Escape all reserved characters except those inside code blocks or links
  reserved_chars = r'_*[]()~`>#+-=|{}.!\\'
  return re.sub(f'([{re.escape(reserved_chars)}])', r'\\\1', str(text))


@functools.lru_cache(maxsize=64)
def qrz_whois(call: str, ttl: int = ttl_hash()) -> CallInfo:
  del ttl
  now = datetime.now()
  qrz = QRZ()
  qrz.authenticate(Config.qrz_call, Config.qrz_key)
  if (ci := qrz.get_call(call)) is None:
    raise QRZ.NotFound(f'"{call}" Not found')
  try:
    full_dt = datetime.combine(ci.expdate, dttime())
    expired = (full_dt - now).days < 0
  except (AttributeError, KeyError, TypeError, ValueError):
    expired = None
  return CallInfo(call, ci.name_fmt, ci.grid, ci.email, ci.country, ci.state,
                  ci.latlon, ci.expdate, expired)


async def load_json(url: str) -> list[dict[str, Any]]:
  async with httpx.AsyncClient() as client:
    response = await client.get(url)
  if response.is_error:
    error = response.json()
    LOG.error('Resource not available [%s]', error['message'])
    raise IOError
  return response.json()


async def load_spots(url: str) -> list[dict[str, Any]]:
  data = await load_json(url)
  if not data:
    return []

  for spot in data:
    try:
      spot['frequency'] = float(spot['frequency']) / 1000
    except (ValueError, KeyError):
      spot['frequency'] = float('nan')

  return data


async def sota(cond: str, bot: ExtBot, chat_id: int) -> None:
  max_spot = 5
  isoformat = [
    "%Y-%m-%dT%H:%M:%S.%fZ",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S"
  ]
  if not cond:
    cond = 'all'

  url = SOTA_URL.format(length=max_spot, cond=cond)
  LOG.info(url)
  try:
    data = await load_json(url)
  except IOError:
    await bot.send_message(chat_id=chat_id, text='Resource not available at the moment')
    return None

  if not data:
    await bot.send_message(chat_id=chat_id, text='No spots at the moment')
    return

  text = ['Latest SOTA spots']
  for spot in data:
    timestamp = None
    try:
      for fmt in isoformat:
        try:
          timestamp = datetime.strptime(spot["timeStamp"], fmt).replace(tzinfo=timezone.utc)
          break
        except ValueError:
          pass

      text.append('───────────')
      text.append(rf'Summit: `{esc_md(spot["summitCode"])}`')
      text.append(f'Band: {esc_md(spot["frequency"])}')
      if 'mode' in spot:
        text.append(f'Mode: {esc_md(spot["mode"])}')
      text.append(f'Activator: {esc_md(spot["activatorCallsign"])} {esc_md(spot["activatorName"])}')
      if 'associationCode' in spot:
        text.append(f'Association: {esc_md(spot["associationCode"])}')
      if 'comments' in spot:
        text.append(f'Comments: {esc_md(spot["comments"])}')
      if timestamp:
        text.append(f'Time: {esc_md(timestamp.strftime("%Y-%m-%d %H:%M"))}')
    except KeyError as err:
      LOG.error('%s - %s', spot, err)

  await bot.send_message(
    chat_id=chat_id, text='\n'.join(text), disable_web_page_preview=True,
    parse_mode=ParseMode.MARKDOWN_V2
  )


async def pota(max_spot: int, conds: dict, bot: ExtBot, chat_id: int, message_id: int):
  try:
    data = await load_spots(POTA_URL)
  except IOError:
    await bot.send_message(chat_id=chat_id, reply_to_message_id=message_id,
                           text='Resource not available at the moment')
    return

  if not data:
    await bot.send_message(chat_id=chat_id, reply_to_message_id=message_id,
                           text='No park activation at the moment')
    return

  if 'filter' in conds:
    data = [d for d in data if d['locationDesc'].startswith(conds['filter'].upper())]
  if 'mode' in conds:
    data = [d for d in data if d['mode'] == conds['mode'].upper()]

  data = data[:max_spot]

  if len(data) == 0:
    await bot.send_message(chat_id=chat_id, text='No park activation found at this time.')
    return

  spot_url = esc_md('https://pota.app/#/park/')
  response = [f"*Latest {len(data)} POTA activationas*"]

  for spot in data:
    spot["mode"] = spot["mode"] if spot["mode"] else 'Multiple'
    text = f'''───────────
{esc_md(spot["activator"])} \\(*{esc_md(spot["mode"])}*\\), \
`{esc_md(spot["reference"])}`, {esc_md(spot["name"])}
Location: {esc_md(spot["locationDesc"])}, Grid: {esc_md(spot["grid6"])}
Frequency: {esc_md(spot["frequency"])} MHz
{spot_url}{esc_md(spot["reference"])}'''
    response.append(text)

  text = '\n'.join(response)
  await bot.send_message(chat_id=chat_id, text=text, reply_to_message_id=message_id,
                         disable_web_page_preview=True, parse_mode=ParseMode.MARKDOWN_V2)


async def format_callinfo(user: str, callsign: str, call_info: CallInfo) -> str:
  try:
    response = [f'*From:* {user}'] if user else []
    # Add call information (always present)
    call_header = f"*To:* {esc_md(call_info.call)}"
    if callsign != call_info.call:
      call_header += f" Previously: {callsign}"
    response.append(call_header)

    # Add optional fields if they exist
    field_mappings = {
      "full_name": "*Name:*",
      "email": "*Email:*",
      "grid": "*Grid:*",
      "country": "*Country:*",
      "state": "*State:*",
    }

    for attr, label in field_mappings.items():
      if value := getattr(call_info, attr, None):
        response.append(f"{label} {esc_md(value)}")

    # Add QRZ link (always present)
    response.append(f"*QRZ:* {esc_md('https://qrz.com/db/' + call_info.call)}")

    # Add expired notice if applicable
    if call_info.expired:
      response.append('`The call sign is EXPIRED`')

    return '\n'.join(response)
  except Exception as exc:
    # Proper exception handling
    LOG.error("Error retrieving callsign info: %s", exc)
    return f"Error retrieving information for {callsign}"


async def whois(update: Update, context: CallbackContext) -> int:
  bot = context.bot
  chat_id = get_effective_chat(update).id
  message = get_message(update)

  job = context.job_queue.run_once(
    whois_send_timeout_message,
    when=30,
    data=chat_id,
    name=f"whois_timeout_{chat_id}"
  )
  context.user_data['timeout_job'] = job

  text = message.text.strip() if message.text else ""
  result = re.match(r'^/whois\s+(\w+)$', text, re.IGNORECASE)
  if result:
    callsign = result.group(1).upper()
    return await process_callsign(update, context, callsign)

  # Otherwise, ask the user for a callsign
  await bot.send_message(chat_id=chat_id, text="<b>Please enter a callsign:</b> (30s timeout)",
                         parse_mode="HTML")
  return ASK_CALLSIGN


async def whois_send_timeout_message(context: ContextTypes.DEFAULT_TYPE):
  chat_id = context.job.data
  await context.bot.send_message(
    chat_id=chat_id,
    text="⏰ Whois lookup timed out after 30 seconds of inactivity."
  )


async def whois_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
  # Cancel timeout job
  if 'timeout_job' in context.user_data:
    context.user_data['timeout_job'].schedule_removal()

  await update.message.reply_text("Cancelled.")
  return ConversationHandler.END


async def received_callsign(update: Update, context: CallbackContext) -> int:
  """Handles the callsign entered after /whois without argument."""
  if 'timeout_job' in context.user_data:
    context.user_data['timeout_job'].schedule_removal()

  message = get_message(update)

  if not message:
    return ConversationHandler.END

  if message.text:
    callsign = message.text.strip().upper()
    await process_callsign(update, context, callsign)
  else:
    await message.reply_text("Please provide a valid callsign.")

  return ConversationHandler.END


async def process_callsign(update: Update, context: CallbackContext, callsign: str) -> int:
  """Shared logic to lookup and respond with call info."""
  bot = context.bot
  chat_id = get_effective_chat(update).id
  user = get_effective_user(update)
  username = getattr(user, 'first_name', getattr(user, 'username', '?'))

  try:
    call_info = await asyncio.to_thread(qrz_whois, callsign)
  except QRZ.NotFound as err:
    await bot.send_message(chat_id=chat_id, text=str(err))
    LOG.error(err)
    return ConversationHandler.END
  except QRZ.SessionError as err:
    await bot.send_message(chat_id=chat_id, text='QRZ Identification error')
    LOG.error(err)
    raise

  text = await format_callinfo(username, callsign, call_info)
  try:
    await bot.send_message(chat_id=chat_id, text=text, disable_web_page_preview=True,
                           parse_mode=ParseMode.MARKDOWN_V2)
  except Exception as err:
    LOG.error("Failed to send MarkdownV2 message: %s", err)
    # Fallback to plain text if Markdown fails
    await bot.send_message(chat_id=chat_id, text=text)

  if call_info.latlon:
    await bot.send_location(chat_id=chat_id, latitude=call_info.lat, longitude=call_info.lon)

  return ConversationHandler.END


async def ohms(update: Update, context: CallbackContext) -> None:
  bot = context.bot
  chat_id = get_effective_chat(update).id
  message_id = get_message(update).message_id
  photo_url = 'https://bsdworld.org/misc/OhmsLaw.png'
  await bot.send_photo(chat_id=chat_id, caption="Ohms law formulas",
                       reply_to_message_id=message_id,
                       photo=photo_url)


async def start(update: Update, context: CallbackContext) -> None:
  bot = context.bot
  chat_id = get_effective_chat(update).id
  message_id = get_message(update).message_id
  text = (
    f"__Thanks for using CQCall Bot {esc_md(__version__)}__",
    "",
    "The commands are:",
    esc_md("  - /whois [call] - Displays the information about a call sign."),
    esc_md("  - /pota - Show the latest POTA activations."),
    esc_md("  - /pota [nb] [filter=] [mode=] - Filter by location and mode."),
    esc_md("  - /park [park id] - Show the park information."),
    esc_md("Admin commands:"),
    esc_md("  - /quiz - Send a new quiz."),
    esc_md("  - /quizstatus - Show status statistics."),
    esc_md("  - /resetquiz - Reset the quiz to the beginning."),
  )
  await bot.send_message(chat_id, text='\n'.join(text),
                         reply_to_message_id=message_id,
                         parse_mode=ParseMode.MARKDOWN_V2)


@async_lru_cache(maxsize=1024)
async def get_summit(summit_code: str, ttl: int = ttl_hash()) -> dict:
  del ttl
  request = "SELECT * FROM sota_summit WHERE SummitCode=?"
  async with aiosqlite.connect(Config.dbname) as db:
    db.row_factory = aiosqlite.Row
    async with db.execute(request, (summit_code,)) as cursor:
      summit = await cursor.fetchone()
  if not summit:
    raise KeyError(f'Summit id: {summit_code} not found')
  return dict(summit)


async def sota_summit(update: Update, context: CallbackContext) -> None:
  chat_id = get_effective_chat(update).id
  bot = context.bot

  if context.args is None or len(context.args) != 1:
    await bot.send_message(chat_id=chat_id, text='The summit id missing. Try /summit summit_id')
    return

  sid = context.args[0].upper()
  try:
    summit = await get_summit(sid)
  except KeyError as err:
    await bot.send_message(chat_id=chat_id, text=str(err))
    return

  await bot.send_location(
    chat_id=chat_id, longitude=summit['Longitude'], latitude=summit['Latitude']
  )
  response = [f"*Summit Id:* `{esc_md(summit['SummitCode'])}`"]
  response.append(f"*Region Name:* {esc_md(summit['RegionName'])}")
  response.append(f"*Name:* {esc_md(summit['SummitName'])}")
  response.append(f"*Altitude:* {esc_md(summit['AltM'])} Meters")
  response.append(f"*Points:* {esc_md(summit['Points'])}")
  if summit['BonusPoints'] > 0:
    response.append(f"*Bonus:* {esc_md(summit['BonusPoints'])}")
  response.append(f"*Activation Count*: {esc_md(summit['ActivationCount'])}")
  await bot.send_message(chat_id=chat_id, text='\n'.join(response),
                         parse_mode=ParseMode.MARKDOWN_V2)


@async_lru_cache(maxsize=1024)
async def get_park(parkid: str, ttl: int = ttl_hash()) -> dict:
  del ttl
  request = "SELECT * FROM pota_parks WHERE reference=?"
  async with aiosqlite.connect(Config.dbname) as db:
    db.row_factory = aiosqlite.Row
    async with db.execute(request, (parkid,)) as cursor:
      park_info = await cursor.fetchone()
  if not park_info:
    raise KeyError(f'Park id: {parkid} not found')
  return dict(park_info)


async def park(update: Update, context: CallbackContext) -> None:
  chat_id = get_effective_chat(update).id
  message_id = get_message(update).message_id
  bot = context.bot
  if context.args is None or len(context.args) != 1:
    await bot.send_message(chat_id=chat_id, text='The park id is missing. Try /park park_id')
    return

  ref = context.args[0].upper()
  if not re.match(r'^\w{2}-\d{4}.*', ref):
    await bot.send_message(chat_id=chat_id, text='Wrong park id')
    return

  try:
    park_info = await get_park(ref)
  except KeyError as err:
    await bot.sent_message(chat_id=chat_id, text=str(err))

  text = []
  text.append(f'Park: {esc_md(park_info[r"reference"])}')
  text.append(f'Name: {esc_md(park_info[r"name"])}')
  text.append(f'Location: {esc_md(park_info[r"locationDesc"])}')
  text.append(f'{"_Active_" if park_info["active"] == "1" else "_Inactive_"}')
  text.append(f'{esc_md("https://pota.app/#/park/" + ref)}')

  await bot.send_message(chat_id=chat_id, text='\n'.join(text),
                         reply_to_message_id=message_id,
                         disable_web_page_preview=True,
                         parse_mode=ParseMode.MARKDOWN_V2)

  if all((park_info.get('latitude'), park_info.get('longitude'))):
    await bot.send_location(
      chat_id=chat_id, latitude=park_info['latitude'], longitude=park_info['longitude']
    )


async def pota_router(update: Update, context: CallbackContext) -> None:
  chat_id = get_effective_chat(update).id
  bot = context.bot
  try:
    message_id = get_message(update).message_id
  except ValueError as err:
    LOG.error('Error message empty: %d', err)
    return
  if context.args is None or len(context.args) == 0:
    await pota(5, {}, bot, chat_id, message_id)
    return

  conds = {}
  cnt = 5
  for arg in context.args:
    if match := re.match(r'(\d+)', arg):
      cnt = int(match.group(1))
    elif match := re.match(r'(\w+)=([\w-]+)', arg):
      key = match.group(1).lower()
      val = match.group(2).upper()
      if key not in ('filter', 'mode'):
        LOG.warning('Invalid argument %s=%s', key, val)
      else:
        conds[key] = val

  cnt = cnt if cnt <= 10 else 10
  await pota(cnt, conds, bot, chat_id, message_id)


async def sota_router(update: Update, context: CallbackContext) -> None:
  chat_id = get_effective_chat(update).id
  bot = context.bot
  if context.args is None or len(context.args) == 0:
    conds = 'all/all'
  else:
    conds = context.args[0].upper()
  await sota(conds, bot, chat_id)


async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
  chat_id = get_effective_chat(update).id
  bot = context.bot
  message = get_message(update)
  message_text = message.text
  user = get_effective_user(update)
  username = getattr(user, 'first_name', getattr(user, 'username', '?'))

  # Check if the message matches the pattern
  if not message_text:
    return
  if not (match := R_QRZ_URL.match(message_text)):
    return
  callsign = match.group(1)
  try:
    await message.delete()
    try:
      call_info = await asyncio.to_thread(qrz_whois, callsign)
    except QRZ.NotFound as err:
      await bot.send_message(chat_id=chat_id, text=str(err))
      return

    text = await format_callinfo(username, callsign, call_info)
    await bot.send_message(chat_id=chat_id, text=text,
                           disable_web_page_preview=True,
                           parse_mode=ParseMode.MARKDOWN_V2)

    if call_info.latlon:
      await bot.send_location(chat_id=chat_id, latitude=call_info.lat, longitude=call_info.lon)

  except Exception as exc:
    LOG.warning(exc)
    await bot.send_message(chat_id=chat_id, text="Not authorized to modify the message")


async def async_run(application: Application) -> None:
  # Initialize and start the bot
  await application.initialize()
  await application.start()
  if not application.updater:
    raise ValueError('Telegram application updater error')
  await application.updater.start_polling()

  LOG.info('Prod environment using asyncio.Event()')
  LOG.info("Bot is running. Press Ctrl+C to stop.")
  await set_commands(application)

  # Keep running until cancelled
  try:
    await asyncio.Event().wait()
  except (KeyboardInterrupt, SystemExit):
    LOG.info("Stopping bot...")

  # Gracefully shut down
  await application.updater.stop()
  await application.stop()
  await application.shutdown()
  LOG.info("Bot shut down successfully.")


async def log_command(update: Update, _: CallbackContext) -> None:
  user = get_effective_user(update)
  if not update.message:
    raise ValueError('Message object not set')
  command = update.message.text
  LOG.info("User %d (%s) sent command: %s", user.id, user.username, command)


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
  """Log the error and send a telegram message to notify the developer."""
  LOG.error("Exception while handling an update")

  # Build the message with some markup and additional information about what happened.
  if isinstance(update, Update):
    filename = f'error-{int(time.time())}.json'
    update_str = update.to_json()
    caption = "An exception was raised while handling an update."
    buf = io.BytesIO()
    buf.write(update_str.encode('utf-8'))
    buf.seek(0)
    await context.bot.send_document(chat_id=Config.developer_id, document=buf,
                                    filename=filename, caption=caption)
  else:
    await context.bot.send_message(chat_id=Config.developer_id, text=str(update))


async def set_commands(application: Application) -> None:
  commands = [BotCommand(k, v) for k, v in HELP.items()]
  try:
    await application.bot.set_my_commands(commands)
    LOG.info("Commands have been updated successfully!")
  except Exception as err:
    LOG.error("Error: %s", err)


def main() -> None:
  LOG.info('qrzbot version: %s', __version__)
  Config.load()
  if not (Config.dbname and os.path.exists(Config.dbname)):
    LOG.error('%s not found', Config.dbname)
    sys.exit(os.EX_DATAERR)

  application = Application.builder().token(Config.token).build()
  application.add_error_handler(error_handler)

  whois_conv = ConversationHandler(
    entry_points=[CommandHandler("whois", whois)],
    states={
      ASK_CALLSIGN: [MessageHandler(filters.TEXT & ~filters.COMMAND, received_callsign)],
    },
    fallbacks=[CommandHandler("cancel", whois_cancel)],
    conversation_timeout=32
  )

  # Add handlers
  application.add_handler(CommandHandler("start", start))
  application.add_handler(MessageHandler(
    filters.TEXT & filters.Regex(R_QRZ_URL) & ~filters.COMMAND,
    message_handler)
  )

  application.add_handler(CommandHandler("help", start))
  application.add_handler(whois_conv)

  application.add_handler(CommandHandler("pota", pota_router))
  application.add_handler(CommandHandler("park", park))

  application.add_handler(CommandHandler("sota", sota_router))
  application.add_handler(CommandHandler("summit", sota_summit))
  application.add_handler(CommandHandler("OhmsLaw", ohms))
  application.add_handler(CommandHandler("quiz", send_quiz))
  application.add_handler(CommandHandler("resetquiz", reset_quiz))
  application.add_handler(CommandHandler("quizstatus", quiz_status))

  application.add_handler(MessageHandler(filters.COMMAND, log_command), group=-1)

  if sys.platform == 'darwin':
    LOG.setLevel(logging.DEBUG)
  asyncio.run(async_run(application))  # Production


if __name__ == '__main__':
  main()
