#!/usr/bin/env python
#

import asyncio
import csv
import html
import json
import logging
import os
import re
import traceback
from typing import Any, Dict, List, Optional

from telegram import BotCommand, Poll, Update
from telegram.constants import ParseMode
from telegram.ext import (Application, CallbackContext, CommandHandler,
                          ContextTypes, ConversationHandler, MessageHandler,
                          filters)

from .config import Config
from .tools import get_effective_chat, get_effective_user

CONFIRM = 1

logging.basicConfig(
  format="%(asctime)s - %(name)s[%(process)d]:%(lineno)d - %(levelname)s - %(message)s",
  datefmt='%H:%M:%S',
  level=logging.INFO
)


class QuizManager:
  # pylint: disable=too-few-public-methods
  def __init__(self, csv_file: str):
    self.csv_file = csv_file
    self.questions = self._load_questions()

  def _load_questions(self) -> List[Dict[str, Any]]:
    """Load questions from CSV file."""
    questions = []
    with open(self.csv_file, 'r', encoding='utf-8') as file:
      reader = csv.reader(file)
      for line_no, row in enumerate(reader, 1):
        if len(row) < 4:
          logging.warning('Wrong question format line: %d', line_no)
          continue

        question = f"{row[0]} - {row[2]}"
        correct_option = ord(row[1]) - 65
        options = row[3:]

        questions.append({
          "question": question,
          "options": options,
          "correct_option_id": correct_option
        })
    return questions

  def get_question(self, chatid: str, question_index: Optional[int] = None) -> tuple:
    """Get a specific question or a new one based on tracking."""
    if not self.questions:
      raise ValueError("No questions available")

    if question_index is not None and 0 <= question_index < len(self.questions):
      return self.questions[question_index], question_index

    tracking_data = load_tracking_data()
    last_index = tracking_data["chats"][chatid].get("question_index", -1)
    new_index = (last_index + 1) % len(self.questions)
    return self.questions[new_index], new_index


def load_tracking_data() -> Dict[str, Any]:
  tracking_data: dict[str, dict] = {"chats": {}}
  if os.path.exists(Config.poll_tracking_file):
    try:
      with open(Config.poll_tracking_file, 'r', encoding='utf=8') as file:
        data = json.load(file)
        # Ensure the data has the expected structure
        if "chats" not in data:
          data = tracking_data
        return data
    except Exception as err:
      logging.error("Error loading tracking data: %s", err)
  return tracking_data


# Save tracking data
def save_tracking_data(data: Dict[str, Any]) -> None:
  with open(Config.poll_tracking_file, 'w', encoding='utf=8') as file:
    json.dump(data, file, indent=2)


# Check if user is an admin in the chat
async def is_admin(update: Update, context: CallbackContext) -> bool:
  """Check if the user is an admin in the chat."""
  chat_id = get_effective_chat(update).id
  user_id = get_effective_user(update).id
  if user_id == Config.developer_id:
    return True

  try:
    chat_member = await context.bot.get_chat_member(chat_id, user_id)
    return chat_member.status in ['creator', 'administrator']
  except Exception as err:
    logging.error("Error checking admin status: %s", err)
    return False


async def start(update: Update, context: CallbackContext) -> None:
  """Send a welcome message when the command /start is issued."""
  bot = context.bot
  chat_id = get_effective_chat(update).id
  if update.message is None:
    return
  text = (
    'Welcome to the Level 2 question pool.\n\n'
    'Admin commands:\n'
    '/quiz - Send a random question\n'
    '/resetquiz - Reset the quiz progress\n'
    '/quizstatus - Check quiz status'
  )
  await bot.send_message(chat_id=chat_id, text=text)


async def send_quiz(update: Update, context: CallbackContext) -> None:
  # pylint: disable=too-many-locals
  # I need to break that function in several parts
  """Send a quiz question from the CSV file."""
  bot = context.bot
  chat_id = get_effective_chat(update).id
  chatid = str(chat_id)
  user = get_effective_user(update)
  user_id = user.id
  user_name = user.username if user.username else chat_id
  admin = await is_admin(update, context)
  logging.info('%s (Admin: %s) sending a new poll', user_name, admin)

  try:
    if not admin:
      await bot.send_message(chat_id, text="Only group administrators can send a quiz.")
      return

    # Load tracking data - ensure the "chats" key exists
    tracking_data = load_tracking_data()

    # Initialize chat data if not exists
    if chatid not in tracking_data["chats"]:
      tracking_data["chats"][chatid] = {
        "question_index": 0,
        "admin_id": user_id  # First user to use quiz becomes admin for simplicity
      }
      save_tracking_data(tracking_data)

    try:
      quiz_manager = QuizManager(Config.questions_file)
      if not quiz_manager.questions:
        raise ValueError("No questions found")
    except (IOError, ValueError) as err:
      logging.error(str(err))
      await bot.send_message(chat_id=chat_id, text=str(err))
      return

    try:
      question_data, question_index = quiz_manager.get_question(chatid)
    except ValueError as err:
      await bot.send_message(chat_id=chat_id, text=f"Error getting question: {str(err)}")
      return

    # Update tracking data
    tracking_data["chats"][chatid]["question_index"] = question_index
    save_tracking_data(tracking_data)

    explanation = (
      f"Question {tracking_data['chats'][chatid]['question_index'] + 1} "
      f"of {len(quiz_manager.questions)}"
    )
    # Send the quiz and pin it
    message = await bot.send_poll(
      chat_id=chat_id,
      question=question_data["question"],
      options=question_data["options"],
      type=Poll.QUIZ,
      correct_option_id=question_data["correct_option_id"],
      explanation=explanation,
      is_anonymous=False
    )
    try:
      await bot.pin_chat_message(chat_id=chat_id, message_id=message.message_id)
    except Exception as err:
      logging.warning("Poll created but could't pin it: %s", err)

  except Exception as err:
    tb = traceback.format_exc()
    logging.error("Error in send_quiz: %s\n%s", err, tb)
    await bot.send_message(chat_id=chat_id, text=f"Error sending quiz: {str(err)}")


async def reset_quiz(update: Update, context: CallbackContext) -> None:
  """Reset the quiz progress for this chat (admin only)."""
  bot = context.bot
  chat_id = get_effective_chat(update).id
  chatid = str(chat_id)
  try:
    # Check if user is admin
    if not await is_admin(update, context):
      await bot.send_message(
        chat_id=chat_id, text="Only group administrators can reset the quiz."
      )
      return

    # Load and update tracking data
    tracking_data = load_tracking_data()

    # Make sure the structure exists
    if "chats" not in tracking_data:
      tracking_data["chats"] = {}

    if chatid in tracking_data["chats"]:
      tracking_data["chats"][chatid]["question_index"] = -1
      save_tracking_data(tracking_data)
      await bot.send_message(
        chat_id=chat_id, text="Quiz progress has been reset. Use /quiz to start fresh."
      )
    else:
      await bot.send_message(chat_id=chat_id, text="No quiz has been started in this chat yet.")
  except Exception as err:
    tb = traceback.format_exc()
    logging.error("Error in reset_quiz: %s\n%s", err, tb)
    await bot.send_message(chat_id=chat_id, text=f"Error resetting quiz: {str(err)}")


async def quiz_status(update: Update, context: CallbackContext) -> None:
  """Check the quiz status for this chat."""
  bot = context.bot
  chat_id = get_effective_chat(update).id
  chatid = str(chat_id)
  try:
    # Load tracking data
    tracking_data = load_tracking_data()

    # Check if CSV file exists
    if not os.path.exists(Config.questions_file):
      await bot.send_message(
        chat_id=chat_id,
        text="Quiz file not found."
      )
      return

    # Make sure the structure exists
    if "chats" not in tracking_data:
      tracking_data["chats"] = {}

    if chatid in tracking_data["chats"]:
      quiz_manager = QuizManager(Config.questions_file)

      if not quiz_manager.questions:
        await bot.send_message(chat_id=chat_id, text="No questions found in the CSV file.")
        return

      total_questions = len(quiz_manager.questions)
      try:
        question_index = tracking_data["chats"][chatid]["question_index"]
        question = quiz_manager.questions[question_index]['question']
      except KeyError:
        question_index = 0
        question = 'Error'

      if question_index < 0:
        await bot.send_message(
          chat_id, text="You haven't started the quiz. Use /quiz to start fresh"
        )
      else:
        await bot.send_message(chat_id, text=(
          f"Quiz Status:\n"
          f"○ <b>Total Questions:</b> {total_questions}\n"
          f"○ <b>Question Pool Index:</b> {question_index}\n"
          f"○ <b>Last Question:</b>\n{html.escape(question)}\n"
        ), parse_mode=ParseMode.HTML)
    else:
      await bot.send_message(
        chat_id=chat_id,
        text="No quiz has been started in this chat. Use /quiz to start fresh")
  except Exception as err:
    tb = traceback.format_exc()
    logging.error("Error in quiz_status: %s\n%s", err, tb)
    await bot.send_message(chat_id=chat_id, text=f"Error checking quiz status: {str(err)}")


async def set_commands(application):
  commands = [
    BotCommand('quiz', 'Send a new Ham quiz'),
    BotCommand('resetquiz', 'Start over'),
    BotCommand('quizstatus', 'Show quiz status'),
  ]

  try:
    await application.bot.set_my_commands(commands)
    logging.info("Commands have been updated successfully!")
  except Exception as err:
    logging.error("Error: %s", err)


async def reset_confirmation(update: Update, _) -> int:
  await update.message.reply_text("Are you sure? (yes/no)")
  return CONFIRM


async def reset_confirmed(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
  await reset_quiz(update, context)
  return ConversationHandler.END


async def reset_denied(update: Update, _) -> int:
  await update.message.reply_text("Cancelled.")
  return ConversationHandler.END


def reset_handler():
  reset_dialog = ConversationHandler(
    entry_points=[CommandHandler("resetquiz", reset_confirmation)],
    states={
      CONFIRM: [
        MessageHandler(filters.Regex(re.compile(r"^yes$", re.IGNORECASE)), reset_confirmed),
        MessageHandler(filters.Regex(re.compile(r"^no$", re.IGNORECASE)), reset_denied),
      ],
    },
    fallbacks=[],
  )
  return reset_dialog


async def test(token) -> None:
  """Start the bot."""
  application = Application.builder().token(token).build()
  await set_commands(application)

  # Add command handlers
  application.add_handler(CommandHandler("start", start))
  application.add_handler(CommandHandler("quiz", send_quiz))
  application.add_handler(CommandHandler("quizstatus", quiz_status))
  application.add_handler(reset_handler())

  await application.initialize()
  await application.start()
  if application.updater is None:
    raise ValueError('Telegram application updater error')
  await application.updater.start_polling()

  logging.info('Prod environment using asyncio.Event()')
  logging.info("Bot is running. Press Ctrl+C to stop.")
  await set_commands(application)

  try:
    await asyncio.Event().wait()
  except (KeyboardInterrupt, SystemExit):
    logging.info("Stopping bot...")

  # Gracefully shut down
  await application.updater.stop()
  await application.stop()
  await application.shutdown()
  logging.info("Bot shut down successfully.")


if __name__ == '__main__':
  Config.load()
  TOKEN = Config.token

  asyncio.run(test(TOKEN))
