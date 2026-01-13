#!/usr/bin/env python
#

import asyncio
import csv
import json
import logging
import os
import traceback
from typing import Any, Dict, List, Optional

from telegram import BotCommand, Poll, Update
from telegram.ext import Application, CallbackContext, CommandHandler

from .tools import get_effective_chat, get_effective_user

POLL_TRACKING_FILE = "/var/tmp/poll_tracking.json"
QUESTIONS_CSV_FILE = "/var/tmp/questions.csv"

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
    last_index = -1
    if "chats" in tracking_data and chatid in tracking_data["chats"]:
      last_index = tracking_data["chats"][chatid].get("last_question_index", -1)

    next_index = (last_index + 1) % len(self.questions)
    return self.questions[next_index], next_index


def load_tracking_data() -> Dict[str, Any]:
  if os.path.exists(POLL_TRACKING_FILE):
    try:
      with open(POLL_TRACKING_FILE, 'r', encoding='utf=8') as file:
        data = json.load(file)
        # Ensure the data has the expected structure
        if "chats" not in data:
          data["chats"] = {}
        return data
    except Exception as err:
      logging.error("Error loading tracking data: %s", err)
      return {"chats": {}}
  return {"chats": {}}


# Save tracking data
def save_tracking_data(data: Dict[str, Any]) -> None:
  with open(POLL_TRACKING_FILE, 'w', encoding='utf=8') as file:
    json.dump(data, file)


# Check if user is an admin in the chat
async def is_admin(update: Update, context: CallbackContext) -> bool:
  """Check if the user is an admin in the chat."""
  chat_id = get_effective_chat(update).id
  user_id = get_effective_user(update).id

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
  bot.send_message(chat_id=chat_id, text=text)


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
    if "chats" not in tracking_data:
      tracking_data["chats"] = {}

    # Initialize chat data if not exists
    if chatid not in tracking_data["chats"]:
      tracking_data["chats"][chatid] = {
        "last_question_index": -1,
        "questions_sent": 0,
        "admin_id": user_id  # First user to use quiz becomes admin for simplicity
      }

    try:
      quiz_manager = QuizManager(QUESTIONS_CSV_FILE)
      if not quiz_manager.questions:
        raise ValueError("No questions found")
    except (IOError, ValueError) as err:
      logging.error(str(err))
      await bot.send_message(chat_id=chat_id, text=str(err))
      return

    try:
      question_data, next_index = quiz_manager.get_question(chatid)
    except ValueError as err:
      await bot.send_message(chat_id=chat_id, text=f"Error getting question: {str(err)}")
      return

    explanation = (
      f"Question {tracking_data['chats'][chatid]['questions_sent'] + 1} "
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

    # Update tracking data
    tracking_data["chats"][chatid]["last_question_index"] = next_index
    tracking_data["chats"][chatid]["questions_sent"] += 1
    tracking_data["chats"][chatid]["last_poll_id"] = message.poll.id
    tracking_data["chats"][chatid]["last_message_id"] = message.message_id
    save_tracking_data(tracking_data)

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
      tracking_data["chats"][chatid]["last_question_index"] = -1
      tracking_data["chats"][chatid]["questions_sent"] = 0
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
    if not os.path.exists(QUESTIONS_CSV_FILE):
      await bot.send_message(
        chat_id=chat_id,
        text="Quiz file not found. An admin can create a sample file using /createcsv command."
      )
      return

    # Make sure the structure exists
    if "chats" not in tracking_data:
      tracking_data["chats"] = {}

    if chatid in tracking_data["chats"]:
      quiz_manager = QuizManager(QUESTIONS_CSV_FILE)

      if not quiz_manager.questions:
        await bot.send_message(chat_id=chat_id, text="No questions found in the CSV file.")
        return

      total_questions = len(quiz_manager.questions)
      questions_sent = tracking_data["chats"][chatid]["questions_sent"]
      last_index = tracking_data["chats"][chatid]["last_question_index"]

      current_cycle = (questions_sent // total_questions) + 1
      current_position = (last_index + 1) % total_questions
      if current_position == 0:
        current_position = total_questions

      await bot.send_message(chat_id, text=(
        f"Quiz Status:\n"
        f"- Questions sent: {questions_sent}\n"
        f"- Total questions: {total_questions}\n"
        f"- Current position: Question {current_position} of {total_questions}\n"
        f"- Current cycle: {current_cycle}\n"
        f"- Next question will be: #{(last_index + 1) % total_questions + 1}"
      ))
    else:
      await bot.send_message(chat_id=chat_id, text="No quiz has been started in this chat yet.")
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


async def test(token) -> None:
  """Start the bot."""
  application = Application.builder().token(token).build()
  await set_commands(application)

  # Add command handlers
  application.add_handler(CommandHandler("start", start))
  application.add_handler(CommandHandler("quiz", send_quiz))
  application.add_handler(CommandHandler("resetquiz", reset_quiz))
  application.add_handler(CommandHandler("quizstatus", quiz_status))

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
  TOKEN = "TOKEN"
  POLL_TRACKING_FILE = "./poll_tracking.json"
  QUESTIONS_CSV_FILE = "./questions.csv"

  asyncio.run(test(TOKEN))
