#! /usr/bin/env python
# vim:fenc=utf-8
#
# Copyright © 2025 fred <github-fred@hidzz.com>
#
# Distributed under terms of the BSD 3-Clause license.

from typing import cast

from telegram import Chat, Message, Update, User


def get_effective_chat(update: Update) -> Chat:
  if update.effective_user is None:
    raise ValueError('No effective chat in update')
  return cast(Chat, update.effective_chat)


def get_effective_user(update: Update) -> User:
  if update.effective_user is None:
    raise ValueError('No effective user in update')
  return cast(User, update.effective_user)


def get_message(update: Update) -> Message:
  if update.message is None:
    raise ValueError('No message object set')
  return cast(Message, update.message)
