#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2026 fred <github-fred@hidzz.com>
#
# Distributed under terms of the BSD 3-Clause license.

import logging
from pathlib import Path

import toml

CONFIG_FILES = ("qrzbot.toml", "~/.local/qrzbot.toml", "/etc/qrzbot.toml")
POLL_TRACKING_FILE = "/var/tmp/poll_tracking.json"
QUESTIONS_FILE_CSV = "/var/tmp/questions.csv"

logging.basicConfig(
  format="%(asctime)s - %(name)s[%(process)d]:%(lineno)d - %(levelname)s - %(message)s",
  datefmt='%H:%M:%S',
  level=logging.INFO
)
LOG = logging.getLogger(__name__)


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
  poll_tracking_file: str = POLL_TRACKING_FILE
  questions_file: str = QUESTIONS_FILE_CSV

  def __new__(cls, *args, **kwargs):
    raise TypeError(f'{cls.__name__} is a static class and cannot be instanciated')

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
