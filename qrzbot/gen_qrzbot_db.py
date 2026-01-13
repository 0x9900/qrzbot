#! /usr/bin/env python
# vim:fenc=utf-8
#
# Copyright © 2025 fred <github-fred@hidzz.com>
#
# Distributed under terms of the BSD 3-Clause license.
#

import csv
import logging
import os
import sqlite3
import sys
from pathlib import Path

import requests

from qrzbot import Config

DETECT_TYPES = sqlite3.PARSE_DECLTYPES

BUFSIZE = 1024

SQL_TABLES = """
PRAGMA synchronous = EXTRA;
PRAGMA journal_mode = WAL;
CREATE TABLE IF NOT EXISTS pota_parks (
  reference TEXT NOT NULL,
  name TEXT NOT NULL,
  active BOOL,
  entityId INTEGER NOT NULL,
  locationDesc TEXT NOT NULL,
  latitude FLOAT,
  longitude FLOAT,
  grid TEXT
);

CREATE TABLE IF NOT EXISTS sota_summit (
  SummitCode TEXT NOT NULL,
  AssociationName TEXT,
  RegionName TEXT,
  SummitName TEXT,
  AltM FLOAT,
  AltFt FLOAT,
  GridRef1 FLOAT,
  GridRef2 FLOAT,
  Longitude FLOAT,
  Latitude FLOAT,
  Points INTEGER,
  BonusPoints INTEGER,
  ValidFrom DATE,
  ValidTo DATE,
  ActivationCount INTEGER,
  ActivationDate DATE,
  ActivationCall TEXT
);

CREATE TABLE IF NOT EXISTS states (
id TEXT,
country_id INTEGER,
country_code TEXT,
country_name TEXT,
state_code TEXT,
name TEXT,
type TEXT,
latitude FLOAT,
longitude FLOAT
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_reference ON pota_parks(reference);

CREATE UNIQUE INDEX IF NOT EXISTS idx_SummitCode ON sota_summit(SummitCode);

CREATE UNIQUE INDEX IF NOT EXISTS idx_country_state on states(country_code, state_code);

"""

INSERT_PARK = '''
INSERT INTO pota_parks
VALUES (
  :reference, :name, :active, :entityId, :locationDesc, :latitude, :longitude, :grid
)'''

INSERT_SUMMIT = '''
INSERT INTO sota_summit VALUES (
  :SummitCode, :AssociationName, :RegionName, :SummitName, :AltM,
  :AltFt, :GridRef1, :GridRef2, :Longitude, :Latitude, :Points, :BonusPoints,
  :ValidFrom, :ValidTo, :ActivationCount, :ActivationDate, :ActivationCall
)'''

INSERT_STATE = '''
INSERT INTO states VALUES (
  :id, :country_id, :country_code, :country_name, :state_code, :name, :type,
  :latitude, :longitude
)'''

SOTA_SUMMITS = "https://storage.sota.org.uk/summitslist.csv"
ALL_PARKS = "https://pota.app/all_parks_ext.csv"
STATES_SHA = "f6c8a63567ccbae0fce8a8ab5bde2a97"
STATES = (f"https://gist.githubusercontent.com/0x9900/{STATES_SHA}"
          "/raw/307623e9bbadc07f8de1a9b333ade26402e654dd/states.csv")

logging.basicConfig(
  format="%(asctime)s - %(name)s[%(process)d]:%(lineno)d - %(levelname)s - %(message)s",
  datefmt='%H:%M:%S',
  level=logging.INFO
)
# set higher logging level for httpx to avoid all GET and POST requests being logged
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def download_csv(url, filename):
  logger.info('Downloading: %s', url)
  response = requests.get(url, timeout=30, stream=True)
  with open(filename, "wb") as file:
    for chunk in response.iter_content(chunk_size=8192):  # Read in chunks
      file.write(chunk)


def connect_db(db_name: str, timeout: int = 5) -> sqlite3.Connection:
  try:
    conn = sqlite3.connect(db_name, timeout=timeout,
                           detect_types=DETECT_TYPES, isolation_level=None)
    logger.debug("Database: %s", db_name)
  except sqlite3.OperationalError as err:
    logger.error("Database: %s - %s", db_name, err)
    sys.exit(os.EX_IOERR)
  return conn


def create_db(db_name: str) -> None:
  with connect_db(db_name) as conn:
    curs = conn.cursor()
    curs.executescript(SQL_TABLES)


def insert_pota(dbname, filename):
  logger.info("Create pota table")
  parkfile = Path(filename)
  with parkfile.open('r', encoding="utf-8") as fd:
    reader = csv.DictReader(fd)
    with connect_db(dbname) as db:
      entries = []
      for idx, row in enumerate(reader):
        entries.append(row)
        if idx % BUFSIZE == 0:
          db.executemany(INSERT_PARK, entries)
          entries.clear()
      if entries:
        db.executemany(INSERT_PARK, entries)


def insert_summit(dbname, filename):
  logger.info('Create summit table')
  summitfile = Path(filename)
  with summitfile.open('r', encoding="utf-8") as fd:
    fd.readline()
    reader = csv.DictReader(fd)
    with connect_db(dbname) as db:
      entries = []
      for idx, row in enumerate(reader):
        entries.append(row)
        if idx % BUFSIZE == 0:
          db.executemany(INSERT_SUMMIT, entries)
          entries.clear()
      if entries:
        db.executemany(INSERT_SUMMIT, entries)


def insert_states(dbname, filename):
  logger.info('Create state table')
  statefile = Path(filename)
  with statefile.open('r', encoding="utf-8") as fd:
    reader = csv.DictReader(fd)
    with connect_db(dbname) as db:
      entries = []
      for idx, rec in enumerate(reader):
        entries.append(rec)
        if idx % BUFSIZE == 0:
          db.executemany(INSERT_STATE, entries)
          entries.clear()
      if entries:
        db.executemany(INSERT_STATE, entries)


def main():
  Config.load()
  db_name = Path(Config.dbname).expanduser().absolute()
  logger.info('Database name: %s', db_name)

  tasks = [
    {"func": insert_states, "url": STATES, "filename": '/tmp/states.csv'},
    {"func": insert_summit, "url": SOTA_SUMMITS, "filename": '/tmp/summits.csv'},
    {"func": insert_pota, "url": ALL_PARKS, "filename": '/tmp/park.csv'},
  ]

  if db_name.exists():
    db_name.unlink()

  create_db(db_name)
  for task in tasks:
    try:
      download_csv(task["url"], task["filename"])
      task['func'](db_name, task["filename"])
    except sqlite3.IntegrityError as err:
      logger.info("%r - %s", task['func'], err)
      return


if __name__ == "__main__":
  main()
