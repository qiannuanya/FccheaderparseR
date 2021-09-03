
import datetime
import logging
import logging.handlers
import os
import shutil


def _timestamp():
    now = datetime.datetime.now()
    now_str = now.strftime("%Y%m%d%H%M")
    ret