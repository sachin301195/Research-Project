# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 12:19:42 2021

@author: busach
"""

import subprocess
import logging
import multiprocessing
import time
import matplotlib.pyplot as plt
import psutil
import pathlib as Path
import os
import sys

def configure_logger():
    timestamp = time.strftime("%Y-%m-%d")
    _logger = logging.getLogger(__name__)
    _logger.setLevel(logging.INFO)
    Path("./logs").mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler('./logs/application-main-'+timestamp+'.log')
    file_handler.setLevel(logging.INFO)
    _logger.addHandler(file_handler)
    formatter = logging.Formatter('%(asctime)s  %(name)s  %(levelname)s: %(message)s')
    file_handler.setFormatter(formatter)
    return _logger  

logger = configure_logger()

