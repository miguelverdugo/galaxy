# -*- coding: utf-8 -*-
import os
import inspect

from . import models
from .models import *
from galaxy.galaxy import galaxysource

__pkg_dir__ = os.path.dirname(inspect.getfile(inspect.currentframe()))
__data_dir__ = os.path.join(__pkg_dir__, "data")
