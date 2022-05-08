import os
import json
import sys
import numpy as np
import logging
import time

start_time = time.time()

def mkdir_safe(directory):
    try:
        os.makedirs(directory)
    except FileExistsError:
        pass


def log_message(message):
    elapsed_time = time.time() - start_time
    print("[%.2f s] %s" % (elapsed_time, message))
    logging.debug("[%.2f s] %s" % (elapsed_time, message))


def mute():
    # disable render output
    logfile = '/dev/null'
    open(logfile, 'a').close()
    old = os.dup(1)
    sys.stdout.flush()
    os.close(1)
    os.open(logfile, os.O_WRONLY)
    return old


def unmute(old):
    # disable output redirection
    os.close(1)
    os.dup(old)
    os.close(old)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)