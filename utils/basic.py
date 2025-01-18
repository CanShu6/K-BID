import logging
import os


def create_log_id(dir_path):
    log_count = 0
    file_path = os.path.join(dir_path, 'log{:d}.log'.format(log_count))
    while os.path.exists(file_path):
        log_count += 1
        file_path = os.path.join(dir_path, 'log{:d}.log'.format(log_count))
    return log_count


def logging_config(folder=None, name=None, level=logging.DEBUG,
                   console_level=logging.DEBUG, no_console=True):

    if not os.path.exists(folder):
        os.makedirs(folder)
    for handler in logging.root.handlers:
        logging.root.removeHandler(handler)
    logging.root.handlers = []
    log_path = os.path.join(folder, name + ".log")
    print("All logs will be saved to %s" % log_path)

    logging.root.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logfile = logging.FileHandler(log_path)
    logfile.setLevel(level)
    logfile.setFormatter(formatter)
    logging.root.addHandler(logfile)

    if not no_console:
        log_console = logging.StreamHandler()
        log_console.setLevel(console_level)
        log_console.setFormatter(formatter)
        logging.root.addHandler(log_console)
    return folder
