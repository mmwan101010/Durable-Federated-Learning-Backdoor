import datetime


def get_strNowTime():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')