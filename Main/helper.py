def progress(count, total):
    percent = (count / total) * 100
    print('\r', "%.2f" % round(percent, 2) + "% completed", end=' ')