def loadavg():
    loadavg = {}

    with open('/proc/loadavg') as f:
        loadavg_split = f.read().split()
        f.close()
        loadavg['loadavg_1'] = loadavg_split[0]
        loadavg['loadavg_5'] = loadavg_split[1]
        loadavg['loadavg_15'] = loadavg_split[2]
        loadavg['nr'] = loadavg_split[3]
        loadavg['last_pid'] = loadavg_split[4]

    return loadavg


if __name__ == '__main__':
    loadavg = loadavg()
    print('---')
    print(loadavg)
    print('---')
    for k, v in loadavg.items():
        print(k, v)
