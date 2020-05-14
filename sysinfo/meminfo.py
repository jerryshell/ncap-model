def meminfo():
    meminfo = {}

    with open('/proc/meminfo') as f:
        for line in f:
            line_split = line.split(':')
            key = line_split[0].strip()
            value = line_split[1].strip()
            meminfo[key] = value

    return meminfo


if __name__ == '__main__':
    meminfo = meminfo()
    print('---')
    print(meminfo)
    print('---')
    for k, v in meminfo.items():
        print(k, v)
    print('---')
    print('MemTotal: {0}'.format(meminfo['MemTotal']))
    print('MemFree: {0}'.format(meminfo['MemFree']))
    print('SwapTotal: {0}'.format(meminfo['SwapTotal']))
    print('SwapFree: {0}'.format(meminfo['SwapFree']))
