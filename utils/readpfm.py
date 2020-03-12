import re
import numpy as np
import sys
 

def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('latin-1') #读取第一行
    if 'PF' in header:
        color = True
    elif 'Pf' in header:
        color = False
    else:
        raise Exception('Not a PFM file.')

    line = file.readline().decode('latin-1') #这里读取第二行
    width, height = re.findall('\d+', line) #找到line中的数字的字符串，作为列表返回
    width = int(width)
    height = int(height)

    scale = float(file.readline().rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale
