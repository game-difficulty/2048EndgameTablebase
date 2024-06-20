import os
import lzma

import numpy as np
from numba import njit


@njit(nogil=True)
def compress_data_how(data):  # u32 u8 u8 u8 u8 u32
    # 存储索引的列表
    ind0 = np.empty(255, dtype='uint8,uint32')  # f4,ind1_pos
    ind1 = np.empty(65535, dtype='uint8,uint32')  # f3,ind2_pos
    ind2 = np.empty(1048575, dtype='uint8,uint32')  # f2,ind3_pos
    ind3 = np.empty(16777215, dtype='uint8,uint64')  # f1,data_pos
    ind0[0]['f0'], ind0[0]['f1'] = 0, 0
    ind1[0]['f0'], ind1[0]['f1'] = 0, 0
    ind2[0]['f0'], ind2[0]['f1'] = 0, 0
    ind3[0]['f0'], ind3[0]['f1'] = 0, 0
    i0, i1, i2, i3 = np.uint8(1), np.uint8(1), np.uint8(1), np.uint8(1)  # 初始化起始索引,每8位最大255

    d0, d1, d2, d3 = data[0]['f4'], data[0]['f3'], data[0]['f2'], data[0]['f1']  # 初始化当前跟踪的前缀值
    j0 = 0
    # 遍历数组, 每8位发生变化时插入新的节点
    for j0 in range(len(data)):
        if data[j0]['f4'] != d0:
            ind0[i0]['f0'], ind0[i0]['f1'] = d0, i1
            d0 = data[j0]['f4']
            i0 += 1
            ind1[i1]['f0'], ind1[i1]['f1'] = d1, i2
            d1 = data[j0]['f3']
            i1 += 1
            ind2[i2]['f0'], ind2[i2]['f1'] = d2, i3
            d2 = data[j0]['f2']
            i2 += 1
            ind3[i3]['f0'], ind3[i3]['f1'] = d3, j0 - 1
            d3 = data[j0]['f1']
            i3 += 1
        elif data[j0]['f3'] != d1:
            ind1[i1]['f0'], ind1[i1]['f1'] = d1, i2
            d1 = data[j0]['f3']
            i1 += 1
            ind2[i2]['f0'], ind2[i2]['f1'] = d2, i3
            d2 = data[j0]['f2']
            i2 += 1
            ind3[i3]['f0'], ind3[i3]['f1'] = d3, j0 - 1
            d3 = data[j0]['f1']
            i3 += 1
        elif data[j0]['f2'] != d2:
            ind2[i2]['f0'], ind2[i2]['f1'] = d2, i3
            d2 = data[j0]['f2']
            i2 += 1
            ind3[i3]['f0'], ind3[i3]['f1'] = d3, j0 - 1
            d3 = data[j0]['f1']
            i3 += 1
        elif data[j0]['f1'] != d3:
            ind3[i3]['f0'], ind3[i3]['f1'] = d3, j0 - 1
            d3 = data[j0]['f1']
            i3 += 1

    ind0[i0]['f0'], ind0[i0]['f1'] = d0, i1
    ind1[i1]['f0'], ind1[i1]['f1'] = d1, i2
    ind2[i2]['f0'], ind2[i2]['f1'] = d2, i3
    ind3[i3]['f0'], ind3[i3]['f1'] = d3, j0
    i0 += 1
    i1 += 1
    i2 += 1
    i3 += 1

    return ind0[:i0], ind1[:i1], ind2[:i2], ind3[:i3]


def compress_and_save(ind3, data, output_filename, lvl=1):
    start = 0
    current_size = 0

    c = 1
    segments = np.empty(int(len(data) / 16000) + 2, dtype='uint32,uint64')
    segments[0] = 0
    # 将所有压缩后的数据写入一个文件
    with open(output_filename, 'wb') as f:
        for i in range(1, len(ind3) - 1):
            end = int(ind3[i]['f1']) + 1
            if end - start > 32768:  # 约每32768条打一个包
                segment = data[start:end].tobytes()
                compressor = lzma.LZMACompressor(preset=lvl)
                block = compressor.compress(segment) + compressor.flush()
                bytes_written = f.write(block)  # 写入文件并获取写入的字节数
                current_size += bytes_written
                ind3[i]['f1'] = 0  # 更新索引，表示block之间的切换节点
                start = end

                segments[c] = (i, current_size)
                c += 1
            else:
                ind3[i]['f1'] -= start  # 更新索引为相对位置
        # 处理最后一批数据
        segment = data[start:].tobytes()
        compressor = lzma.LZMACompressor(preset=lvl)
        block = compressor.compress(segment) + compressor.flush()
        bytes_written = f.write(block)  # 写入文件并获取写入的字节数
        current_size += bytes_written
        ind3[i + 1]['f1'] = 0
        segments[c] = (i + 1, current_size)
        c += 1

    return ind3, segments[:c]


def trie_compress_progress(path, filename):
    target_file = os.path.join(path, filename[:-4] + 'zt')
    fullfilepath = os.path.join(path, filename)
    os.makedirs(target_file, exist_ok=True)

    book = np.fromfile(fullfilepath, dtype="uint32,uint8,uint8,uint8,uint8,uint32")
    ind0, ind1, ind2, ind3 = compress_data_how(book)
    book_ = np.empty(len(book), dtype='uint32,uint32')  # 后32位和成功率，后面分块压缩
    book_['f0'] = book['f0']
    book_['f1'] = book['f5']
    del book
    target_dir = os.path.join(target_file, filename[:-4])
    ind3, segments = compress_and_save(ind3, book_, target_dir + 'z', lvl=1)  # 分块压缩，更新索引并记录分块大小
    ind0['f1'] += 1 + len(ind0)
    ind1['f1'] += 1 + len(ind0) + len(ind1)
    root = np.array([(0, 1)], dtype='uint8,uint32')
    ind = np.concatenate([root, ind0, ind1, ind2])  # 把前缀树的前三层合并储存
    ind3.astype('uint8,uint16').tofile(target_dir + 'ii')
    ind.tofile(target_dir + 'i')
    segments.tofile(target_dir + 's')
    os.rename(target_file, target_file[:-1])  # .zt文件夹变成.z


@njit(nogil=True)
def _search_trie(ind, board):
    board_prefix = [np.uint8((board >> np.uint64(56)) & np.uint64(0xff)),
                    np.uint8((board >> np.uint64(48)) & np.uint64(0xff)),
                    np.uint8((board >> np.uint64(40)) & np.uint64(0xff))]

    low, mid, high = 2, 2, ind[1]['f1'] - 1
    for f in range(3):
        found = False
        while low <= high:
            mid = (low + high) // 2
            if ind[mid]['f0'] == board_prefix[f]:
                found = True
                break
            elif ind[mid]['f0'] < board_prefix[f]:
                low = mid + 1
            else:
                high = mid - 1
        if not found:
            return 0, 0  # 没找到
        else:
            low = ind[mid - 1]['f1'] + 1
            high = ind[mid]['f1']
    return low, high  # 读取.ii文件中的[low:high+1]


def search_tree(filepath, ind, segments, board):
    low, high = _search_trie(ind, board)
    if low == 0 and high == 0:
        return 0, 0, 0, np.empty(0,dtype='uint8,uint16')  # 没找到
    ind3_seg = np.fromfile(filepath + 'ii', dtype='uint8,uint16', count=high-low+2, offset=low*3-3)  # 这里多读一组

    pos1 = search_block2(ind3_seg[1:], np.uint8((board >> np.uint64(32)) & np.uint64(0xff)))  # 最后一个前缀
    if not pos1[1]:
        return 0, 0, 0, np.empty(0,dtype='uint8,uint16')   # 没找到
    start, end = _get_seg_position(segments, pos1[0] + low)

    return int(start), int(end), pos1[0], ind3_seg


@njit(nogil=True)
def _get_seg_position(segments, pos):
    #  在segments里找到小于pos1的最大值和不小于pos1的最小值,获取压缩块block的范围
    low, high = 0, len(segments) - 1
    start, end = 0, 0
    while low <= high:
        mid = (low + high) // 2
        if segments[mid]['f0'] < pos:
            low = mid + 1
        else:
            start, end = segments[mid - 1]['f1'], segments[mid]['f1']
            high = mid - 1
    return start, end


@njit(nogil=True)
def search_block(block, target):
    low, high = 0, len(block) - 1
    while low <= high:
        mid = (low + high) // 2
        if block[mid]['f0'] == target:
            return block[mid]['f1'] / 4000000000
        elif block[mid]['f0'] < target:
            low = mid + 1
        else:
            high = mid - 1
    return 0  # 没找到


@njit(nogil=True)
def search_block2(block, target):
    low, high = 0, len(block) - 1
    while low <= high:
        mid = (low + high) // 2
        if block[mid]['f0'] == target:
            return mid, True
        elif block[mid]['f0'] < target:
            low = mid + 1
        else:
            high = mid - 1
    return 0, False  # 没找到


def trie_decompress_search(filepath, board, ind, segments):
    start, end, pos, ind3_seg = search_tree(filepath, ind, segments, board)
    if not start and not end:
        return 0.0  # 没找到
    with open(filepath + 'z', 'rb') as f:
        f.seek(start)  # 定位到块的起始位置
        compressed_data = f.read(end - start)  # 读取压缩数据块
        decompressed_data = lzma.decompress(compressed_data)  # 解压数据块
    block = np.frombuffer(decompressed_data, dtype='uint32,uint32')
    target = np.uint32(board & np.uint64(0xffffffff))
    high = ind3_seg[pos + 1]['f1'] + 1
    low = ind3_seg[pos]['f1']
    if high == 1:
        high = len(block)
    if low != 0:
        low += 1
    result = search_block(block[low:high], target)
    return result
