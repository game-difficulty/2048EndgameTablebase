import os
import lzma
import concurrent.futures
from typing import List
from multiprocessing import shared_memory, freeze_support
import time

import numpy as np
from numba import njit


@njit(nogil=True)
def compress_data_how(data):  # u32 u8 u8 u8 u8 u32
    # 存储索引的列表
    ind0 = np.empty(255, dtype='uint8,uint32')  # f4,ind1_pos
    ind1 = np.empty(65535, dtype='uint8,uint32')  # f3,ind2_pos
    ind2 = np.empty(4194303, dtype='uint8,uint32')  # f2,ind3_pos
    ind3 = np.empty(134217727, dtype='uint8,uint64')  # f1,data_pos
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


# 单线程
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


def compress_segment(segment_data, lvl):
    compressor = lzma.LZMACompressor(preset=lvl)
    block = compressor.compress(segment_data) + compressor.flush()
    return block


def worker(start_idx, end_idx, segments_info, shm_name, shape, dtype, lvl, temp_filename, worker_idx):
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    data_slice = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
    current_size = 0
    segments = []

    with open(temp_filename, 'wb') as f:
        for i in range(start_idx, end_idx):
            start = segments_info[i]['f0']
            end = segments_info[i]['f1']
            segment_data = data_slice[start:end].tobytes()
            compressed_data = compress_segment(segment_data, lvl)
            bytes_written = f.write(compressed_data)
            current_size += bytes_written
            segments.append((segments_info[i]['f2'], current_size))

    existing_shm.close()
    return segments, worker_idx


@njit(nogil=True)
def get_segment_pos(ind3, len_data):
    segments_info = np.empty(ind3[-1]['f1'] // 16384, dtype='uint32,uint32,uint32')
    c = 0
    start = 0
    for i in range(1, len(ind3) - 1):
        end = int(ind3[i]['f1']) + 1
        if end - start > 32768:  # 约每32768条打一个包
            segments_info[c]['f0'], segments_info[c]['f1'], segments_info[c]['f2'] = start, end, i
            start = end
            ind3[i]['f1'] = 0  # 更新索引，表示block之间的切换节点
            c += 1
        else:
            ind3[i]['f1'] -= start  # 更新索引为相对位置
    ind3[len(ind3) - 1]['f1'] = 0
    # 处理最后一批数据
    segments_info[c]['f0'], segments_info[c]['f1'], segments_info[c]['f2'] = start, len_data, len(ind3) - 1
    c += 1
    segments_info = segments_info[:c]
    return segments_info, ind3


def compress_and_save_p(ind3, data, output_filename, lvl=1):
    num_workers = max(2, os.cpu_count())

    # 计算所有分段的位置
    segments_info, ind3 = get_segment_pos(ind3, len(data))

    # 创建共享内存
    shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
    shm_data = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
    np.copyto(shm_data, data)

    segment_size = len(segments_info) // num_workers
    futures = []

    # 并行压缩
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        for worker_idx in range(num_workers):
            start_idx = worker_idx * segment_size
            end_idx = (worker_idx + 1) * segment_size if worker_idx < num_workers - 1 else len(segments_info)
            temp_filename = output_filename + f"{worker_idx}"
            futures.append(executor.submit(
                worker, start_idx, end_idx, segments_info, shm.name, shm_data.shape, shm_data.dtype,
                lvl, temp_filename, worker_idx))

    # 处理分段数据
    all_segments: List[np.ndarray | None] = [None] * num_workers
    for future in concurrent.futures.as_completed(futures):
        segments, worker_idx = future.result()
        all_segments[worker_idx] = np.array(segments, dtype='uint32,uint64')
    for i in range(1, num_workers):
        all_segments[i]['f1'] += all_segments[i - 1][-1]['f1']
    all_segments = [np.array([(0, 0)], dtype='uint32,uint64')] + all_segments
    all_seg: np.ndarray = np.concatenate(all_segments)

    # 合并所有临时文件
    with open(output_filename, 'wb') as final_file:
        for worker_idx in range(num_workers):
            temp_filename = output_filename + f"{worker_idx}"
            with open(temp_filename, 'rb') as temp_file:
                final_file.write(temp_file.read())
            os.remove(temp_filename)

    return ind3, all_seg


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
    if len(book_) >= 2097152:
        func = compress_and_save_p
    else:
        func = compress_and_save
    ind3, segments = func(ind3, book_, target_dir + 'z', lvl=1)  # 分块压缩，更新索引并记录分块大小

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
        return 0, 0, 0, np.empty(0, dtype='uint8,uint16')  # 没找到
    ind3_seg = np.fromfile(filepath + 'ii', dtype='uint8,uint16', count=high - low + 2, offset=low * 3 - 3)  # 这里多读一组

    pos1 = search_block2(ind3_seg[1:], np.uint8((board >> np.uint64(32)) & np.uint64(0xff)))  # 最后一个前缀
    if not pos1[1]:
        return 0, 0, 0, np.empty(0, dtype='uint8,uint16')  # 没找到
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
        if target == block['f0'][0]:
            return block['f1'][0] / 4000000000
        high = len(block)
    if low != 0:
        low += 1
    result = search_block(block[low:high], target)
    return result


"""
以下代码用于根据压缩后的文件还原原始数组
"""


def restore_indices(i_file):
    # 读取合并后的前三级索引 (root + ind0 + ind1 + ind2)
    ind = np.fromfile(i_file, dtype=[('f0', 'uint8'), ('f1', 'uint32')])

    # ind[0]为root节点, root = (0,1)

    # 还原ind0:
    idx = 2
    # 我们期待f0从0开始递增，并在某处再次出现0代表ind0结束
    while idx < len(ind):
        if ind[idx - 1]['f0'] > ind[idx]['f0']:
            break
        idx += 1

    ind0 = ind[1:idx].copy()

    # 还原ind1:
    # 当前idx位置为ind1的起始标记
    # 在ind1中，f1字段递增，直到f1再度出现0时结束
    start1 = idx
    idx += 1
    while idx < len(ind):
        if ind[idx]['f1'] == 0 and ind[idx]['f1'] < ind[idx - 1]['f1']:
            break
        idx += 1

    ind1 = ind[start1:idx].copy()

    # 剩余的即为ind2
    ind2 = ind[idx:].copy()

    # 逆向处理偏移：
    # 压缩时执行过：
    # ind0['f1'] += 1 + len(ind0)
    # ind1['f1'] += 1 + len(ind0) + len(ind1)
    ind0['f1'] -= (1 + len(ind0))
    ind1['f1'] -= (1 + len(ind0) + len(ind1))

    return ind0, ind1, ind2


def restore_book_and_ind3(z_file, s_file, ii_file):
    # 1. 还原book_数组
    # 读取segments
    segments = np.fromfile(s_file, dtype=[('f0', 'uint32'), ('f1', 'uint64')])
    # segments中: segments[0]=(0,0), segments[i]=(block_end_ind3, cumulative_size)
    # 块数 = len(segments)-1

    # 根据segments从z_file中解压所有块
    block_lengths = []
    decompressed_blocks = []

    with open(z_file, 'rb') as f:
        for i in range(1, len(segments)):
            block_start = segments[i - 1]['f1']
            block_end = segments[i]['f1']
            block_size = block_end - block_start
            f.seek(block_start)
            block_data = f.read(block_size)
            # 解压
            decompressed_data = lzma.decompress(block_data)
            # 每条记录8字节(uint32,uint32)
            num_records = len(decompressed_data) // 8
            block_lengths.append(num_records)
            decompressed_blocks.append(decompressed_data)

    # 拼接所有解压块形成原始数据数组book_
    book_ = np.frombuffer(b''.join(decompressed_blocks), dtype=[('f0', 'uint32'), ('f5', 'uint32')])

    # 2. 重建ind3
    # 从ii_file中读取压缩后的ind3
    ind3 = np.fromfile(ii_file, dtype=[('f0', 'uint8'), ('f1', 'uint16')]).astype('uint8,uint64')

    # k为每块长度数组的累积和，用于重建绝对位置
    k = np.array(block_lengths, dtype=np.uint64)
    k_cumsum = np.cumsum(k)

    # 在处理ind3时，如果f1=0表示块分界点，需要换下一个块索引
    # 如果f1!=0 则f1 += 对应块的起始偏移 (前面块的总和)
    block_idx = -1
    offset = 0
    for i in range(len(ind3)):
        if ind3[i]['f1'] == 0:
            is_boundary = i in segments['f0']
            if is_boundary:
                # 说明是块分界点，块索引加1
                block_idx += 1
                offset = 0 if block_idx == 0 else k_cumsum[block_idx - 1]
            ind3[i]['f1'] += offset
            if is_boundary and block_idx > 0:
                ind3[i]['f1'] -= 1
        else:
            # 非0，说明这条记录在当前块内的offset，需要加上该块起始的行偏移量
            # k_cumsum[block_idx-1]是当前块的起始偏移，但当block_idx=0时，没有前块，
            # 因此对第一个块(block_idx=0)来说，起始偏移量为0，不用加。
            offset = 0 if block_idx == 0 else k_cumsum[block_idx - 1]
            ind3[i]['f1'] += offset

    return book_, ind3


def restore_book(path):
    """把压缩后的book还原"""
    z_file, s_file = path + "z", path + "s"
    ii_file = path + "ii"
    i_file = path + "i"

    book_, ind3 = restore_book_and_ind3(z_file, s_file, ii_file)
    ind0, ind1, ind2 = restore_indices(i_file)
    book = np.empty(len(book_), dtype='uint32,uint8,uint8,uint8,uint8,uint32')
    # 先填充已知的列 f0, f5
    book['f0'] = book_['f0']
    book['f5'] = book_['f5']
    book = _restore_book(ind0, ind1, ind2, ind3, book)
    book.tofile(path + 'book')


@njit(nogil=True)
def _restore_book(ind0, ind1, ind2, ind3, book):
    # 对 ind0 层遍历 (f4)
    for i0 in range(1, len(ind0)):
        f4_val = ind0[i0]['f0']
        i1_end = int(ind0[i0]['f1'] + 1)
        i1_start = int(ind0[i0 - 1]['f1'] + 1)

        # 对 ind1 层遍历 (f3)
        for i1 in range(i1_start, i1_end):
            f3_val = ind1[i1]['f0']
            i2_end = int(ind1[i1]['f1'] + 1)
            i2_start = int(ind1[i1 - 1]['f1'] + 1)

            # 对 ind2 层遍历 (f2)
            for i2 in range(i2_start, i2_end):
                f2_val = ind2[i2]['f0']
                i3_end = int(ind2[i2]['f1'] + 1)
                i3_start = int(ind2[i2 - 1]['f1'] + 1)

                # 对 ind3 层遍历 (f1)
                for i3 in range(i3_start, i3_end):
                    f1_val = ind3[i3]['f0']
                    i3_end = int(ind3[i3]['f1'] + 1)
                    i3_start = int(ind3[i3 - 1]['f1'] + 1) if i3 != 1 else 0
                    # 填充 [start_line, end_line] 行的 f4,f3,f2,f1 值
                    book['f4'][i3_start:i3_end] = f4_val
                    book['f3'][i3_start:i3_end] = f3_val
                    book['f2'][i3_start:i3_end] = f2_val
                    book['f1'][i3_start:i3_end] = f1_val
    return book


# if __name__ == '__main__':
#     freeze_support()
#     t0 = time.time()
#     trie_compress_progress(r"Q:\tables\test", "444_2048_0_354.book")
#     print(time.time() - t0)

# if __name__ == '__main__':
# restore_book(r"D:\2048calculates\table\free10w_1024\free10w_1024_399.z\free10w_1024_399.")

# if __name__ == '__main__':
#     _path = r'D:/2048calculates/table/free8w_128\free8w_128_40.z\free8w_128_40.'
#     _ind = np.fromfile(_path + 'i', dtype='uint8,uint32')
#     _segments = np.fromfile(_path + 's', dtype='uint32,uint64')
#     _result = trie_decompress_search(_path, np.uint64(320262826360831), _ind, _segments)
#     print(_result)
