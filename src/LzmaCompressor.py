import bisect
import concurrent.futures
import lzma
import math
import os
import subprocess
from subprocess import CREATE_NO_WINDOW

import numpy as np

from Config import logger

internal_path = os.path.join("_internal", "7z.exe")

if os.path.exists(internal_path):
    seven_zip_exe = internal_path
else:
    seven_zip_exe = os.path.join("7zip", "7z.exe")
    if not os.path.exists(seven_zip_exe):
        logger.warning(f"7z executable not found!")
        
        
BLOCK_SIZE = 32768


def is_seven_zip_available():
    """
    检查 seven_zip_exe 是否存在且可用
    """
    return os.path.exists(seven_zip_exe) and os.access(seven_zip_exe, os.X_OK)


def compress_with_lzma(input_file, output_file):
    """
    使用 Python 标准库 lzma 将文件压缩为 .xz 格式，并删除原文件
    """
    # 使用 lzma 的 FORMAT_XZ 格式，生成 .xz 文件
    with open(input_file, 'rb') as f_in, lzma.open(output_file, 'wb', format=lzma.FORMAT_XZ) as f_out:
        f_out.write(f_in.read())
    logger.debug(f"文件已使用 lzma 压缩为 .xz 格式到: {output_file}")
    # 删除原文件
    os.remove(input_file)


def decompress_with_lzma(input_file, output_file):
    """
    使用 Python 标准库 lzma 解压文件，并删除压缩文件
    """
    with lzma.open(input_file, 'rb') as f_in, open(output_file, 'wb') as f_out:
        f_out.write(f_in.read())
    # logger.debug(f"文件已使用 lzma 解压到: {output_file}")
    # 删除压缩文件
    os.remove(input_file)


def compress_with_7z(input_file, lvl=1):
    """
    使用 7z.exe 或 lzma 将指定文件压缩为同名 .7z 文件，并删除原文件
    """
    output_file = f"{input_file}.7z"
    if not os.path.exists(input_file):
        logger.debug(f"FileNotFound: {input_file}")
        return

    if is_seven_zip_available():
        # 如果 7z.exe 可用，则使用 7z.exe 压缩
        max_threads = os.cpu_count()
        cmd = [
            seven_zip_exe, "a",  # "a" 表示添加文件到压缩包
            "-t7z",  # 使用 7z 格式
            "-m0=lzma2",  # 使用 LZMA2 算法
            f"-mx={lvl}",  # 压缩级别
            f"-mmt={max_threads}",  # 使用最大线程数
            "-sdel",  # 压缩后删除原始文件
            output_file,  # 输出的 .7z 文件路径
            input_file  # 输入的文件路径
        ]
        subprocess.run(cmd, check=True, creationflags=CREATE_NO_WINDOW)
        logger.debug(f"compressed: {output_file}")
    else:
        # 如果 7z.exe 不可用，则使用 Python 标准库 lzma 压缩
        compress_with_lzma(input_file, output_file)


def decompress_with_7z(archive_file):
    """
    使用 7z.exe 或 lzma 将指定 .7z 文件解压回原始文件路径，并删除压缩文件
    """
    original_file = os.path.splitext(archive_file)[0]  # 去掉 .7z 后缀
    if not os.path.exists(archive_file):
        logger.debug(f"FileNotFound: {archive_file}")
        return

    if is_seven_zip_available():
        # 如果 7z.exe 可用，则使用 7z.exe 解压
        cmd = [
            seven_zip_exe, "x",  # "x" 表示解压文件到指定路径
            archive_file,  # 输入的 .7z 文件路径
            f"-o{os.path.dirname(original_file)}",  # 解压到原文件所在目录
            "-y"  # 自动确认所有提示（如覆盖文件）
        ]
        subprocess.run(cmd, check=True, creationflags=CREATE_NO_WINDOW)
        logger.debug(f"decompressed: {original_file}")
        # 删除压缩文件
        os.remove(archive_file)
    else:
        # 如果 7z.exe 不可用，则使用 Python 标准库 lzma 解压
        decompress_with_lzma(archive_file, original_file)


def compress_uint64_array(data, output_filepath, lvl=1):
    if len(data) < 65536:
        return
    if len(data) > 4194304:
        segments = _compress_uint64_array_p(data, output_filepath + '.zi', lvl)
        segments.tofile(output_filepath + '.s')
        return

    # 存储分段信息：每段的第一个值和文件偏移量
    segments = np.empty(math.ceil(len(data) / BLOCK_SIZE), dtype='uint64,uint64')
    current_offset = 0

    # 打开输出文件
    with open(output_filepath + '.zi', 'wb') as f:
        i = 0
        n = len(data)

        while i < n:
            end = min(i + BLOCK_SIZE, n)
            segment_data = data[i:end]
            first_value = data[i]
            data_bytes = segment_data.tobytes()
            compressor = lzma.LZMACompressor(preset=lvl)
            compressed_data = compressor.compress(data_bytes) + compressor.flush()
            bytes_written = f.write(compressed_data)
            segments[i // BLOCK_SIZE] = (first_value, current_offset)
            current_offset += bytes_written
            i = end

    segments.tofile(output_filepath + '.s')


def _compress_uint64_array_p(data, output_filename, lvl=1):
    if len(data) == 0:
        return np.array([], dtype='uint64,uint64')

    num_workers = os.cpu_count()

    # 计算总段数和每个工作线程处理的段数
    total_segments = math.ceil(len(data) / BLOCK_SIZE)
    segments_per_worker = math.ceil(total_segments / num_workers)

    all_segments = np.empty(total_segments, dtype='uint64,uint64')

    # 准备批量数据
    batch_data = []
    batch_indices = []
    batch_first_values = []

    for seg_idx in range(total_segments):
        start_idx = seg_idx * BLOCK_SIZE
        end_idx = min((seg_idx + 1) * BLOCK_SIZE, len(data))
        segment_data = data[start_idx:end_idx]
        first_value = data[start_idx] if len(segment_data) > 0 else 0

        batch_data.append(segment_data)
        batch_indices.append(seg_idx)
        batch_first_values.append(first_value)

    # 分批处理
    batch_size = segments_per_worker
    all_results = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []

        for i in range(0, total_segments, batch_size):
            end_idx = min(i + batch_size, total_segments)

            # 提交批量任务
            future = executor.submit(
                worker_batch,
                0,  # worker_id
                batch_data[i:end_idx],
                batch_indices[i:end_idx],
                batch_first_values[i:end_idx],
                lvl
            )
            futures.append(future)

        # 收集所有结果
        for future in concurrent.futures.as_completed(futures):
            all_results.extend(future.result())

    # 按段索引排序
    all_results.sort(key=lambda x: x[0])

    # 写入最终文件
    current_offset = 0
    with open(output_filename, 'wb') as f:
        for seg_idx, compressed_data, first_value in all_results:
            bytes_written = f.write(compressed_data)
            all_segments[seg_idx] = (first_value, current_offset)
            current_offset += bytes_written

    return all_segments


def worker_batch(worker_id, data_slice, segment_indices, first_values, lvl):
    results = []
    for i, (seg_data, seg_idx, first_val) in enumerate(zip(data_slice, segment_indices, first_values)):
        data_bytes = seg_data.tobytes()
        compressor = lzma.LZMACompressor(preset=lvl)
        compressed_data = compressor.compress(data_bytes) + compressor.flush()
        results.append((seg_idx, compressed_data, first_val))
    return results


def decompress_uint64_array(input_filename, segments):
    if len(segments) == 0:
        return np.array([], dtype=np.uint64)

    all_data = []

    with open(input_filename, 'rb') as f:
        for i in range(len(segments)):
            # 获取当前段的偏移量和下一个段的偏移量
            current_offset = segments[i]['f1']
            next_offset = segments[i + 1]['f1'] if i + 1 < len(segments) else None

            # 定位到当前段
            f.seek(current_offset)

            # 读取压缩数据
            if next_offset is not None:
                compressed_size = next_offset - current_offset
                compressed_data = f.read(compressed_size)
            else:
                # 最后一段，读取到文件末尾
                compressed_data = f.read()

            # 解压数据
            decompressor = lzma.LZMADecompressor()
            decompressed_data = decompressor.decompress(compressed_data)

            # 将字节转换为uint64数组
            segment_array = np.frombuffer(decompressed_data, dtype=np.uint64)
            all_data.append(segment_array)

    # 合并所有段
    return np.concatenate(all_data) if all_data else np.array([], dtype=np.uint64)


def find_value_uint64_compressed(input_filename, segments, value):
    """
    在不完全解压的情况下，找到特定值在原数组中的索引，如果未找到则返回None
    """
    seg_first_values = segments['f0']
    if len(segments) == 0 or value < seg_first_values[0]:
        return None

    seg_idx = bisect.bisect_left(seg_first_values, value)

    if seg_idx < len(seg_first_values) and seg_first_values[seg_idx] == value:
        return seg_idx * BLOCK_SIZE
    seg_idx -= 1

    # 解压找到的段
    segment = segments[seg_idx]
    file_offset = segment['f1']

    # 读取并解压该段
    with open(input_filename, 'rb') as f:
        if seg_idx + 1 < len(segments):
            # 有下一段，则读取到下一段的偏移量
            next_offset = segments[seg_idx + 1]['f1']
            compressed_size = next_offset - file_offset
        else:
            # 最后一段，读取到文件末尾
            f.seek(0, 2)  # 定位到文件末尾
            file_size = f.tell()
            compressed_size = file_size - file_offset

        f.seek(file_offset)
        compressed_data = f.read(compressed_size)

    decompressor = lzma.LZMADecompressor()
    decompressed_data = decompressor.decompress(compressed_data)
    segment_data = np.frombuffer(decompressed_data, dtype=np.uint64)

    # 在段内进行二分查找
    idx_in_segment = bisect.bisect_left(segment_data, value)

    # 检查是否找到
    if idx_in_segment < len(segment_data) and segment_data[idx_in_segment] == value:
        # 计算全局索引
        global_index = seg_idx * BLOCK_SIZE + idx_in_segment
        return global_index
    else:
        return None


# if __name__ == "__main__":
    # compress_with_7z(r"Q:\tables\4431_2048_725.book", lvl=1)
    # decompress_with_7z(r"Q:\tables\4431_2048_725.book.7z")

# if __name__ == "__main__":
#     test_data = np.fromfile(r"C:\Users\Administrator\Desktop\test\4.i", dtype=np.uint64)
#     import time
#     t0=time.time()
#     output_file = r"C:\Users\Administrator\Desktop\test\compressed_data.bin"
#     segments = compress_uint64_array(test_data, output_file, lvl=1)
#     print(time.time()-t0)
#     segments.tofile(r"C:\Users\Administrator\Desktop\test\seg")
#
#     decompressed_data = decompress_uint64_array(output_file, segments)
#     assert np.array_equal(test_data, decompressed_data)
#
#     test_ind = [0,1,2,50000,65534,65535,65536,65537,262143,262144,-3,-1]
#     test_values = [test_data[i] for i in test_ind if i < len(test_data)]
#
#     for test_value in test_values:
#         index = find_value_uint64_compressed(output_file, segments, test_value)
#         assert test_value == test_data[index]
