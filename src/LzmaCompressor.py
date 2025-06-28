import os
import subprocess
import lzma
from Config import logger


internal_path = os.path.join("_internal", "7z.exe")

if os.path.exists(internal_path):
    seven_zip_exe = internal_path
else:
    seven_zip_exe = os.path.join("7zip", "7z.exe")
    if not os.path.exists(seven_zip_exe):
        logger.warning(f"7z executable not found!")


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
        subprocess.run(cmd, check=True)
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
        subprocess.run(cmd, check=True)
        logger.debug(f"decompressed: {original_file}")
        # 删除压缩文件
        os.remove(archive_file)
    else:
        # 如果 7z.exe 不可用，则使用 Python 标准库 lzma 解压
        decompress_with_lzma(archive_file, original_file)


# if __name__ == "__main__":
    # compress_with_7z(r"Q:\tables\4431_2048_725.book", lvl=1)
    # decompress_with_7z(r"Q:\tables\4431_2048_725.book.7z")
