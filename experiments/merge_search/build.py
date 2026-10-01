import os
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXE = HERE / ('merge_probe.exe' if os.name == 'nt' else 'merge_probe')


def environment():
    compiler = os.environ.get('CXX') or shutil.which('g++')
    if not compiler and Path('C:/Apps/mingw64/bin/g++.exe').exists():
        compiler = 'C:/Apps/mingw64/bin/g++.exe'
    if not compiler:
        raise RuntimeError('A C++17 compiler is required; set CXX')
    env = dict(os.environ)
    env['PATH'] = str(Path(compiler).resolve().parent) + os.pathsep + env.get('PATH', '')
    return compiler, env


def build():
    compiler, env = environment()
    command = [compiler, '-std=c++17', '-O3', '-I' + str(ROOT / 'native_core/include'),
               str(HERE / 'merge_probe.cpp'), '-o', str(EXE)]
    subprocess.run(command, env=env, check=True, timeout=90)
    return command


if __name__ == '__main__':
    print(build())
