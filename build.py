#!/usr/bin/env python3
"""Build the paper."""

import os
import sys
import subprocess
import platform

if platform.system() == 'Windows':
    python = 'python'
else:
    python = 'python3'

project = "6gxsn"


def print_with_line(s, char='#'):
    s += ' '
    s += char * (80 - len(s))
    print(s)


def print_then_call(*args, **kwargs):
    print_with_line(' '.join(args), '-')
    subprocess.run(args, check=True, **kwargs)


def fetch_data():
    print_with_line('fetch data')
    print_then_call("osf", "-p", project, "clone", os.path.join("data"))
    for folder in ["refractive index", "heterostructure", "ws2 monolayers"]:
        print_then_call(
            "tar",
            "-xf",
            os.path.join("data", "osfstorage", f"{folder}.zip"),
            "-C",
            os.path.join("data")
        )
 

def build_data():
    print_with_line('workup data')
    print_then_call(python, os.path.join('data', 'workup_heterostructure.py'))
    print_then_call(python, os.path.join('data', 'workup_ws2_control.py'))
    print_then_call(python, os.path.join('data', 'workup_ws2_na_assisted.py'))
    print_then_call(python, os.path.join('data', 'define_clusters.py'))


def build_figures():
    print_with_line('figures')
    print_then_call(python, os.path.join('figures', 'figures.py'))


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('no arguments given---building everything!')
        sys.argv.append('all')
    if 'fetch' in sys.argv or 'all' in sys.argv:
        fetch_data()
    if 'data' in sys.argv or 'all' in sys.argv:
        build_data()
    if 'figures' in sys.argv or 'all' in sys.argv:
        build_figures()
    print_with_line('building done!')
