import json
import re
import urllib.request
from datetime import datetime

from packaging.specifiers import SpecifierSet
from packaging.version import Version


def get_data(url):
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode())


def get_supported_python_versions(package_name, version):
    url = f"https://pypi.org/pypi/{package_name}/{version}/json"
    data = get_data(url)

    prefix = "Programming Language :: Python :: "
    pattern = prefix + "[0-9]+\\.[0-9]+"
    python_versions = []

    classifiers = data.get("info", {}).get("classifiers", [])
    for c in classifiers:
        if re.match(pattern, c):
            version_str = c[len(prefix) :]
            python_versions.append(version_str)

    return python_versions


def get_oldest_pypy_package_version(package_name, target_python_version):
    # Download versions list
    url = f"https://pypi.org/pypi/{package_name}/json"
    data = get_data(url)

    # Get valid versions
    valid_versions = []
    target_ver = Version(target_python_version)
    releases = data.get("releases", {})
    for ver_str, files in releases.items():
        if not files:
            continue
        try:
            ver = Version(ver_str)
        except Exception:
            continue

        req_python = files[0].get("requires_python")
        if datetime.fromisoformat(files[0].get("upload_time")) < datetime(2017, 1, 1):
            continue
        if req_python:
            try:
                spec = SpecifierSet(req_python)
                if target_ver not in spec:
                    continue
            except Exception:
                pass

        valid_versions.append(ver)

    # Binary search
    assert valid_versions
    valid_versions.sort(reverse=True)
    add = 1 << (len(valid_versions).bit_length() - 1)
    ind = 0
    while add:
        if ind + add < len(valid_versions):
            version = valid_versions[ind + add]
            py_versions = get_supported_python_versions(package_name, str(version))
            ok = False
            for v in py_versions:
                if Version(v) >= target_ver:
                    ok = True
                    break
            if ok:
                ind += add
        add >>= 1

    return str(valid_versions[ind])


def get_adjacent_python_versions(target_python_version):
    url = "https://endoflife.date/api/python.json"
    data = get_data(url)

    target_ver = Version(target_python_version)
    prev, next = None, None
    for v in data:
        v = Version(v["cycle"])
        if v < target_ver:
            if prev is None or prev < v:
                prev = v
        elif v > target_ver:
            if next is None or next > v:
                next = v
    return str(prev), str(next)
