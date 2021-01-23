from pathlib import Path
from functools import partial
import setuptools


def find_meta(meta, meta_file):
    """Extract __*meta*__ from `meta_file`."""
    import re
    meta_match = re.search(r"^__{meta}__\s+=\s+['\"]([^'\"]*)['\"]".format(meta=meta), meta_file, re.M)
    if meta_match:
        return meta_match.group(1)
    raise RuntimeError("Unable to find __{meta}__ string.".format(meta=meta))


def read(fp):
    """Return the contents of the resulting file. Assume UTF-8 encoding."""
    import codecs
    with codecs.open(fp, "rb", "utf-8") as f:
        return f.read()


CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "Intended Audience :: End Users/Desktop",
    "Topic :: Scientific/Engineering",
    "Environment :: Console",
    "Operating System :: MacOS",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: POSIX :: Linux",
    "Operating System :: Unix",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
]
PACKAGES = setuptools.find_packages(where="src")
find_meta = partial(find_meta, meta_file=read((Path("src") / str(next(iter(PACKAGES))) / "__init__.py").resolve()))
INSTALL_REQUIRES = read("requirements.txt").strip().split("\n")
NAME = find_meta("title")
KEYWORDS = [NAME]

if __name__ == "__main__":
    setuptools.setup(
        name=NAME,
        description=find_meta("description"),
        license=find_meta("license"),
        url=find_meta("url"),
        version=find_meta("version"),
        author=find_meta("author"),
        maintainer=find_meta("author"),
        long_description=read("README.md"),
        long_description_content_type="text/markdown",
        packages=PACKAGES,
        package_dir={"": "src"},
        classifiers=CLASSIFIERS,
        keywords=KEYWORDS,
        install_requires=INSTALL_REQUIRES,
        python_requires='>=3.6',
    )
