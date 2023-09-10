from setuptools import setup, Extension

module1 = Extension(
    'cparse',
    sources=['cparse.cpp'],
    libraries=['libpcap'],
    library_dirs=['/usr/local/lib'],
    include_dirs=['/usr/local/include'],
)

setup(
    ext_modules=[Extension('cparse', sources=['cparse.cpp'], libraries=['pcap']),],
)