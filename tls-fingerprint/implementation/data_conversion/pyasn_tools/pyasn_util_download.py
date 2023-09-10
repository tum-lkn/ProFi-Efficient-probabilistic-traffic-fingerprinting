#!/usr/bin/python

# Copyright (c) 2009-2017 Hadi Asghari
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Script to download the latest routeview bgpdata, or for a certain period
# Thanks to Vitaly Khamin (https://github.com/khamin) for the FTP code

from __future__ import print_function, division
from datetime import date, datetime
from time import time
from ftplib import FTP
from subprocess import call
from sys import argv, exit, stdout, version_info
try:
    from pyasn import __version__
except:
    pass  # not fatal if we can't get version
if version_info[0] < 3:
    from urllib2 import urlopen
else:
    from urllib.request import urlopen




def ftp_download(server, remote_dir, remote_file, local_file, print_progress=True):
    """Downloads a file from an FTP server and stores it locally"""
    ftp = FTP(server)
    ftp.login()
    ftp.cwd(remote_dir)
    if print_progress:
        print('Downloading ftp://%s/%s/%s' % (server, remote_dir, remote_file))
    filesize = ftp.size(remote_file)
    # perhaps warn before overwriting file?
    with open(local_file, 'wb') as fp:
        def recv(s):
            fp.write(s)
            recv.chunk += 1
            recv.bytes += len(s)
            if recv.chunk % 100 == 0 and print_progress:
                print('\r %.f%%, %.fKB/s' % (recv.bytes*100 / filesize,
                      recv.bytes / (1000*(time()-recv.start))), end='')
                stdout.flush()
        recv.chunk, recv.bytes, recv.start = 0, 0, time()
        ftp.retrbinary('RETR %s' % remote_file, recv)
    ftp.close()
    if print_progress:
        print('\nDownload complete.')


def find_latest_in_ftp(server, archive_root, sub_dir, print_progress=True):
    """Returns (server, filepath, filename) for the most recent file in an FTP archive"""
    if print_progress:
        print('Connecting to ftp://' + server)
    ftp = FTP(server)
    ftp.login()
    months = sorted(ftp.nlst(archive_root), reverse=True)  # e.g. 'route-views6/bgpdata/2016.12'
    filepath = '/%s/%s' % (months[0], sub_dir)
    if print_progress:
        print("Finding most recent archive in %s ..." % filepath)
    ftp.cwd(filepath)
    fls = ftp.nlst()
    if not fls:
        filepath = '/%s/%s' % (months[1], sub_dir)
        if print_progress:
            print("Finding most recent archive in %s ..." % filepath)
        ftp.cwd(filepath)
        fls = ftp.nlst()
        if not fls:
            raise LookupError("Cannot find file to download. Please report a bug on github?")
    filename = max(fls)
    ftp.close()
    return (server, filepath, filename)


def find_latest_routeviews(archive_ipv):
    # RouteViews archives are as follows:
    # ftp://archive.routeviews.org/datapath/YYYYMM/ribs/XXXX
    archive_ipv = str(archive_ipv)
    assert archive_ipv in ('4', '6', '46', '64')
    return find_latest_in_ftp(server='archive.routeviews.org',
                              archive_root='bgpdata' if archive_ipv == '4' else
                                           'route-views6/bgpdata' if archive_ipv == '6' else
                                           'route-views4/bgpdata',  # 4+6
                              sub_dir='RIBS')

