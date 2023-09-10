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

# MRT RIB log import  [to convert to a text IP-ASN lookup table]
# file to use per day should be of the RouteViews or RIPE RIS series, e.g.:
# http://archive.routeviews.org/bgpdata/2009.11/RIBS/rib.20091125.0600.bz2

from __future__ import print_function, division
from pyasn import mrtx, __version__
from time import time
from sys import argv, exit, stdout
from glob import glob
from datetime import datetime, timedelta
from subprocess import call



def convert(infile, outfile):
    """converts gzipped input to .DAT"""
    prefixes = mrtx.parse_mrt_file(infile,
                                    print_progress=True,
                                    skip_record_on_error=False)
    mrtx.dump_prefixes_to_file(prefixes, outfile, infile)
    
    v6 = sum(1 for x in prefixes if ':' in x)
    v4 = len(prefixes) - v6
    print('IPASN database saved (%d IPV4 + %d IPV6 prefixes)' % (v4, v6))
