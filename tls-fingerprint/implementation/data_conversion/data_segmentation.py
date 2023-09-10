"""
Helper functions that extract data or features from traces in
.PCAPNG or .CSV formats, and functions converting .PCAPNG to .CSVs
and those if needed to Pandas dataframes
"""

# External imports
import pandas as pd
import numpy as np
import time
import socket
import pyasn
from typing import List, Tuple

# Internal imports
from pyasn_tools.update import update_asn_cache
from pyasn_tools.pyasn_util_convert import convert


def update_asn_cache_local(update=True, unzip=True) -> None:
    """
    Wrapper function for updating the asn cache.
    Downloads, unzips, and saves the asn dat file.
    """
    if update: update_asn_cache()
    if update or unzip:
        convert('./latestv4.bz2', "./pyasn_tools/latestv4.dat")


def extract_asn_seq(trace:pd.DataFrame, filters:List[str]=["unique_ips", "unique_asns"], cache_params:List=[True, True]) -> List:
    """
    Takes a trace in dataframe format and extracts a sequence of contacted ASNs.
    Sequence of ASNs is based on the order in which they are contacted.
    Optionally filter output such that only unique ASNs are added as contacted,
    or filter such that ASNs are only added if IP Address is unique
    """
    # Update or unzip pyasn cache if desired
    update_asn_cache_local(update=cache_params[0], unzip=cache_params[1])

    # IP Destination and _ws_col info Columns, only rows with 'Client Hello'
    asn_df = (trace[trace['_ws.col.Info']=="Client Hello"])[["ip.dst", "_ws.col.Info"]]
    if "unique_ips" in filters:
        asn_df = asn_df.drop_duplicates(subset=['ip.dst'])

    # Get our asn lookup table
    pyasn_db = pyasn.pyasn('./pyasn_tools/latestv4.dat')

    # Find ASN to each IP Address
    ips = asn_df['ip.dst'].tolist()
    asns = []
    for ip in ips:
        asn = pyasn_db.lookup(ip)[0]
        if "unique_asns" in filters:
            if asn not in asns: asns.append(asn)
        else: asns.append(asn)
    print(asns)
