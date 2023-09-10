from .pyasn_util_download import find_latest_routeviews
from .pyasn_util_download import ftp_download
from .pyasn_util_convert import convert

# find and download latest archive

def update_asn_cache():
    srvr, rp, fn = find_latest_routeviews(4)
    ftp_download(srvr, rp, fn, 'latestv4.bz2')
    # extracting dat file
    convert('latestv4.bz2', 'latestv4.dat')
