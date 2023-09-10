"""
This module implements functionality that extracts TLS flow from a packet
capture file.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
import subprocess as subp
import re
import os
from io import StringIO
from typing import List, Dict, Any, Tuple, Union, ClassVar
IX = pd.IndexSlice


# TLS record types and their resolution to human readable form.
TLS_MAP = {
    20: 'CHANGE_CIPHER_SPEC',
    21: 'ALERT',
    22: {
        1: 'CLIENT_HELLO',
        2: 'SERVER_HELLO',
        11: 'CERTIFICATE',
        12: 'SERVER_KEY_EXCHANGE',
        13: 'CERTIFICATE_REQUEST',
        14: 'SERVER_DONE',
        15: 'CERTIFICATE_VERIFY',
        16: 'CLIENT_KEY_EXCHANGE',
        20: 'FINISHED'
    },
    23: 'APPLICATION_DATA'
}


class TraceFailure(Exception):
    def __init__(self, url, reason, msg):
        super(TraceFailure, self).__init__(msg)
        self.message = msg
        self.url = url
        self.reason = reason

    def __str__(self):
        return f"Trace for URL {self.url} failed for reason {self.reason} with message {self.message}"


class MainFlow(object):
    """
    Represents a main flow of a trace.
    """

    @classmethod
    def from_dict(self, d: Dict[str, Any]) -> 'MainFlow':
        flow = MainFlow(
            src_ip=d['src_ip'],
            src_port=d['src_port'],
            dst_ip=d['dst_ip'],
            dst_port=d['dst_port'],
            tcp_header_length=d['tcp_header_length'],
            ip_header_length=d['ip_header_length'],
            total_num_frames=d.get('total_num_frames', len(d['tls_records']))
        )
        # Default total_num_frames to number of frames for backwards compatibility.
        # If this attribute is not present, then all packets from the main flow
        # have been added to this object.
        flow.tls_records = [TlsRecord.from_dict(x) for x in d['tls_records']]
        flow.frames = [Frame.from_dict(x) for x in d['frames']]
        return flow

    def __init__(self, src_ip: str, src_port: int, dst_ip: str, dst_port: int,
                 tcp_header_length: int, ip_header_length: int, total_num_frames: int):
        """
        Initializes object.

        Args:
            src_ip: Source IP, i.e., IP of client initiating the TLS session.
            src_port: High ranging source port.
            dst_ip: Destination IP of server TLS session is initiated to.
            dst_port: Destination port, usually 443.
            tcp_header_length (depreciated): The length of the TCP headers. TCP header length
                can be variable and is necessary for offset calculations. This
                is the header length of the first packet in the flow.
            ip_header_length (depreciated): The length of the IP headers. IP header lengths
                can be variable and is necessary for offset calculations. Length
                from first packet of flow.
            total_num_frames: The total number of frames extracted from the PCAP
                file. The Flow can have less frames later on in case a missing
                TCP segment was discovered during parsing. In this case, the
                parser returns the flow without continuing to dissect packets.

        Note:
            tcp_header_length and ip_header_length are depreciated in favour of
            filds in the Frame class. The fields will be removed in future versions.
        """
        self.src_ip = src_ip
        self.src_port = int(src_port)
        self.dst_ip = dst_ip
        self.dst_port = int(dst_port)
        self.tls_records: List[TlsRecord] = []
        self.frames: List[Frame] = []
        self.tcp_header_length = int(tcp_header_length)
        self.ip_header_length = int(ip_header_length)
        self.total_num_frames = total_num_frames

    def __str__(self) -> str:
        s = f'Flow: {self.src_ip}\t{self.src_port}\t{self.dst_ip}\t{self.dst_port}'
        s += '\n\t' + '\n\t'.join([str(f) for f in self.frames]) if len(self.frames) > 0 else ''
        return s

    def to_dict(self) -> Dict[str, Any]:
        return {
            'src_ip': self.src_ip,
            'src_port': int(self.src_port),
            'dst_ip': self.dst_ip,
            'dst_port': int(self.dst_port),
            'tcp_header_length': int(self.tcp_header_length),
            'ip_header_length': int(self.ip_header_length),
            'frames': [f.to_dict() for f in self.frames],
            'tls_records': [r.to_dict() for r in self.tls_records],
            'total_num_frames': int(self.total_num_frames)
        }


class TlsRecord(object):
    """
    Represents a TLS record.
    """
    header_size = 5

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'TlsRecord':
        if 'time_epoch' not in d:
            return TlsRecord(time_epoch=None, **d)
        else:
            return TlsRecord(**d)

    @classmethod
    def from_record(cls, record: 'TlsRecord') -> 'TlsRecord':
        return cls(
            length=record.length,
            content_type=record.content_type,
            minor_version=record.minor_version,
            major_version=record.major_version,
            record_number=record.record_number,
            header_start_byte=record.header_start_byte,
            handshake_type=record.handshake_type,
            direction=record.direction,
            time_epoch=record.time_epoch
        )

    def __init__(self, length: int, content_type: int, minor_version: int,
                 major_version: int, record_number: int, direction: int,
                 header_start_byte: int, time_epoch: float, handshake_type: int=None):
        """
        Initializes object.

        Args:
            length: The length of the record in bytes (decimal).
            content_type: The TLS record (content) type (first level of `TLS_MAP`).
            minor_version: The minor TLS version.
            major_version: The major TLS version.
            record_number: The record number, increases by one for each new record.
            direction: -1 for outbound (client -> server), 1 for inbound (server -> client).
            header_start_byte: The byte at which the header of this record started
                in the corresponding frame.
            time_epoch: Timestamp of frame the record is contained in.
            handshake_type: The handshake type.
        """
        self.length = length
        self.content_type = content_type
        self.minor_version = minor_version
        self.major_version = major_version
        self.record_number = record_number
        self.handshake_type = handshake_type
        self.header_start_byte = header_start_byte
        self.time_epoch = time_epoch
        self.direction = direction

    def __str__(self) -> str:
        d = 'outbound' if self.direction < 0 else 'inbound'
        msg_type = TLS_MAP[self.content_type]
        if self.content_type == 22:
            try:
                msg_type = TLS_MAP[self.content_type][self.handshake_type]
            except Exception as e:
                msg_type = f"ENCRYPTED_HANDSHAKE_MESSAGE {self.handshake_type}"
        s = f'Record: {self.record_number}\t{self.time_epoch}\t{msg_type:30s}\t' + \
            f'{self.major_version}\t{self.minor_version}\t{d:8s}\t{self.header_start_byte}->{self.length}'
        return s

    def to_dict(self) -> Dict[str, Any]:
        return {
            'length': int(self.length),
            'content_type': int(self.content_type),
            'minor_version': int(self.minor_version),
            'major_version': int(self.major_version),
            'record_number': int(self.record_number),
            'handshake_type': None if self.handshake_type is None else int(self.handshake_type),
            'header_start_byte': int(self.header_start_byte),
            'direction': int(self.direction),
            'time_epoch': float(self.time_epoch)
        }


class Frame(object):
    """
    Represents a normal L2 frame.
    """

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Frame':
        frame = Frame(
            tcp_length=d['tcp_length'],
            frame_number=d['frame_number'],
            time_epoch=d['time_epoch'],
            direction=d['direction'],
            tcp_header_offset=d['tcp_header_offset'],
            ip_header_length=d.get('ip_header_length', -1),
            tcp_header_length=d.get('tcp_header_length', -1)
        )
        for x in d['tls_records']:
            if 'time_epoch' not in x:
                x['time_epoch'] = frame.time_epoch
            frame.tls_records.append(TlsRecord.from_dict(x))
        return frame

    def __init__(self, tcp_length: int, frame_number: int, time_epoch: float, direction: int,
                 ip_header_length: int, tcp_header_length: int, tcp_header_offset: int=None):
        """
        Initializes object.

        Args:
            tcp_length: Length of the TCP payload.
            frame_number: The number of the frame in the packet capture.
            time_epoch: The timestamp of the packet.
            direction: -1 for outbound (client -> server), 1 for inbound (server -> client).
            tcp_header_offset (depreciated): The byte at which the TCP paylod starts, i.e.,
                sum of the sizes of the L2, L3 and L4 headers. Will be removed
                in future versions.
            tcp_header_length: The length of the TCP headers. TCP header length
                can be variable and is necessary for offset calculations.
            ip_header_length: The length of the IP headers. IP header lengths
                can be variable and is necessary for offset calculations.
        """
        self.tcp_length = tcp_length
        self.frame_number = frame_number
        self.time_epoch = time_epoch
        self.tls_records: List[TlsRecord] = []
        self.direction = direction
        if tcp_header_offset is None:
            assert ip_header_length > 0
            assert tcp_header_length > 0
            self.tcp_header_offset = 14 + ip_header_length + tcp_header_length
        else:
            if ip_header_length == -1 or tcp_header_length == -1:
                self.tcp_header_offset = tcp_header_offset
            else:
                self.tcp_header_offset = 14 + ip_header_length + tcp_header_length
        self.ip_header_length = ip_header_length
        self.tcp_header_length = tcp_header_length

    def __str__(self) -> str:
        d = 'outbound' if self.direction < 0 else 'inbound'
        s = f"Frame: {self.frame_number}\t{self.time_epoch}\t{d:8s}\t{self.tcp_length}"
        s += '\n\t\t' + '\n\t\t'.join([str(r) for r in self.tls_records]) if len(self.tls_records) > 0 else ''
        return s

    def to_dict(self) -> Dict[str, Any]:
        return {
            'tcp_length': int(self.tcp_length),
            'frame_number': int(self.frame_number),
            'time_epoch': float(self.time_epoch),
            'tls_records': [r.to_dict() for r in self.tls_records],
            'direction': int(self.direction),
            'tcp_header_offset': int(self.tcp_header_offset),
            'ip_header_length': int(self.ip_header_length),
            'tcp_header_length': int(self.tcp_header_length)
        }


class MainFlowIter(object):
    """
    Traces can contain duplicates or spurious re-transmissions. This iterator
    checks if a segment with a sequence length has already been send and if so,
    ignores this packet.
    """
    def __init__(self, main_flow: pd.DataFrame):
        self.time = 0
        self.idx_server_client = 0
        self.idx_client_server = 0
        self.main_flow = main_flow
        self.server_ip = main_flow.at[0, 'ip.dst']
        self.client_ip = main_flow.at[0, 'ip.src']
        self.server_port = main_flow.at[0, 'tcp.dstport']
        self.client_port = main_flow.at[0, 'tcp.srcport']
        self.seqs_client_server = {}
        self.seqs_server_client = {}
        # Extract packets in each direction.
        self.client_server = main_flow.set_index(['ip.src', 'tcp.srcport'])\
                                 .sort_index()\
                                 .loc[IX[self.client_ip, self.client_port], :]\
                                 .sort_values(by='tcp.seq')\
                                 .reset_index()
        self.server_client = main_flow.set_index(['ip.src', 'tcp.srcport']) \
                                 .sort_index() \
                                 .loc[IX[self.server_ip, self.server_port], :] \
                                 .sort_values(by='tcp.seq') \
                                 .reset_index()

    def __iter__(self):
        return self

    def __next__(self) -> pd.Series:
        duplicate_encountered = True
        row = None
        while duplicate_encountered:
            # Stop if both indices exceed the amount of frames in each direction.
            if self.idx_client_server >= self.client_server.shape[0] and self.idx_server_client >= self.server_client.shape[0]:
                raise StopIteration
            else:
                # Retrieve the next frame based on time. Due to re-transmissions,
                # timings can be a bit off for multi-packet messages that intersect
                # in an area of bi-directional communication. This is rare, though.
                t_sc = self.server_client.iloc[self.idx_server_client].loc['frame.number'] \
                    if self.idx_server_client < self.server_client.shape[0] else 1e9
                t_cs = self.client_server.iloc[self.idx_client_server].loc['frame.number'] \
                    if self.idx_client_server < self.client_server.shape[0] else 1e9
                if t_sc < t_cs:
                    dir_df = self.server_client
                    idx = self.idx_server_client
                    seqs = self.seqs_server_client
                    self.idx_server_client += 1
                else:
                    dir_df = self.client_server
                    idx = self.idx_client_server
                    seqs = self.seqs_client_server
                    self.idx_client_server += 1
                seq_num = int(dir_df.iloc[idx].loc['tcp.seq'])
                frame_num = dir_df.iloc[idx].loc['frame.number']
                if seq_num in seqs:
                    duplicate_encountered = True
                    # print(f"Duplicate segment {seq_num} on frame {frame_num}.")
                else:
                    duplicate_encountered = False
                    row = dir_df.iloc[idx, :]
                    seqs[seq_num] = True
        return row


class RandomRecordSizeDefense(object):
    """
    Functor that takes as input a mainflow and returns a mainflow with warped
    TLS records and packets.
    The Defense randomly generates TLS Record sizes and changes the orginal ones
    accordingly. This can result in a difference in the packet sequence as well.
    The Functor returns a new mainflow object with the number of elements larger
    or equal to max_seq_length.
    """
    def __init__(self, max_seq_length: int, get_random_length: callable):
        """
        Initializes object.
        Args:
            max_seq_length: Maximum length of obfuscated main flow.
            get_random_length: Callable that returns a random TLS Record length
                given the original record.
        """
        self.max_seq_length = max_seq_length
        self.get_random_length = get_random_length
        self.mtu = 1500
        self.tls_record_header_length = 5
        self.random = np.random.RandomState(seed=max_seq_length)

    def main_flow_iter(self, main_flow: MainFlow | Dict[str, Any]) -> Tuple[Frame, TlsRecord]:
        if type(main_flow) == dict:
            last_record_id = -1
            for frame_d in main_flow['frames']:
                frame = Frame.from_dict(frame_d)
                for record in frame.tls_records:
                    if record.record_number == last_record_id:
                        continue
                    else:
                        last_record_id = record.record_number
                        yield frame, record
        else:
            for frame in main_flow.frames:
                for record in frame.tls_records:
                    yield frame, record

    def _make_new_flow(self, main_flow: MainFlow | Dict[str, Any]) -> MainFlow:
        if type(main_flow) == MainFlow:
            new_flow = MainFlow(
                src_ip=main_flow.src_ip,
                src_port=main_flow.src_port,
                dst_ip=main_flow.dst_ip,
                dst_port=main_flow.dst_port,
                tcp_header_length=main_flow.tcp_header_length,
                ip_header_length=main_flow.ip_header_length,
                total_num_frames=main_flow.total_num_frames
            )
        else:
            new_flow = MainFlow(
                src_ip=main_flow.get("src_ip", None),
                src_port=main_flow.get("src_port", -1),
                dst_ip=main_flow.get("dst_ip", None),
                dst_port=main_flow.get("dst_port", -1),
                tcp_header_length=main_flow.get("tcp_header_length", -1),
                ip_header_length=main_flow.get("ip_header_length", -1),
                total_num_frames=main_flow.get("total_num_frames", -1)
            )
        return new_flow

    def _make_new_frame(self, frame: Frame, main_flow: MainFlow) -> Frame:
        new_frame = Frame(
            tcp_length=frame.tcp_length,
            frame_number=len(main_flow.frames) + 1,
            time_epoch=-1,
            direction=frame.direction,
            ip_header_length=frame.ip_header_length,
            tcp_header_length=frame.tcp_header_length,
            tcp_header_offset=frame.tcp_header_offset
        )
        return new_frame

    def __call__(self, main_flow: MainFlow | Dict[str, Any]) -> MainFlow:
        new_flow = self._make_new_flow(main_flow)
        prev_frame_number = None
        remaining_packet_size = None
        for frame, tls_record in self.main_flow_iter(main_flow):
            if len(new_flow.tls_records) > self.max_seq_length and \
                    len(new_flow.frames) > self.max_seq_length:
                break
            if tls_record.content_type != 23:
                # In case of handshaketype duplicate the packets as is.
                # This is important since handshake packets cannot be padded
                # with the normal TLS protocol.
                if prev_frame_number != frame.frame_number:
                    new_frame = self._make_new_frame(frame, new_flow)
                    prev_frame_number = frame.frame_number
                    new_flow.frames.append(new_frame)
                    remaining_packet_size = self.mtu - frame.ip_header_length - frame.tcp_header_length
                new_record = TlsRecord.from_record(tls_record)
                new_frame.tls_records.append(new_record)
                new_flow.tls_records.append(new_record)
                remaining_packet_size -= tls_record.length + tls_record.header_size
                continue
            # Iterate over Frames and TLS Records in those frames.
            # The original data length. This data will be either contained in
            # one TLS record or split across multiple records.
            remaining_record_size = tls_record.length
            if frame.frame_number != prev_frame_number:
                # Check if we get a new frame. If so, trigger the creation of
                # a new frame in the mutated flow by setting remaining_packet_size
                # to zero.
                prev_frame_number = frame.frame_number
                remaining_packet_size = 0
            while remaining_record_size > 0:
                # Get a random TlsRecord size and subtract it from the original
                # size. Negative values are fine, indicates padding. Positive
                # values indicate a split of the original payload into multiple
                # TLS Records.
                size = self.get_random_length(tls_record)
                remaining_record_size -= size
                new_record = TlsRecord.from_record(tls_record)
                new_record.length = size  # Set the correct size.
                if remaining_packet_size == 0:
                    new_record.header_start_byte = 0
                else:
                    new_record.header_start_byte = (self.mtu - frame.ip_header_length -
                                                    frame.tcp_header_length) - remaining_packet_size
                new_record.record_number = len(new_flow.tls_records) + 1  # Note needed, still to be ok.
                new_flow.tls_records.append(new_record)
                # Used in arithmetics to account for the size of the TLS header.
                tls_header_len = self.tls_record_header_length
                while size > 0:
                    if remaining_packet_size < self.tls_record_header_length or self.random.uniform() < 0.5:
                        # If the remaining packet size is smaller than the length
                        # of the TLS header create a new frame. In practice, the
                        # header can span multiple packets, I am not supporting this,
                        # though.
                        new_frame = self._make_new_frame(frame, new_flow)
                        new_frame.tcp_length = 0
                        new_frame.tls_records.append(new_record)
                        new_flow.frames.append(new_frame)
                        remaining_packet_size = self.mtu - frame.ip_header_length - frame.tcp_header_length
                    # Get the minimum of size and the packet size. This value is
                    # subtracted from the size variable to account for the data
                    # that is transmitted. The value is added to the frame's
                    # tcp_length attribute to account for the payload in the packet.
                    # Note that one packet can have multiple TLS records, thats
                    # why we increment. The tls header is set to zero after that,
                    # i.e., considered only once for one record.
                    tcp_len = int(np.min([size, remaining_packet_size - tls_header_len]))
                    size -= tcp_len
                    new_frame.tcp_length += tcp_len + tls_header_len
                    remaining_packet_size -= tcp_len + tls_header_len
                    tls_header_len = 0
        return new_flow


def get_ips_from_dns(pth_to_trace: str, hostname: str, url: str) -> List[str]:
    """
    Extract the IP addresses from DNS traffic in a packet capture file.

    Filter all DNS responses from the packet capture that contain `hostname`
    in the queries. Then retrieve all addresses from the first record.

    Args:
        pth_to_trace: Path to the packet capture file.
        hostname: Name that is resolved in the DNS query.

    Returns:
        ips: List of IP addresses.
    """
    condition = f'dns.resp.name == \\"{hostname}\\"'
    if hostname.startswith("www"):
        condition = f'dns.resp.name matches \\"[wW]{{3}}.{hostname[4:]}\\"'
    dns_cmd = f'tshark -r {pth_to_trace} -Y "{condition}" -T fields ' \
        '-e frame.number ' \
        '-e frame.time_epoch ' \
        '-e dns.a  ' \
        '-e dns.resp.name ' \
        '-E header=y -E separator=";"'
    out = subp.check_output(dns_cmd, shell=True)
    dns_replies = pd.read_table(StringIO(out.decode('utf8')), sep=';')
    if dns_replies.shape[0] > 0:
        ips = dns_replies.iloc[0].loc['dns.a'].split(',')
    else:
        msg = f"No DNS entries in trace {pth_to_trace} for hostname {hostname}."
        raise TraceFailure(url, "No DNS Response", msg)
        # raise ValueError()
    return ips


def get_client_hello(pth_to_trace: str, ip: str) -> pd.DataFrame:
    """
    Retrieve the client hellos that were initiated to a specific end host
    identified by an IP address.

    Args:
        pth_to_trace: Path to the packet capture file.
        ip: IP address of the receiver of the client hello (destination IP
            address).

    Returns:
        client_hello: DataFrame with information to the client hello.
    """
    client_hello_cmd = f'tshark -r {pth_to_trace} ' \
        f'-Y "tls.handshake.type == 1 && ip.dst=={ip}" ' \
        '-T fields ' \
        '-e frame.number ' \
        '-e frame.time_epoch ' \
        '-e ip.src ' \
        '-e tcp.srcport ' \
        '-e ip.dst ' \
        '-e tcp.dstport ' \
        '-E header=y -E separator=";"'
    out = subp.check_output(client_hello_cmd, shell=True)
    client_hello = pd.read_table(StringIO(out.decode('utf8')), sep=';')
    return client_hello


def four_tuple_from_client_hello(client_hello: pd.DataFrame) -> Tuple[str, int, str, int]:
    """
    Extract source and destination IP and source and destination ports from the
    output of the `get_client_hello` function. Returns the four tuple of the
    first client hello in the data frame.

    Args:
        client_hello: DataFrame returned by `get_client_hello`.

    Returns:
        src_ip: IP address of the node initiating the TLS session.
        src_port: Source port, usually in the high ranges.
        dst_ip: IP address of the server to which a TLS session is initiated.
        dst_port: Destination port, usually 443.
    """
    src_ip = client_hello.loc[0, 'ip.src']
    src_port = int(client_hello.loc[0, 'tcp.srcport'])
    dst_ip = client_hello.loc[0, 'ip.dst']
    dst_port = int(client_hello.loc[0, 'tcp.dstport'])
    return src_ip, src_port, dst_ip, dst_port


def get_main_flow(pth_to_trace: str, client_hello: pd.DataFrame, url: str) -> pd.DataFrame:
    """
    Extract all packets from a packet capture that belong to a specific TLS
    session.

    Only TLS packets are extracted. No control messages are extracted. Starts
    with the ClientHello.

    Args:
        pth_to_trace: Path to the packet capture file.
        client_hello: Output of `get_client_hello` function.

    Returns:
        main_flow: DataFrame with data to packets belonging to a TLS
            session identified by the first CLIENT_HELLO in `client_hello`.
    """
    src_ip, src_port, dst_ip, dst_port = four_tuple_from_client_hello(client_hello)
    main_flow_cmd = f'tshark -r {pth_to_trace} ' \
        f'-Y "((ip.src == {src_ip}   && '\
        f'tcp.srcport == {src_port}  && ' \
        f'ip.dst == {dst_ip} && ' \
        f'tcp.dstport == {dst_port}) || ' \
        f'(ip.src == {dst_ip}        && ' \
        f'tcp.srcport == {dst_port} && ' \
        f'ip.dst == {src_ip} && ' \
        f'tcp.dstport == {src_port})) && ' \
        f'tcp.len > 0" -o tcp.desegment_tcp_streams:false -T fields ' \
        '-e frame.number ' \
        '-e frame.time_epoch ' \
        '-e ip.src ' \
        '-e tcp.srcport ' \
        '-e tcp.seq ' \
        '-e ip.dst ' \
        '-e tcp.dstport ' \
        '-e tcp.len ' \
        '-e tcp.hdr_len ' \
        '-e tls.record.content_type ' \
        '-e tls.record.length ' \
        '-e tls.record.version ' \
        '-e tls.handshake.type ' \
        '-e ip.hdr_len ' \
        '-e ipv6.nxt ' \
        '-e ipv6.dst ' \
        '-e _ws.col.Info ' \
        '-E header=y -E separator=";"'
    out = subp.check_output(main_flow_cmd, shell=True)
    main_flow = pd.read_table(StringIO(out.decode('utf8')), sep=';')
    if main_flow.shape[0] < 4:
        msg = "MainFlow is empty and does not contain packets for the flow" \
                         f" {src_ip}, {src_port}, {dst_ip}, {dst_port}"
        raise TraceFailure(url, "Empty Main Flow", msg)
        # raise ValueError(msg)
    return main_flow


def extract_hexdump(raw_dump: str) -> List[str]:
    hexdump = [s[6:53].strip() for s in raw_dump.split(os.linesep)]
    return " ".join(hexdump).split(" ")


def extract_record_info(pth_to_trace: str, frame_number: int,
                        tmp_file_path: str, byte_offset=0):
    skip_to = 0
    out = ''
    if not out.startswith("0000"):
        skip_to = out.find(os.linesep) + 1  # Skip the line: Frame (\d+) Bytes: if present.
    num_rows = int(byte_offset / 16)
    skip_to += num_rows * 73  # 73 is length of one row in binary output.


def extract_record_info_re(pth_to_trace: str, frame_number: int,
                           tmp_file_path: str, prev_content: Dict[str, Union[str, int]],
                           url: str, byte_offset=0) -> Dict[str, Union[str, int]]:
    """
    Search for a TLS header in the hexdump of a packet and retrieve the TLS
    header info.

    Args:
        pth_to_trace: Path to a packet capture file.
        frame_number: Number of the frame in which TLS headers should be searched
            for.
        tmp_file_path: File to a temporary file storing the binary output produced
            by the call of the tshark command. The output is not needed and can
            be overwritten by subsequent calls.
        prev_content: Content of a previous group dict returned by this function.
            Used to handle TLS records that span multiple packets.
        byte_offset: How many bytes to skip in the payload before starting to
            look for a header.

    Info:
        The returned dictionary has the following entries:
        - numBytes0: How many bytes precede the row of the tshark hexdump in
                     which the header has been found.
        - anyBytes0: Sequence of bytes in the row of the hexdump that precede
                     the matched header.
        - recType: TLS record type.
        - ascii1: The ASCI output of the tshark hexdump, necessary in case a
                  line break occurs between the TLS record type and the version
                  number.
        - numBytes2: Byte count in case a line break was present.
        - major: The major version of the TLS record.
        - ascii2: Matches ASCI output in case a line break occurs.
        - numBytes3: Matches the byte offset in case a line break occurs.
        - minor: The minor version of the TLS record.
        - ascii3: Matches ASCI output in case a line break occurs.
        - numBytes4: Matches the byte offset in case a line break occurs.
        - length0: The first byte of the TLS record length field.
        - ascii4: Matches ASCI output in case a line break occurs.
        - numBytes5: Matches the byte offset in case a line break occurs.
        - length1: The second byte of the TLS record length field.
        - ascii5: Matches ASCI output in case a line break occurs.
        - numBytes6: Matches the byte offset in case a line break occurs.
        - handshakeType: The first byte of the record content, identifies the
                         handshake message type.
                         
    Example:
        0000  02 42 55 b3 54 37 02 42 ac 11 00 02 08 00 45 00   .BU.T7.B......E.
        0010  00 d9 da 1f 40 00 40 06 43 20 ac 11 00 02 12 cd   ....@.@.C ......
        0020  5d ff be 7a 01 bb ae 3d 18 ce d8 55 d2 c0 50 18   ]..z...=...U..P.
        0030  01 f5 a8 98 00 00 4d 5a 03 00 ac 00 00 17 03 03   .....$.0...8.wL.
        0040  00 ac 01 8a 01 b9 f4 be 8d ef 11 07 60 d7 2a 0a   ............`.*.
        0050  c2 af ee 0d af 24 b7 30 af f5 86 38 ac 77 4c 1d   .....$.0...8.wL.
        0060  dc cd ba b6 bc 71 dc 59 a9 66 7a 3c a3 87 63 4c   .....q.Y.fz<..cL
        0070  20 2f f3 f3 65 9f b7 85 05 63 4e 7f 20 56 15 03    /..e....cN. V..
        0080  03 ff 2f 74 c0 7b ed 3a bf 0c 7a a0 b8 21 03 16   0(.t.{.:..z..!..
        0090  03 03 7b e6 7e 6f 63 8b e5 ee d1 ba 4a 14 03 03   ..{.~oc.....J...
        00a0  1f 3a 74 65 ed c7 b9 ad 67 50 7f 58 17 03 03 99   .:te....gP.X....
        00b0  94 5c b5 aa 40 a6 03 c4 d2 42 81 1a 17 03 03 dd   .\..@....B..|.V.
        00c0  b7 b3 b8 da 4b ce 25 09 c5 28 96 94 5a 26 bc 15   ....K.%..(..Z&..
        00d0  14 03 03 0f 56 15 03 03 5c 96 d5 92 fc d8 d8 d3   Ud..V..S\.......
        00e0  9d ed 40 06 6b 44 17

        {
            'numBytes0': '0030',
            'anyBytes':: '  01 f5 a8 98 00 00 4d 5a 03 00 ac 00 00 ',
            'recType': '17',
            'ascii1: '',
            'numBytes2': '',
            'major: '03',
            'ascii2: '',
            'numBytes3': '',
            'minor': '03',
            'ascii3': '   .....$.0...8.wL.',
            'numBytes4': '0040 ',
            'ascii4': '',
            'numBytes5': '',
            'length0': '00',
            'ascii5': '',
            'numBytes6': '',
            'length1': 'ac',
            'ascii6': '',
            'numBytes7': '',
            'handshakeType: '01'
        }

    Returns:
        groupdict: Named matches of the regex used to search for the TLS header
            in the hexdump of a packet.

    """
    # Match byte numbers.
    # Match any sequence of bytes.
    # Match TLS record type.
    # End of row. Match ASCII output, new line and byte numbers.
    # Match TLS major version.
    # End of row. Match ASCII output, new line and byte numbers.
    # Match TLS minor version.
    # End of row. Match ASCII output, new line and byte numbers.
    # Match first byte of TLS record length.
    # End of row. Match ASCII output, new line and byte numbers.
    # Match second byte of TLS record length.
    # End of row. Match ASCII output, new line and byte numbers.
    # Match handshake type. Only valid if record type is 22, i.e., 0x16
    # pattern = re.compile(
    #     "(?P<numBytes0>[0-9a-f]{4})?"
    #     "(?P<anyBytes0>  ([0-9a-f]{2} )*)?"
    #     "(?P<recType>14|15|16|17)"
    #     "((?P<ascii1>   .{16}\n)(?P<numBytes2>[0-9a-f]{4} ))?"
    #     " (?P<major>0[123])"
    #     "((?P<ascii2>   .{16}\n)(?P<numBytes3>[0-9a-f]{4} ))?"
    #     " (?P<minor>0[123])"
    #     "((?P<ascii3>   .{16}\n)(?P<numBytes4>[0-9a-f]{4} ))?"
    #     " (?P<length0>[0-9a-f]{2})"
    #     "((?P<ascii4>   .{16}\n)(?P<numBytes5>[0-9a-f]{4} ))?"
    #     " (?P<length1>[0-9a-f]{2})"
    #     "((?P<ascii5>   .{16}\n)(?P<numBytes6>[0-9a-f]{4} ))?"
    #     " (?P<handshakeType>[0-9a-f]{2})"
    # )
    num_rows = int(byte_offset / 16)
    remove_rows = num_rows * 73
    removed_bytes = num_rows * 16
    offset = byte_offset - removed_bytes
    correction = 0
    tls_cmd = f'tshark -r {pth_to_trace} -Y "frame.number == {frame_number}" ' + \
              f'-o tcp.desegment_tcp_streams:FALSE -x -w {tmp_file_path}'
    out = subp.check_output(tls_cmd, shell=True).decode('utf8').strip()
    hexdump = extract_hexdump(out[remove_rows:])

    next_step = True
    group_dict = {'is_partial': offset + 5 >= len(hexdump)}
    if 'recType' in prev_content:
        group_dict['recType'] = prev_content['recType']
        correction += 1
    elif hexdump[offset] in ['14', '15', '16', '17'] and next_step:
        group_dict['recType'] = hexdump[offset]
    else:
        msg = f"Record Type not found in frame {frame_number}, offset is {offset}, hexdump at " +\
              f"offset is:\n{hexdump[offset:]}\nSkipped content is: \n" +\
              f"{hexdump[:offset]}"
        raise TraceFailure(url, "Record Type not found", msg)
    if 'major' in prev_content:
        group_dict['major'] = prev_content['major']
        correction += 1
    elif offset + 1 >= len(hexdump):
        pass
    elif hexdump[offset + 1 - correction] == '03' and next_step:
        group_dict['major'] = hexdump[offset + 1 - correction]
    else:
        msg = f"Major Version not found in frame {frame_number}, offset is {offset}, hexdump at " +\
              f"offset is:\n{hexdump[offset:]}\nSkipped content is: \n" +\
              f"{hexdump[:offset]}"
        raise TraceFailure(url, "Major Version not found", msg)
    if 'minor' in prev_content:
        group_dict['minor'] = prev_content['minor']
        correction += 1
    elif offset + 2 >= len(hexdump):
        pass
    elif re.fullmatch('0[123]', hexdump[offset + 2 - correction]) is None and next_step:
        msg = f"Minor Version not found in frame {frame_number}, offset is {offset}, hexdump at " + \
              f"offset is:\n{hexdump[offset:]}\nSkipped content is: \n" + \
              f"{hexdump[:offset]}"
        raise TraceFailure(url, "Major Version not found", msg)
    else:
        group_dict['minor'] = hexdump[offset + 2 - correction]
    if next_step:
        if 'length0' in prev_content:
            group_dict['length0'] = prev_content['length0']
            correction += 1
        else:
            if offset + 3 < len(hexdump):
                group_dict['length0'] = hexdump[offset + 3 - correction]
        if 'length1' in prev_content:
            group_dict['length1'] = prev_content['length1']
            group_dict['recordLength'] = prev_content.get('recordLength', int(hexdump[offset + 3] + hexdump[offset + 4], 16))
            correction += 1
        else:
            if offset + 4 < len(hexdump):
                group_dict['length1'] = hexdump[offset + 4 - correction]
                group_dict['recordLength'] = int(hexdump[offset + 3 - correction] + hexdump[offset + 4 - correction], 16)

        if offset + 5 < len(hexdump):
            group_dict['handshakeType'] = hexdump[offset + 5]  # In this case the header is complete --> no prev-content
    group_dict['prev_matched_bytes'] = correction
    return group_dict


def get_frame_size(groupdict: Dict[str, str]) -> int:
    """
    Helper function to retrieve the size of the TLS frame from the groupdict
    returned by `extract_record_info_re`.

    Args:
        groupdict: Dictionary with matches.

    Returns:
        Length of TLS frame in decimal form.
    """
    return int(groupdict['length0'].strip() + groupdict['length1'].strip(), 16)


def get_record_type(groupdict: Dict[str, str]) -> int:
    """
    Helper function to retrieve the record type of the TLS frame from the groupdict
    returned by `extract_record_info_re`.

    Args:
        groupdict: Dictionary with matches.

    Returns:
        TLS record type in decimal form.
    """
    return int(groupdict['recType'], 16)


def get_handshake_type(groupdict: Dict[str, str]) -> int:
    """
    Helper function to retrieve the handshake type of the TLS frame from the groupdict
    returned by `extract_record_info_re`.

    Args:
        groupdict: Dictionary with matches.

    Returns:
        TLS record handshake type in decimal form.
    """
    return int(groupdict['handshakeType'], 16)


def get_major_version(groupdict: Dict[str, str]) -> int:
    """
    Helper function to retrieve the major version of the TLS frame from the groupdict
    returned by `extract_record_info_re`.

    Args:
        groupdict: Dictionary with matches.

    Returns:
        Major version in decimal form.
    """
    return int(groupdict['major'], 16)


def get_minor_version(groupdict: Dict[str, str]) -> int:
    """
    Helper function to retrieve the minor version of the TLS frame from the groupdict
    returned by `extract_record_info_re`.

    Args:
        groupdict: Dictionary with matches.

    Returns:
        Minor version in decimal form.
    """
    return int(groupdict['minor'], 16)


def get_header_start_byte_from_group_dict(groupdict: Dict[str, str], tcp_header_offset: int) -> int:
    """
    Helper function to retrieve the start byte of the TLS frame in the TCP payload from the
    groupdict returned by `extract_record_info_re`.
    
    Args:
        groupdict: Dictionary with matches.
        tcp_header_offset: Begin of the TCP payload.

    Returns:
        Startbyte of the TLS record relative to the beginning of the TCP
        payload.
    """
    tmp = groupdict['numBytes0'].lstrip('0')
    row_offset = 0 if len(tmp) == 0 else int(tmp, 16)
    tmp = groupdict['anyBytes0'].strip()
    col_offset = 0 if len(tmp) == 0 else len(tmp.split(' '))
    return row_offset + col_offset - tcp_header_offset


def find_records(pth_to_trace: str, frame: Frame, record_number: int,
                 offset: int, records: List[TlsRecord], tmp_file_name: str,
                 prev_group_dict: Dict[str, str], url: str) -> Tuple[Dict[str, str], Union[int, None]]:
    """
    Extract all TLS records contained in a single L2 frame.

    Args:
        pth_to_trace: Path to the packet capture.
        frame: The current frame in which to look for TLS records.
        record_number: The number of the record (if one is detected).
        offset: Where to start looking for in the payload of the frame.
        records: Backup list in which records are recusively added to.

    Returns:
        remaining_bytes: The remaining number of bytes of the TLS record.
    """
    groupdict = extract_record_info_re(pth_to_trace, frame.frame_number,
                                       tmp_file_path=tmp_file_name,
                                       url=url,
                                       byte_offset=offset,
                                       prev_content=prev_group_dict)
    if groupdict['is_partial']:
        return groupdict, None
    else:
        record = TlsRecord(
            length=get_frame_size(groupdict),
            content_type=get_record_type(groupdict),
            major_version=get_major_version(groupdict),
            minor_version=get_minor_version(groupdict),
            direction=frame.direction,
            record_number=record_number,
            header_start_byte=offset - frame.tcp_header_offset,
            time_epoch=frame.time_epoch
        )
        if record.content_type == 22:
            record.handshake_type = get_handshake_type(groupdict)
        records.append(record)

        remaining_payload = frame.tcp_length - record.header_start_byte - \
                            (record.header_size - groupdict['prev_matched_bytes'])
        if remaining_payload <= record.length:
            # The TLS record reaches into a later packet. Calculate the remaining
            # content of the record. End of recursion.
            return groupdict, record.length - remaining_payload
        else:
            # The TLS record finishes within the same L2 frame. Search for the
            # beginning of the next TLS record, increase the offset to exclude
            # the already matched record.
            #print("Search from record in frame ", str(frame), '\n start at ',
            #      record.header_start_byte + record.header_size + record.length + frame.tcp_header_offset)
            return find_records(
                pth_to_trace=pth_to_trace,
                frame=frame,
                record_number=record_number + 1,
                offset=record.header_start_byte + (record.header_size - groupdict['prev_matched_bytes']) + record.length + frame.tcp_header_offset,
                records=records,
                tmp_file_name=tmp_file_name,
                prev_group_dict={},
                url=url
            )


def label_main_flow(pth_to_trace: str, main_flow: pd.DataFrame, tmp_file_name: str, url: str) -> MainFlow:
    """
    Label all data packets in a TLS session and extract the contained TLS records.

    Args:
        pth_to_trace: Path to the packet capture.
        main_flow: DataFrame containing the packets of the TLS session. Output
            of the function `get_main_flow`.

    Returns:
        flow: Flow record with labeled packets and TLS records.
    """
    flow = MainFlow(
        src_ip=main_flow.at[0, 'ip.src'],
        src_port=int(main_flow.at[0, 'tcp.srcport']),
        dst_ip=main_flow.at[0, 'ip.dst'],
        dst_port=int(main_flow.at[0, 'tcp.dstport']),
        tcp_header_length=int(main_flow.at[0, 'tcp.hdr_len']),
        ip_header_length=int(main_flow.at[0, 'ip.hdr_len']),
        total_num_frames=int(main_flow.shape[0])
    )
    # remaining bytes of records has to be tracked separately for each direction.
    # Records from client and server can interleave, i.e., a packet with the
    # beginning of a tLS record send by the client can be followed by a packet
    # send from the server.
    remaining_bytes = {flow.src_ip: 0, flow.dst_ip: 0}
    next_seq = {flow.src_ip: None, flow.dst_ip: None}
    prev_records = {flow.src_ip: None, flow.dst_ip: None}
    prev_group_dicts = {flow.src_ip: {'is_partial': False}, flow.dst_ip: {'is_partial': False}}
    flow_iterator = MainFlowIter(main_flow)
    for row in flow_iterator:
        frame_src_ip = row.at['ip.src']
        records = []
        seq_number = row.at['tcp.seq']
        if next_seq[frame_src_ip] is None:
            next_seq[frame_src_ip] = seq_number + row.at['tcp.len']
        elif seq_number == next_seq[frame_src_ip]:
            next_seq[frame_src_ip] = seq_number + row.at['tcp.len']
        else:
            # Sequence number is missing --> happens during end of trace.
            # return flow, an be found out by comparing frames vs. number of
            # extracted packets.
            return flow

        frame = Frame(
            tcp_length=row.at['tcp.len'],
            frame_number=row.at['frame.number'],
            time_epoch=row.at['frame.time_epoch'],
            direction=-1 if row.at['ip.src'] == flow.src_ip else 1,
            ip_header_length=row.at['ip.hdr_len'],
            tcp_header_length=row.at['tcp.hdr_len']
        )
        # print(frame.frame_number)

        if remaining_bytes[frame_src_ip] == 0:
            # The previous packet terminated the active TLS record, i.e., no
            # more bytes are expected. CUrrent frame should thus include a
            # TLS header at the beginning of the TCP payload.
            records = []
            gd, rb = find_records(
                pth_to_trace=pth_to_trace,
                frame=frame,
                record_number=len(flow.tls_records),
                offset=remaining_bytes[frame_src_ip] + frame.tcp_header_offset,
                records=records,
                tmp_file_name=tmp_file_name,
                prev_group_dict=prev_group_dicts[frame_src_ip] \
                    if prev_group_dicts[frame_src_ip]['is_partial'] else {},
                url=url
            )
            prev_group_dicts[frame_src_ip] = gd
            if len(records) > 0:
                prev_records[frame_src_ip] = records[-1]
                frame.tls_records.extend(records)
            if gd['is_partial']:
                remaining_bytes[frame_src_ip] = 0
            else:
                remaining_bytes[frame_src_ip] = rb
        else:
            # Previous frame has not yet finished, i.e., some bytes are still
            # missing.
            record = TlsRecord.from_record(prev_records[frame_src_ip])
            record.direction = frame.direction
            frame.tls_records.append(record)
            if remaining_bytes[frame_src_ip] < frame.tcp_length:
                # The previous record finishes somewhere within this frame. Thus,
                # search for the beginning of a new header in the payload of the
                # current frame.
                records = []
                gd, rb = find_records(
                    pth_to_trace=pth_to_trace,
                    frame=frame,
                    record_number=len(flow.tls_records),
                    offset=remaining_bytes[frame_src_ip] + frame.tcp_header_offset,
                    records=records,
                    tmp_file_name=tmp_file_name,
                    prev_group_dict=prev_group_dicts[frame_src_ip]
                        if prev_group_dicts[frame_src_ip]['is_partial'] else {},
                    url=url
                )
                prev_group_dicts[frame_src_ip] = gd
                if len(records) > 0:
                    prev_records[frame_src_ip] = records[-1]
                    frame.tls_records.extend(records)
                if gd['is_partial']:
                    remaining_bytes[frame_src_ip] = 0
                else:
                    remaining_bytes[frame_src_ip] = rb
            else:
                # The current packet carries payload of the previous TLS record
                # only and does not contain a new TLS record header.
                remaining_bytes[frame_src_ip] -= frame.tcp_length
        flow.frames.append(frame)
        flow.tls_records.extend(records)
    return flow


def get_hostname(url: str) -> str:
    """
    Extract the hostname from an URL. URL has to start with https://.

    Args:
        url: URL that has been queried.

    Returns:
        hostname
    """
    assert url.startswith('https://'), "URL expected to start with https://"
    hostname = url[len('https://'):]
    idx = hostname.find('/')
    # assert idx > 0, f"Could not find slash in url {hostname}."
    if idx < 0:
        return hostname
    else:
        return hostname[:idx]


def extract_main_flow(pth_to_trace: str, hostname: str, url: str) -> pd.DataFrame:
    """
    Utility function to get the main flow given a pacap file and the hostname.

    Args:
        pth_to_trace: Path to packet capture file.
        hostname: Name of remote server.

    Returns:
        main_flow: DataFrame with packets of main flow.
    """
    ips = get_ips_from_dns(pth_to_trace, hostname, url)
    client_hello = get_client_hello(pth_to_trace, ips[0])
    main_flow = get_main_flow(pth_to_trace, client_hello, url)
    first_src_ip = main_flow.at[0, 'ip.src']
    assert first_src_ip.startswith('172.17.0'), 'Source IP incorrect, expected it to ' + \
        f'be in the 172.17.0.0/24 block, but was {first_src_ip}.'
    return main_flow


def extract_store_flow(pth_to_trace: str, out_file: str, hostname: str, url: str) -> None:
    ips = get_ips_from_dns(pth_to_trace, hostname, url)
    client_hello = get_client_hello(pth_to_trace, ips[0])
    src_ip, src_port, dst_ip, dst_port = four_tuple_from_client_hello(client_hello)
    main_flow_cmd = f'tshark -r {pth_to_trace} ' \
                    f'-Y "((ip.src == {src_ip}   && ' \
                    f'tcp.srcport == {src_port}  && ' \
                    f'ip.dst == {dst_ip} && ' \
                    f'tcp.dstport == {dst_port}) || ' \
                    f'(ip.src == {dst_ip}        && ' \
                    f'tcp.srcport == {dst_port} && ' \
                    f'ip.dst == {src_ip} && ' \
                    f'tcp.dstport == {src_port}))" ' \
                    f'-w {out_file}'
    out = subp.check_output(main_flow_cmd, shell=True)


def _extract_tls_records(pth_to_trace: str, hostname: str, tmp_file_name: str, url: str) -> MainFlow:
    """
    Label a trace, i.e. get the TLS records contained in every packet, as well
    as the TLS records.

    Args:
        pth_to_trace: Path to the packet capture file.
        hostname: Name of remote server.

    Returns:
        flow: Parsed frames and TLS records.
    """
    main_flow = extract_main_flow(pth_to_trace, hostname, url)
    flow = label_main_flow(pth_to_trace, main_flow, tmp_file_name, url)
    return flow


def extract_tls_records(pth_to_trace: str, url: str, tmp_file_name: str) -> MainFlow:
    """
    Label a trace, i.e. get the TLS records contained in every packet, as well
    as the TLS records.

    Args:
        pth_to_trace: Path to the packet capture file.
        url: URL that has been queried.

    Returns:
        flow: Parsed frames and TLS records.
    """
    return _extract_tls_records(pth_to_trace, get_hostname(url), tmp_file_name, url)


class FrameToSymbol(object):
    def __init__(self, bin_edges: np.array = None, directional_length: bool = False,
                 skip_record_type: bool = False):
        self.bin_edges = bin_edges
        self.directional_length = directional_length
        self.skip_record_type = skip_record_type

    def __call__(self, frame: Frame | Dict[str, Any]) -> str:
        """
        Convert a Frame object to a symbol. The frame size is discretized based on
        the passed edges.

        The returned symbol has the form: <TLS_RECORD_TYPES>;<FRAME_LENGTH>;<DIRECTION>.
        - TLS_RECORD_TYPES is a list of the record types contained in the frame.
            For example: 22:2|22:10|23
        - FRAME_LENGTH: Is the (discretized) TCP length of the frame.
        - DIRECTION := (S|C) gives the direction of the packet. S is the direction
            of server to client and C the direction of client to server.

        Args:
            frame: Frame that should be symbolized.
            bin_edges: The edges of a histogram used to discretize the data. If None
                is passed no discretization is applied and the raw frame size used.
            directional_length: Whether the length should be multiplied with direction,
                i.e., negative length for outgoing and positive length for incoming
                frames.

        Returns:
            symbol: String symbol of frame.
        """
        length = frame.tcp_length if type(frame) == Frame else frame['tcp_length']
        direction = frame.direction if type(frame) == Frame else frame['direction']
        if self.directional_length:
            length *= direction

        disretized_size = length if self.bin_edges is None \
            else np.argmin(np.abs(self.bin_edges - length))
        record_types = ''
        for r in frame.tls_records if type(frame) == Frame else frame['tls_records']:
            ctype = r.content_type if type(r) == TlsRecord else r['content_type']
            record_types = record_types + f'|{ctype}'
            if ctype == 22:
                htype = r.handshake_type if type(r) == TlsRecord else r['handshake_type']
                record_types = record_types + f':{htype}'
        record_types = record_types[1:]
        direction = 'C' if direction == -1 else 'S'
        if self.skip_record_type:
            symbol = f';{disretized_size};{direction}'
        else:
            symbol = f'{record_types};{disretized_size};{direction}'
        return symbol


class RecordToSymbol(object):
    def __init__(self, bin_edges: np.array = None, directional_length: bool = False,
                 skip_record_type: bool = False):
        self.bin_edges = bin_edges
        self.directional_length = directional_length
        self.skip_record_type = skip_record_type

    def __call__(self, record: TlsRecord | Dict[str, Any]) -> str:
        """
        Convert a Frame object to a symbol. The frame size is discretized based on
        the passed edges.

        The returned symbol has the form: <TLS_RECORD_TYPE>;<FRAME_LENGTH>;<DIRECTION>.
        - TLS_RECORD_TYPE is the type of the TLS record, e.g., 22:1, or 23.
        - FRAME_LENGTH: Is the (discretized) TCP length of the record.
        - DIRECTION := (S|C) gives the direction of the record. S is the direction
            of server to client and C the direction of client to server.

        Args:
            record: Frame that should be symbolized.
            bin_edges: The edges of a histogram used to discretize the data. If None
                is passed no discretization is applied and the raw frame size used.
            directional_length: Whether the length should be multiplied with direction,
                i.e., negative length for outgoing and positive length for incoming
                frames.

        Returns:
            symbol: String symbol of frame.
        """
        rlength = record.length if type(record) == TlsRecord else record['length']
        ctype = record.content_type if type(record) == TlsRecord else record['content_type']
        htype = record.handshake_type if type(record) == TlsRecord else record['handshake_type']
        direction = record.direction if type(record) == TlsRecord else record['direction']
        if self.directional_length:
            rlength *= direction

        disretized_size = rlength if self.bin_edges is None \
            else np.argmin(np.abs(self.bin_edges - rlength))
        record_types = f'{ctype}'
        if ctype == 22:
            record_types += f':{htype}'
        direction = 'C' if direction == -1 else 'S'
        if self.skip_record_type:
            symbol = f';{disretized_size};{direction}'
        else:
            symbol = f'{record_types};{disretized_size};{direction}'
        return symbol


class MainFlowToSymbol(object):
    def __init__(self, seq_length: int, to_symbolize: ClassVar | str, bin_edges: np.array,
                 directional_length: bool = False, skip_record_type: bool = False,
                 skip_handshake: bool = False, direction_to_filter: int = 0):
        """
        Convert a main flow to a sequence of symbols.

        Args:
            seq_length: How many elements (TLSRecords, Frames) should be icluded
                in the returned sequence.
            to_symbolize: Class of the object that should be symbolized, must be in
                {TlsRecord, Frame}.
            bin_edges: The edges of a histogram used to discretize the data. If None
                is passed no discretization is applied and the raw frame size used.
            directional_length: Whether the length should be multiplied with direction,
                i.e., negative length for outgoing and positive length for incoming
                frames.
            skip_record_type: Skip the TLS record type info in the symbols, i.e.,
                use only direction and length info.
            skip_handshake: If true skip all TLS handshake packets/records.
            direction_to_filter: Set to {-1, 1} if direction of host to server,
                or server to host should be filtered. Those packets are "not observed".
        """
        self.seq_length = seq_length
        self.direction_to_filter = direction_to_filter
        if type(bin_edges) == list:
            self.bin_edges = np.array(bin_edges)
        else:
            self.bin_edges = bin_edges
        self.directional_length = directional_length
        self.skip_record_type = skip_record_type
        self.skip_handshake = skip_handshake
        if type(to_symbolize) == str:
            self._to_symbolize = {
                'tlsrecord': TlsRecord,
                'frame': Frame,
                'record': TlsRecord
            }[to_symbolize.lower()]
        else:
            self._to_symbolize = to_symbolize
        self.symbolizer = {
            TlsRecord: RecordToSymbol,
            Frame: FrameToSymbol
        }[self._to_symbolize](self.bin_edges, self.directional_length, self.skip_record_type)

    def get_records_from_frames(self, frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Handles passed dictionaries. Returns dictionary records. Skips all records
        based on the set skip_handshake and direction_to_filter attributes.

        Args:
            frames: List of serialized Frame objects.

        Returns:
            records: List of serialized Record objects.
        """
        records = []
        for frame in frames:
            to_extend = []
            for r in frame['tls_records']:
                if (self.skip_handshake and r['handshake_type'] is not None) \
                        or r['direction'] == self.direction_to_filter:
                    pass
                else:
                    to_extend.append(r)
            # to_extend = [r for r in frame['tls_records'] if (self.skip_handshake and
            #              r['handshake_type'] is not None) or r['direction'] != self.direction_to_filter]
            if len(to_extend) > 0:
                records.extend(frame['tls_records'])
            if len(records) >= self.seq_length:
                break
        return records[:self.seq_length]

    def get_records_from_main_flow(self, tls_records: List[TlsRecord]) -> List[TlsRecord]:
        """
        Get all applicable records from a list of records. IF applicable, filter
        all handshake messages and all records traveling in a specific direction.

        Args:
            tls_records: List of TLSRecord objects.

        Returns:
            records: Filtered records.
        """
        records = []
        for record in tls_records:
            if (self.skip_handshake and record.handshake_type is not None) or \
                    record.direction == self.direction_to_filter:
                continue
            records.append(record)
            if len(records) >= self.seq_length:
                break
        return records

    def get_frames(self, frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Get applicable frames from a list of dictionary serialized frame objects.

        Args:
            frames: List of Frame objects in Dict representation.

        Returns:
            ret: List of applicable dict serialized frame objects.
        """
        ret = []
        for frame in frames:
            r = frame['tls_records'][0]
            if (self.skip_handshake and r['handshake_type'] is not None) or \
                    r['direction'] == self.direction_to_filter:
                continue
            ret.append(frame)
            if len(ret) >= self.seq_length:
                break
        return ret

    def get_frames_from_mainflow(self, frames: List[Frame]) -> List[Frame]:
        """
        Get all applicable frames from a list of frames. IF applicable, filter
        all handshake messages and all frames traveling in a specific direction.

        Args:
            frames: List of Frame objects.

        Returns:
            ret: Filtered frames.
        """
        ret = []
        for frame in frames:
            r = frame.tls_records[0]
            if (self.skip_handshake and r.handshake_type is not None) or \
                    self.direction_to_filter == r.direction:
                continue
            ret.append(frame)
            if len(ret) >= self.seq_length:
                break
        return ret

    def __call__(self, main_flow: MainFlow | Dict[str, Any]) -> List[str]:
        if type(main_flow) == MainFlow and self._to_symbolize == TlsRecord:
            elements_to_symbolize = self.get_records_from_main_flow(main_flow.tls_records)
        elif type(main_flow) == MainFlow and self._to_symbolize == Frame:
            elements_to_symbolize = self.get_frames_from_mainflow(main_flow.frames)
        elif type(main_flow) == dict and self._to_symbolize == TlsRecord:
            elements_to_symbolize = self.get_records_from_frames(main_flow['frames'])
        elif type(main_flow) == dict and self._to_symbolize == Frame:
            elements_to_symbolize = self.get_frames(main_flow['frames'])
        else:
            raise KeyError("Expected {{MainFlow, Dict[str, Any]}} for main flow and"
                           f" {{TlsRecord, Frame}} for to_symbolize, got {str(type(main_flow))}"
                           f" and {str(type(self._to_symbolize))} instead.")
        seq = []
        for i, element in enumerate(elements_to_symbolize):
            seq.append(self.symbolizer(element))
        return seq


if __name__ == '__main__':
    # pcap_path = '/opt/project/data/devel-traces/cnet_firefox_gatherer-01-gxjxb_304479.pcapng'
    # hostname = 'www.cnet.com'
    pcap_path = '/opt/project/data/devel-traces/ebay_firefox_gatherer-01-pqn4d_87069794.pcapng'
    # hostname = 'discord.com'
    # pcap_path = '/opt/project/data/devel-traces/discord_chromium_gatherer-01-n8zqn_95211110.pcapng'
    hostname = 'www.ebay.de'
    # pcap_path = '/opt/project/data/devel-traces/chaturbate_firefox_gatherer-01-vjv4b_5621317.pcapng'
    # hostname = 'chaturbate.com'
    extract_store_flow(pcap_path, '/opt/project/data/devel-traces/mainflow.pcapng', hostname, 'https://www.testurl.de')
    flow = _extract_tls_records(pcap_path, hostname, '/opt/project/data/tmp.bin', 'https://www.testurl.de')
    d = flow.to_dict()
    flow2 = MainFlow.from_dict(d)
    defense = RandomRecordSizeDefense(30, lambda x: np.random.randint(100, 16000))
    flow2_o = defense(flow2)
    main_flow_to_symbol_f = MainFlowToSymbol(10, Frame, None, skip_handshake=True, direction_to_filter=-1)
    main_flow_to_symbol_r = MainFlowToSymbol(10, TlsRecord, None, direction_to_filter=1)
    print(f"Includes {len(flow2.frames)} of {flow2.total_num_frames} packets.")
    print(main_flow_to_symbol_f(flow2))
    print(main_flow_to_symbol_r(flow2))
