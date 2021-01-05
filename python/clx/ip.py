# Copyright (c) 2019, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cudf
import numpy as np


def ip_to_int(values):
    """
    Convert string column of IP addresses to integer values.
    **Addresses must be IPv4. IPv6 not yet supported.**

    :param values: IP addresses to be converted
    :type values: cudf.Series
    :return: Integer representations of IP addresses
    :rtype: cudf.Series

    Examples
    --------
    >>> import clx.ip
    >>> import cudf
    >>> clx.ip.ip_to_int(cudf.Series(["192.168.0.1","10.0.0.1"]))
    0      89088434
    1    1585596973
    dtype: int64
    """
    return cudf.Series(values.str.ip2int())


def int_to_ip(values):
    """
    Convert integer column to IP addresses.
    **Addresses must be IPv4. IPv6 not yet supported.**

    :param values: Integers to be converted
    :type values: cudf.Series
    :return: IP addresses
    :rtype: cudf.Series

    Examples
    --------
    >>> import clx.ip
    >>> import cudf
    >>> clx.ip.int_to_ip(cudf.Series([3232235521, 167772161]))
    0     5.79.97.178
    1    94.130.74.45
    dtype: object
    """
    return cudf.Series(values._column.int2ip())


def is_ip(ips):
    """
    Indicates whether each address is an ip string.
    **Addresses must be IPv4. IPv6 not yet supported.**

    :param ips: IP addresses
    :type values: cudf.Series
    :return: booleans
    :rtype: cudf.Series

    Examples
    --------
    >>> import clx.ip
    >>> import cudf
    >>> clx.ip.is_ip(cudf.Series(["192.168.0.1","10.123.0"]))
    0     True
    1    False
    dtype: bool
    """
    is_ip_REGEX = r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$"
    return ips.str.match(is_ip_REGEX)


def is_reserved(ips):
    """
    Indicates whether each address is reserved.
    **Addresses must be IPv4. IPv6 not yet supported.**

    :param ips: IP addresses
    :type values: cudf.Series
    :return: booleans
    :rtype: cudf.Series

    Examples
    --------
    >>> import clx.ip
    >>> import cudf
    >>> clx.ip.is_reserved(cudf.Series(["127.0.0.1","10.0.0.1"]))
    0    False
    1    False
    dtype: bool
    """
    reserved_ipv4_REGEX = r"^(2(4[0-9]|5[0-5]))\.([0-9]|[1-9][0-9]|1([0-9][0-9])|2([0-4][0-9]|5[0-5]))\.([0-9]|[1-9][0-9]|1([0-9][0-9])|2([0-4][0-9]|5[0-5]))\.([0-9]|[1-9][0-9]|1([0-9][0-9])|2([0-4][0-9]|5[0-5]))$"
    return ips.str.match(reserved_ipv4_REGEX)


def is_loopback(ips):
    """
    Indicates whether each address is loopback.
    **Addresses must be IPv4. IPv6 not yet supported.**

    :param ips: IP addresses
    :type values: cudf.Series
    :return: booleans
    :rtype: cudf.Series

    Examples
    --------
    >>> import clx.ip
    >>> import cudf
    >>> clx.ip.is_loopback(cudf.Series(["127.0.0.1","10.0.0.1"]))
    0     True
    1    False
    dtype: bool
    """
    loopback_ipv4_REGEX = r"^127\.([0-9]|[1-9][0-9]|1([0-9][0-9])|2([0-4][0-9]|5[0-5]))\.([0-9]|[1-9][0-9]|1([0-9][0-9])|2([0-4][0-9]|5[0-5]))\.([0-9]|[1-9][0-9]|1([0-9][0-9])|2([0-4][0-9]|5[0-5]))$"
    return ips.str.match(loopback_ipv4_REGEX)


def is_link_local(ips):
    """
    Indicates whether each address is link local.
    **Addresses must be IPv4. IPv6 not yet supported.**

    :param ips: IP addresses
    :type values: cudf.Series
    :return: booleans
    :rtype: cudf.Series

    Examples
    --------
    >>> import clx.ip
    >>> import cudf
    >>> clx.ip.is_link_local(cudf.Series(["127.0.0.1","169.254.123.123"]))
    0    False
    1    True
    dtype: bool
    """
    link_local_ipv4_REGEX = r"^169\.254\.([0-9]|[1-9][0-9]|1([0-9][0-9])|2([0-4][0-9]|5[0-5]))\.([0-9]|[1-9][0-9]|1([0-9][0-9])|2([0-4][0-9]|5[0-5]))$"
    return ips.str.match(link_local_ipv4_REGEX)


def is_unspecified(ips):
    """
    Indicates whether each address is unspecified.
    **Addresses must be IPv4. IPv6 not yet supported.**

    :param ips: IP addresses
    :type values: cudf.Series
    :return: booleans
    :rtype: cudf.Series

    Examples
    --------
    >>> import clx.ip
    >>> import cudf
    >>> clx.ip.is_unspecified(cudf.Series(["127.0.0.1","10.0.0.1"]))
    0    False
    1    False
    dtype: bool
    """
    unspecified_REGEX = r"^0\.0\.0\.0$"
    return ips.str.match(unspecified_REGEX)


def is_multicast(ips):
    """
    Indicates whether each address is multicast.
    **Addresses must be IPv4. IPv6 not yet supported.**

    :param ips: IP addresses
    :type values: cudf.Series
    :return: booleans
    :rtype: cudf.Series

    Examples
    --------
    >>> import clx.ip
    >>> import cudf
    >>> clx.ip.is_multicast(cudf.Series(["127.0.0.1","224.0.0.0"]))
    0    False
    1    True
    dtype: bool
    """
    is_multicast_ipv4_REGEX = r"^(2(2[4-9]|3[0-9]))\.([0-9]|[1-9][0-9]|1([0-9][0-9])|2([0-4][0-9]|5[0-5]))\.([0-9]|[1-9][0-9]|1([0-9][0-9])|2([0-4][0-9]|5[0-5]))\.([0-9]|[1-9][0-9]|1([0-9][0-9])|2([0-4][0-9]|5[0-5]))$"
    return ips.str.match(is_multicast_ipv4_REGEX)


def is_private(ips):
    """
    Indicates whether each address is private.
    **Addresses must be IPv4. IPv6 not yet supported.**

    :param ips: IP addresses
    :type values: cudf.Series
    :return: booleans
    :rtype: cudf.Series

    Examples
    --------
    >>> import clx.ip
    >>> import cudf
    >>> clx.ip.is_private(cudf.Series(["127.0.0.1","207.46.13.151"]))
    0    True
    1    False
    dtype: bool
    """
    private_REGEX = r"((^0\.([0-9]|[1-9][0-9]|1([0-9][0-9])|2([0-4][0-9]|5[0-5]))\.([0-9]|[1-9][0-9]|1([0-9][0-9])|2([0-4][0-9]|5[0-5]))\.([0-9]|[1-9][0-9]|1([0-9][0-9])|2([0-4][0-9]|5[0-5]))$)|(^10\.([0-9]|[1-9][0-9]|1([0-9][0-9])|2([0-4][0-9]|5[0-5]))\.([0-9]|[1-9][0-9]|1([0-9][0-9])|2([0-4][0-9]|5[0-5]))\.([0-9]|[1-9][0-9]|1([0-9][0-9])|2([0-4][0-9]|5[0-5]))$)|(^127\.([0-9]|[1-9][0-9]|1([0-9][0-9])|2([0-4][0-9]|5[0-5]))\.([0-9]|[1-9][0-9]|1([0-9][0-9])|2([0-4][0-9]|5[0-5]))\.([0-9]|[1-9][0-9]|1([0-9][0-9])|2([0-4][0-9]|5[0-5]))$)|(^169\.254\.([0-9]|[1-9][0-9]|1([0-9][0-9])|2([0-4][0-9]|5[0-5]))\.([0-9]|[1-9][0-9]|1([0-9][0-9])|2([0-4][0-9]|5[0-5]))$)|(^172\.(1[6-9]|2[0-9]|3[0-1])\.([0-9]|[1-9][0-9]|1([0-9][0-9])|2([0-4][0-9]|5[0-5]))\.([0-9]|[1-9][0-9]|1([0-9][0-9])|2([0-4][0-9]|5[0-5]))$)|(^192\.0\.0\.([0-7])$)|(^192\.0\.0\.(1(7[0-1]))$)|(^192\.0\.2\.([0-9]|[1-9][0-9]|1([0-9][0-9])|2([0-4][0-9]|5[0-5]))$)|(^192\.168\.([0-9]|[1-9][0-9]|1([0-9][0-9])|2([0-4][0-9]|5[0-5]))\.([0-9]|[1-9][0-9]|1([0-9][0-9])|2([0-4][0-9]|5[0-5]))$)|(^198\.(1[8-9])\.([0-9]|[1-9][0-9]|1([0-9][0-9])|2([0-4][0-9]|5[0-5]))\.([0-9]|[1-9][0-9]|1([0-9][0-9])|2([0-4][0-9]|5[0-5]))$)|(^198\.51\.100\.([0-9]|[1-9][0-9]|1([0-9][0-9])|2([0-4][0-9]|5[0-5]))$)|(^203\.0\.113\.([0-9]|[1-9][0-9]|1([0-9][0-9])|2([0-4][0-9]|5[0-5]))$)|(^(2(4[0-9]|5[0-5]))\.([0-9]|[1-9][0-9]|1([0-9][0-9])|2([0-4][0-9]|5[0-5]))\.([0-9]|[1-9][0-9]|1([0-9][0-9])|2([0-4][0-9]|5[0-5]))\.([0-9]|[1-9][0-9]|1([0-9][0-9])|2([0-4][0-9]|5[0-5]))$)|(^255\.255\.255\.255$))"
    return ips.str.match(private_REGEX)


def is_global(ips):
    """
    Indicates whether each address is global.
    **Addresses must be IPv4. IPv6 not yet supported.**

    :param ips: IP addresses
    :type values: cudf.Series
    :return: booleans
    :rtype: cudf.Series

    Examples
    --------
    >>> import clx.ip
    >>> import cudf
    >>> clx.ip.is_global(cudf.Series(["127.0.0.1","207.46.13.151"]))
    0    False
    1    True
    dtype: bool
    """
    is_global_REGEX = r"^(100\.(6[4-9]|[7-9][0-9]|1([0-1][0-9]|2[0-7]))\.([0-9]|[1-9][0-9]|1([0-9][0-9])|2([0-4][0-9]|5[0-5]))\.([0-9]|[1-9][0-9]|1([0-9][0-9])|2([0-4][0-9]|5[0-5]))$)"
    part1 = ips.str.match(is_global_REGEX)
    part2 = is_private(ips)
    result = ~part1 & ~part2
    return result


def _netmask_kernel(idx, out1, out2, out3, out4, kwarg1):
    for i, ipnum in enumerate(idx):
        out1[i] = int(kwarg1 / 16777216) % 256
        out2[i] = int(kwarg1 / 65536) % 256
        out3[i] = int(kwarg1 / 256) % 256
        out4[i] = int(kwarg1) % 256


def netmask(ips, prefixlen=16):
    """
    Compute a column of netmasks for a column of IP addresses.
    **Addresses must be IPv4. IPv6 not yet supported.**

    :param ips: IP addresses
    :type values: cudf.Series
    :param prefixlen: Length of the network prefix, in bits, for IPv4 addresses
    :type prefixlen: int
    :return: netmasks
    :rtype: cudf.Series

    Examples
    --------
    >>> import clx.ip
    >>> import cudf
    >>> clx.ip.netmask(cudf.Series(["192.168.0.1","10.0.0.1"]), prefixlen=16)
    0    255.255.0.0
    1    255.255.0.0
    Name: net_mask, dtype: object
    """
    _ALL_ONES = (2 ** 32) - 1
    mask_int = _ALL_ONES ^ (_ALL_ONES >> prefixlen)
    df = cudf.DataFrame()
    df["idx"] = ips.index
    x = df.apply_rows(
        _netmask_kernel,
        incols=["idx"],
        outcols=dict(out1=np.int64, out2=np.int64, out3=np.int64, out4=np.int64),
        kwargs=dict(kwarg1=mask_int),
    )

    out1 = x["out1"].astype(str)
    out2 = x["out2"].astype(str)
    out3 = x["out3"].astype(str)
    out4 = x["out4"].astype(str)
    df["net_mask"] = out1.str.cat(out2, sep=".").str.cat(out3, sep=".").str.cat(out4, sep=".")
    return df["net_mask"]


def _hostmask_kernel(idx, out1, out2, out3, out4, kwarg1):
    for i, ipnum in enumerate(idx):
        out1[i] = int(kwarg1 / 16777216) % 256
        out2[i] = int(kwarg1 / 65536) % 256
        out3[i] = int(kwarg1 / 256) % 256
        out4[i] = int(kwarg1) % 256


def hostmask(ips, prefixlen=16):
    """
    Compute a column of hostmasks for a column of IP addresses.
    **Addresses must be IPv4. IPv6 not yet supported.**

    :param ips: IP addresses
    :type values: cudf.Series
    :param prefixlen: Length of the network prefix, in bits, for IPv4 addresses
    :type prefixlen: int
    :return: hostmasks
    :rtype: cudf.Series

    Examples
    --------
    >>> import clx.ip
    >>> import cudf
    >>> clx.ip.hostmask(cudf.Series(["192.168.0.1","10.0.0.1"], prefixlen=16)
    0    0.0.255.255
    1    0.0.255.255
    Name: hostmask, dtype: object
    """
    _ALL_ONES = (2 ** 32) - 1
    host_mask_int = int(_ALL_ONES ^ (_ALL_ONES >> prefixlen)) ^ _ALL_ONES
    df = cudf.DataFrame()
    df["idx"] = ips.index
    x = df.apply_rows(
        _hostmask_kernel,
        incols=["idx"],
        outcols=dict(out1=np.int64, out2=np.int64, out3=np.int64, out4=np.int64),
        kwargs=dict(kwarg1=host_mask_int),
    )

    out1 = x["out1"].astype(str)
    out2 = x["out2"].astype(str)
    out3 = x["out3"].astype(str)
    out4 = x["out4"].astype(str)
    df["hostmask"] = out1.str.cat(out2, sep=".").str.cat(out3, sep=".").str.cat(out4, sep=".")
    return df["hostmask"]


def _mask_kernel(masked_ip_int, out1, out2, out3, out4, kwarg1):
    for i, ipnum in enumerate(masked_ip_int):
        out1[i] = int(ipnum / 16777216) % 256
        out2[i] = int(ipnum / 65536) % 256
        out3[i] = int(ipnum / 256) % 256
        out4[i] = int(ipnum) % 256


def mask(ips, masks):
    """
    Apply a mask to a column of IP addresses.
    **Addresses must be IPv4. IPv6 not yet supported.**

    :param ips: IP addresses
    :type values: cudf.Series
    :param masks: The host or subnet masks to be applied
    :type masks: cudf.Series
    :return: masked IP addresses
    :rtype: cudf.Series

    Examples
    --------
    >>> import clx.ip
    >>> import cudf
    >>> input_ips = cudf.Series(["192.168.0.1","10.0.0.1"])
    >>> input_masks = cudf.Series(["255.255.0.0", "255.255.0.0"])
    >>> clx.ip.mask(input_ips, input_masks)
    0    192.168.0.0
    1       10.0.0.0
    Name: mask, dtype: object
    """
    df = cudf.DataFrame()
    df["int_mask"] = masks.str.ip2int()
    df["int_ip"] = ips.str.ip2int()
    df["masked_ip_int"] = df["int_mask"] & df["int_ip"]

    x = df.apply_rows(
        _mask_kernel,
        incols=["masked_ip_int"],
        outcols=dict(out1=np.int64, out2=np.int64, out3=np.int64, out4=np.int64),
        kwargs=dict(kwarg1=0),
    )

    out1 = x["out1"].astype(str)
    out2 = x["out2"].astype(str)
    out3 = x["out3"].astype(str)
    out4 = x["out4"].astype(str)
    df["mask"] = out1.str.cat(out2, sep=".").str.cat(out3, sep=".").str.cat(out4, sep=".")
    return df["mask"]
