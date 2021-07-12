# Copyright (c) 2021, NVIDIA CORPORATION.
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

# ref: https://docs.dnsdb.info/dnsdb-api/

import json
import requests
import urllib
import logging

log = logging.getLogger(__name__)


class FarsightLookupClient(object):
    """
    Wrapper class to query DNSDB record in various ways
    Example: by IP, DomainName

    :param server: Farsight server
    :param apikey: API key
    :param limit: limit
    :param http_proxy: HTTP proxy
    :param https_proxy: HTTPS proxy
    """
    def __init__(self, server, apikey, limit=None, http_proxy=None, https_proxy=None):
        self.server = server
        self.headers = {"Accept": "application/json", "X-Api-Key": apikey}
        self.limit = limit
        self.proxy_args = self.__get_proxy_args(http_proxy, https_proxy)

    def query_rrset(self, oname, rrtype=None, bailiwick=None, before=None, after=None):
        """
        Batch version of querying DNSDB by given domain name and time ranges.

        :param oname: DNS domain name.
        :type oname: str
        :param rrtype: The resource record type of the resource record, either using the standard DNS type mnemonic, or an RFC 3597 generic type, i.e. the string TYPE immediately followed by the decimal RRtype number.
        :type rrtype: str
        :param bailiwick: The “bailiwick” of an RRset in DNSDB observed via passive DNS replication is the closest enclosing zone delegated to a nameserver which served the RRset.
        :type bailiwick: str
        :param before: Output results seen before this time.
        :type before: UNIX timestamp
        :param after: Output results seen after this time.
        :type after: UNIX timestamp
        :return: Response
        :rtype: dict

        Examples
        --------
        >>> from clx.osi.farsight import FarsightLookupClient
        >>> client = FarsightLookupClient("https://localhost", "your-api-key")
        >>> client.query_rrset("www.dnsdb.info")
        {"status_code": 200,...}
        >>> client.query_rrset("www.dnsdb.info", rrtype="CNAME", bailiwick="dnsdb.info.", before=1374184718, after=1564909243,)
        {"status_code": 200,...}
        """
        quoted_name = self.__quote(oname)
        if bailiwick:
            if not rrtype:
                rrtype = "ANY"
            path = "rrset/name/%s/%s/%s" % (
                quoted_name,
                rrtype,
                self.__quote(bailiwick),
            )
        elif rrtype:
            path = "rrset/name/%s/%s" % (quoted_name, rrtype)
        else:
            path = "rrset/name/%s" % quoted_name
        return self.__query(path, before, after)

    def query_rdata_name(self, rdata_name, rrtype=None, before=None, after=None):
        """
        Query matches only a single DNSDB record of given owner name and time ranges.

        :param rdata_name: DNS domain name.
        :type rdata_name: str
        :param rrtype: The resource record type of the resource record, either using the standard DNS type mnemonic, or an RFC 3597 generic type, i.e. the string TYPE immediately followed by the decimal RRtype number.
        :type rrtype: str
        :param before: Output results seen before this time.
        :type before: UNIX timestamp
        :param after: Output results seen after this time.
        :type after: UNIX timestamp
        :return: Response
        :rtype: dict

        Examples
        --------
        >>> from clx.osi.farsight import FarsightLookupClient
        >>> client = FarsightLookupClient("https://localhost", "your-api-key", limit=1)
        >>> client.query_rdata_name("www.farsightsecurity.com")
        {"status_code": 200,...}
        >>> client.query_rdata_name("www.farsightsecurity.com", rrtype="PTR", before=1386638408, after=1561176503)
        {"status_code": 200,...}
        """
        quoted_name = self.__quote(rdata_name)
        if rrtype:
            path = "rdata/name/%s/%s" % (quoted_name, rrtype)
        else:
            path = "rdata/name/%s" % quoted_name
        return self.__query(path, before, after)

    def query_rdata_ip(self, rdata_ip, before=None, after=None):
        """
        Query to find DNSDB records matching a specific IP address with given time range.

        :param rdata_ip: The VALUE is one of an IPv4 or IPv6 single address, with a prefix length, or with an address range. If a prefix is provided, the delimiter between the network address and prefix length is a single comma (“,”) character rather than the usual slash (“/”) character to avoid clashing with the HTTP URI path name separator..
        :type rdata_ip: str
        :param before: Output results seen before this time.
        :type before: UNIX timestamp
        :param after: Output results seen after this time.
        :type after: UNIX timestamp
        :return: Response
        :rtype: dict

        Examples
        --------
        >>> from clx.osi.farsight import FarsightLookupClient
        >>> client = FarsightLookupClient("https://localhost", "your-api-key", limit=1)
        >> client.query_rdata_ip("100.0.0.1")
        {"status_code": 200,...}
        >>> client.query_rdata_ip("100.0.0.1", before=1428433465, after=1538014110)
        {"status_code": 200,...}
        """
        path = "rdata/ip/%s" % rdata_ip.replace("/", ",")
        return self.__query(path, before, after)

    def __get(self, url):
        """
        submit http get request
        """
        response = requests.get(url, headers=self.headers, proxies=self.proxy_args)
        return response

    # queries dnsdb.
    def __query(self, path, before=None, after=None):
        res = []
        url = "%s/lookup/%s" % (self.server, path)
        params = self.__get_params(before, after)
        if params:
            url += "?{0}".format(urllib.parse.urlencode(params))
        response = self.__get(url)
        try:
            response.raise_for_status()
            self.__extract_response(response, res)
        except requests.exceptions.HTTPError as e:
            log.error("Error: " + str(e))
        return res

    # convert response to json format.
    def __extract_response(self, response, res):
        raw_result = response.text
        for rec in raw_result.split("\n"):
            if rec.strip():
                res.append(json.loads(rec))

    # initialize proxy arguments.
    def __get_proxy_args(self, http_proxy, https_proxy):
        proxy_args = {}
        if http_proxy:
            proxy_args["http"] = http_proxy
        if https_proxy:
            proxy_args["https"] = https_proxy
        return proxy_args

    # initialize query parameters
    def __get_params(self, before, after):
        params = {}
        if self.limit:
            params["limit"] = self.limit
        if before and after:
            params["time_first_after"] = after
            params["time_last_before"] = before
        else:
            if before:
                params["time_first_before"] = before
            if after:
                params["time_last_after"] = after
        return params

    def __quote(self, path):
        return urllib.parse.quote(path, safe="")
