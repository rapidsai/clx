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

import json
import requests
import urllib
import logging

log = logging.getLogger("FarsightLookupClient")

"""
This class provides functionality to query DNSDB record in various ways
Example: by IP, DomainName
"""


class FarsightLookupClient(object):
    def __init__(self, server, apikey, limit=None, http_proxy=None, https_proxy=None):
        self.server = server
        self.headers = {"Accept": "application/json", "X-Api-Key": apikey}
        self.limit = limit
        self.proxy_args = self.__get_proxy_args(http_proxy, https_proxy)

    # batch version of querying DNSDB by given domain name and time ranges.
    def query_rrset(self, oname, rrtype=None, bailiwick=None, before=None, after=None):
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

    # query matches only a single DNSDB record of given oname and time ranges.
    def query_rdata_name(self, rdata_name, rrtype=None, before=None, after=None):
        quoted_name = self.__quote(rdata_name)
        if rrtype:
            path = "rdata/name/%s/%s" % (quoted_name, rrtype)
        else:
            path = "rdata/name/%s" % quoted_name
        return self.__query(path, before, after)

    # query to find DNSDB records matching a specific IP address with given time range.
    def query_rdata_ip(self, rdata_ip, before=None, after=None):
        path = "rdata/ip/%s" % rdata_ip.replace("/", ",")
        return self.__query(path, before, after)

    def get(self, url):
        response = requests.get(url, headers=self.headers, proxies=self.proxy_args)
        return response

    # queries dnsdb.
    def __query(self, path, before=None, after=None):
        res = []
        url = "%s/lookup/%s" % (self.server, path)
        params = self.__get_params(before, after)
        if params:
            url += "?{0}".format(urllib.parse.urlencode(params))
        response = self.get(url)
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
