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

# use `pip install python-whois`
import whois
import logging
from datetime import datetime

log = logging.getLogger("WhoIsLookupClient")


class WhoIsLookupClient(object):

    str_arr_keys = ["domain_name", "name_servers", "status", "emails", "dnssec"]
    datetime_arr_keys = ["creation_date", "updated_date", "expiration_date"]

    def __init__(self, sep=",", datetime_format="%m-%d-%Y %H:%M:%S"):
        self.sep = sep
        self.datetime_format = datetime_format

    # Function to access parsed WHOIS data for a given domain
    def whois(self, domains, arr2str=True):
        result = []
        for domain in domains:
            resp = whois.whois(domain)
            if arr2str:
                resp_keys = resp.keys()
                resp = self.flatten_str_array(resp, resp_keys)
                resp = self.flatten_datetime_array(resp, resp_keys)
            result.append(resp)
        return result

    def flatten_str_array(self, resp, resp_keys):
        for key in self.str_arr_keys:
            if key in resp_keys and isinstance(resp[key], list):
                resp[key] = self.sep.join(resp[key])
        return resp

    def flatten_datetime_array(self, resp, resp_keys):
        for key in self.datetime_arr_keys:
            values = []
            if key in resp_keys:
                if isinstance(resp[key], list):
                    for ts in resp[key]:
                        values.append(ts.strftime(self.datetime_format))
                    resp[key] = self.sep.join(values)
                else:
                    resp[key] = resp[key].strftime(self.datetime_format)
        return resp
