# use `pip install python-whois`
import whois
import logging
from datetime import datetime

log = logging.getLogger("WhoIsLookupClient")


class WhoIsLookupClient(object):

    str_arr_keys = ["domain_name", "name_servers", "status", "emails", "dnssec"]
    datetime_arr_keys = ["creation_date", "updated_date", "expiration_date"]

    def __init__(self, sep=",", time_format="%m-%d-%Y %H:%M:%S"):
        self.sep = sep
        self.time_format = time_format

    def whois(self, domains, arr2str=True):
        result = []
        for domain in domains:
            resp = self.__whois(domain)
            if arr2str:
                resp = self.flatten_str_array(resp)
                resp = self.flatten_datetime_array(resp)
            result.append(resp)
        return result

    def __whois(self, domain):
        response = whois.whois(domain)
        return response

    def flatten_str_array(self, resp):
        for key in self.str_arr_keys:
            if key in resp.keys() and isinstance(resp[key], list):
                resp[key] = self.sep.join(resp[key])
        return resp

    def flatten_datetime_array(self, resp):
        for key in self.datetime_arr_keys:
            values = []
            if key in resp.keys():
                if isinstance(resp[key], list):
                    for elm in resp[key]:
                        values.append(elm.strftime(self.time_format))
                    resp[key] = self.sep.join(values)
                else:
                    resp[key] = resp[key].strftime(self.time_format)
        return resp
