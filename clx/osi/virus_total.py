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

import os
import json
import requests
from os.path import abspath, basename


class VirusTotalClient(object):
    def __init__(self, api_key=None, proxies=None):
        if api_key is None:
            raise ValueError("Virus Total API key is None.")
        self.__api_key = api_key
        self.__proxies = proxies
        self.__vt_endpoint_dict = self.__create_vt_endpoint_dict()

    @property
    def api_key(self):
        return self.__api_key

    @property
    def proxies(self):
        return self.__proxies

    @property
    def vt_endpoint_dict(self):
        return self.__vt_endpoint_dict

    
    def file_scan(self, file):
        """
        This function allows you to send a file for scanning with VirusTotal. 
        Before performing submissions it would be nice to retrieve the latest report on the file.
        File size limit is 32MB, in order to submit files up to 200MB in size it is mandatory to request a special upload URL 
        using the /file/scan/upload_url endpoint.
        """
        file_size_mb = self.get_file_size(file)
        params = {"apikey": self.api_key}
        files = {"file": (basename(file), open(abspath(file), "rb"))}
        url = self.vt_endpoint_dict["file_scan"]
        if file_size_mb > 32:
            resp = self.scan_big_file(files)
        else:
            resp = self.post(url, params=params, files=files, proxies=self.proxies)
        return resp

    def get_file_size(self, file):
        statinfo = os.stat(file)
        return statinfo.st_size / (1024 * 1024)

    
    def file_rescan(self, *resource):
        """
        This function rescan given files.
        The resource argument can be the MD5, SHA-1 or SHA-256 of the file you want to re-scan.
        """
        params = {"apikey": self.api_key, "resource": ",".join(*resource)}
        resp = self.post(
            self.vt_endpoint_dict["file_rescan"], params=params, proxies=self.proxies
        )
        return resp

    def file_report(self, *resource):
        """
        The resource argument can be the MD5, SHA-1 or SHA-256 of a file for which you want to retrieve 
        the most recent antivirus report. You may also specify a scan_id returned by the /file/scan endpoint.
        """
        params = {"apikey": self.api_key, "resource": ",".join(*resource)}
        resp = self.get(
            self.vt_endpoint_dict["file_report"], params=params, proxies=self.proxies
        )
        return resp

    def url_scan(self, *url):
        """
        This function scan on provided url with VirusTotal.
        """
        params = {"apikey": self.api_key, "url": "\n".join(*url)}
        resp = self.post(
            self.vt_endpoint_dict["url_scan"], params=params, proxies=self.proxies
        )
        return resp

    def url_report(self, *resource):
        """
        The resource argument must be the URL to retrieve the most recent report.
        """
        params = {"apikey": self.api_key, "resource": "\n".join(*resource)}
        resp = self.post(
            self.vt_endpoint_dict["url_report"], params=params, proxies=self.proxies
        )
        return resp

    def ipaddress_report(self, ip):
        """
        Retrieve report using ip address.
        """
        params = {"apikey": self.api_key, "ip": ip}
        resp = self.get(
            self.vt_endpoint_dict["ip_report"], params=params, proxies=self.proxies
        )
        return resp

    def domain_report(self, domain):
        """
        Retrieve report using domain.
        """
        params = {"apikey": self.api_key, "domain": domain}
        resp = self.get(
            self.vt_endpoint_dict["domain_report"], params=params, proxies=self.proxies
        )
        return resp

    def put_comment(self, resource, comment):
        """
        Post comment for a file or URL
        """
        params = {"apikey": self.api_key, "resource": resource, "comment": comment}
        resp = self.post(
            self.vt_endpoint_dict["put_comment"], params=params, proxies=self.proxies
        )
        return resp

    def scan_big_file(self, files):
        """
        Scanning files larger than 32MB
        """
        params = {"apikey": self.api_key}
        upload_url_json = self.get(self.vt_endpoint_dict["upload_url"], params=params)
        upload_url = upload_url_json["upload_url"]
        resp = post(upload_url, files=files)
        return self.validate_response(resp)

    def post(self, endpoint, params, **kwargs):
        resp = requests.post(endpoint, params=params, **kwargs)
        return self.validate_response(resp)

    def get(self, endpoint, params, **kwargs):
        resp = requests.get(endpoint, params=params, **kwargs)
        return self.validate_response(resp)

    def __create_vt_endpoint_dict(self):
        vt_endpoint_dict = {}
        base_url = "https://www.virustotal.com/vtapi/v2"
        vt_endpoint_dict["file_scan"] = "%s/file/scan" % (base_url)
        vt_endpoint_dict["file_rescan"] = "%s/file/rescan" % (base_url)
        vt_endpoint_dict["file_report"] = "%s/file/report" % (base_url)
        vt_endpoint_dict["url_scan"] = "%s/url/scan" % (base_url)
        vt_endpoint_dict["url_report"] = "%s/url/report" % (base_url)
        vt_endpoint_dict["upload_url"] = "%s/file/scan/upload_url" % (base_url)
        vt_endpoint_dict["ip_report"] = "%s/ip-address/report" % (base_url)
        vt_endpoint_dict["domain_report"] = "%s/domain/report" % (base_url)
        vt_endpoint_dict["put_comment"] = "%s/comments/put" % (base_url)
        return vt_endpoint_dict

    def validate_response(self, response):
        if response.status_code == 200:
            json_resp = json.loads(response.text)
            return dict(status_code=response.status_code, json_resp=json_resp)
        return dict(
            status_code=response.status_code, error=response.text, resp=response.content
        )
