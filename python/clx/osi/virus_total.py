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

import os
import json
import requests
from os.path import abspath, basename

# ref https://developers.virustotal.com/reference


class VirusTotalClient(object):
    """
    Wrapper class to query VirusTotal database.

    :param apikey: API key
    :param proxies: proxies
    """
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
        File size limit is 32MB, in order to submit files up to 200MB in size it is mandatory to use `scan_big_file` feature

        :param file: File to be scanned
        :type file: str
        :return: Response
        :rtype: dict

        Examples
        --------
        >>> from clx.osi.virus_total import VirusTotalClient
        >>> client = VirusTotalClient(api_key='your-api-key')
        >>> client.file_scan('test.sh')
        {'status_code': 200, 'json_resp': {'scan_id': '0204e88255a0bd7807547e9186621f0478a6bb2c43e795fb5e6934e5cda0e1f6-1605914572', 'sha1': '70c0942965354dbb132c05458866b96709e37f44'...}
        """
        file_size_mb = self.__get_file_size(file)
        params = {"apikey": self.api_key}
        files = {"file": (basename(file), open(abspath(file), "rb"))}
        url = self.vt_endpoint_dict["file_scan"]
        if file_size_mb > 32:
            resp = self.scan_big_file(files)
        else:
            resp = self.__post(url, params=params, files=files, proxies=self.proxies)
        return resp

    def __get_file_size(self, file):
        statinfo = os.stat(file)
        return statinfo.st_size / (1024 * 1024)

    def file_rescan(self, *resource):
        """
        This function rescan given files.

        :param *resource: The resource argument can be the MD5, SHA-1 or SHA-256 of the file you want to re-scan.
        :type *resource: str
        :return: Response
        :rtype: dict

        Examples
        --------
        >>> from clx.osi.virus_total import VirusTotalClient
        >>> client = VirusTotalClient(api_key='your-api-key')
        >>> client.file_rescan('70c0942965354dbb132c05458866b96709e37f44')
        {'status_code': 200, 'json_resp': {'scan_id': ...}}
        """
        params = {"apikey": self.api_key, "resource": ",".join(*resource)}
        resp = self.__post(
            self.vt_endpoint_dict["file_rescan"], params=params, proxies=self.proxies
        )
        return resp

    def file_report(self, *resource):
        """
        Retrieve file scan reports

        :param *resource: The resource argument can be the MD5, SHA-1 or SHA-256 of a file for which you want to retrieve
        the most recent antivirus report. You may also specify a scan_id returned by the /file/scan endpoint.
        :type *resource: str
        :return: Response
        :rtype: dict

        Examples
        --------
        >>> from clx.osi.virus_total import VirusTotalClient
        >>> client = VirusTotalClient(api_key='your-api-key')
        >>> client.file_report(["99017f6eebbac24f351415dd410d522d"])
        {'status_code': 200, 'json_resp': {'scans': {'Bkav': {'detected': True, 'version': '1.3.0.9899', 'result': 'W32.AIDetectVM.malware1'...}}
        """
        params = {"apikey": self.api_key, "resource": ",".join(*resource)}
        resp = self.__get(
            self.vt_endpoint_dict["file_report"], params=params, proxies=self.proxies
        )
        return resp

    def url_scan(self, *url):
        """Retrieve URL scan reports

        :param *url: A URL for which you want to retrieve the most recent report. You may also specify a scan_id (sha256-timestamp as returned by the URL submission API) to access a specific report.
        :type *url: str
        :return: Response
        :rtype: dict

        Examples
        --------
        >>> from clx.osi.virus_total import VirusTotalClient
        >>> client = VirusTotalClient(api_key='your-api-key')
        >>> client.url_scan(["virustotal.com"])
        {'status_code': 200, 'json_resp': {'permalink': 'https://www.virustotal.com/gui/url/...}}
        """
        params = {"apikey": self.api_key, "url": "\n".join(*url)}
        resp = self.__post(
            self.vt_endpoint_dict["url_scan"], params=params, proxies=self.proxies
        )
        return resp

    def url_report(self, *resource):
        """
        Retrieve URL scan reports

        :param *resource: The resource argument must be the URL to retrieve the most recent report.
        :type *resource: str
        :return: Response
        :rtype: dict

        Examples
        --------
        >>> from clx.osi.virus_total import VirusTotalClient
        >>> client = VirusTotalClient(api_key='your-api-key')
        >>> client.url_report(["virustotal.com"])
        {'status_code': 200, 'json_resp': {'scan_id': 'a354494a73382ea0b4bc47f4c9e8d6c578027cd4598196dc88f05a22b5817293-1605914280'...}
        """
        params = {"apikey": self.api_key, "resource": "\n".join(*resource)}
        resp = self.__post(
            self.vt_endpoint_dict["url_report"], params=params, proxies=self.proxies
        )
        return resp

    def ipaddress_report(self, ip):
        """
        Retrieve report using ip address.

        :param ip: An IP address
        :type ip: str
        :return: Response
        :rtype: dict

        Examples
        --------
        >>> from clx.osi.virus_total import VirusTotalClient
        >>> client = VirusTotalClient(api_key='your-api-key')
        >>> client.ipaddress_report("90.156.201.27")
        {'status_code': 200, 'json_resp': {'asn': 25532, 'undetected_urls...}}
        """
        params = {"apikey": self.api_key, "ip": ip}
        resp = self.__get(
            self.vt_endpoint_dict["ip_report"], params=params, proxies=self.proxies
        )
        return resp

    def domain_report(self, domain):
        """
        Retrieve report using domain.

        :param domain: A domain name
        :type domain: str
        :return: Response
        :rtype: dict

        Examples
        --------
        >>> from clx.osi.virus_total import VirusTotalClient
        >>> client = VirusTotalClient(api_key='your-api-key')
        >>> client.domain_report("027.ru")
        {'status_code': 200, 'json_resp': {'BitDefender category': 'parked', 'undetected_downloaded_samples'...}}
        """
        params = {"apikey": self.api_key, "domain": domain}
        resp = self.__get(
            self.vt_endpoint_dict["domain_report"], params=params, proxies=self.proxies
        )
        return resp

    def put_comment(self, resource, comment):
        """
        Post comment for a file or URL

        :param resource: Either an md5/sha1/sha256 hash of the file you want to review or the URL itself that you want to comment on.
        :type resource: str
        :return: Response
        :rtype: dict

        Examples
        --------
        >>> from clx.osi.virus_total import VirusTotalClient
        >>> client = VirusTotalClient(api_key='your-api-key')
        >>> client.put_comment("75efd85cf6f8a962fe016787a7f57206ea9263086ee496fc62e3fc56734d4b53", "This is a test comment")
        {'status_code': 200, 'json_resp': {'response_code': 0, 'verbose_msg': 'Duplicate comment'}}
        """
        params = {"apikey": self.api_key, "resource": resource, "comment": comment}
        resp = self.__post(
            self.vt_endpoint_dict["put_comment"], params=params, proxies=self.proxies
        )
        return resp

    def scan_big_file(self, files):
        """
        Scanning files larger than 32MB

        :param file: File to be scanned
        :type file: str
        :return: Response
        :rtype: dict

        Examples
        --------
        >>> from clx.osi.virus_total import VirusTotalClient
        >>> client = VirusTotalClient(api_key='your-api-key')
        >>> client.scan_big_file('test.sh')
        {'status_code': 200, 'json_resp': {'scan_id': '0204e88255a0bd7807547e9186621f0478a6bb2c43e795fb5e6934e5cda0e1f6-1605914572', 'sha1': '70c0942965354dbb132c05458866b96709e37f44'...}
        """
        params = {"apikey": self.api_key}
        upload_url_json = self.__get(self.vt_endpoint_dict["upload_url"], params=params)
        upload_url = upload_url_json["upload_url"]
        resp = requests.post(upload_url, files=files)
        return self.__validate_response(resp)

    def __post(self, endpoint, params, **kwargs):
        resp = requests.post(endpoint, params=params, **kwargs)
        return self.__validate_response(resp)

    def __get(self, endpoint, params, **kwargs):
        resp = requests.get(endpoint, params=params, **kwargs)
        return self.__validate_response(resp)

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

    def __validate_response(self, response):
        if response.status_code == 200:
            json_resp = json.loads(response.text)
            return dict(status_code=response.status_code, json_resp=json_resp)
        return dict(
            status_code=response.status_code, error=response.text, resp=response.content
        )
