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

# ref: https://github.com/slashnext/SlashNext-URL-Analysis-and-Enrichment/tree/master/Python%20SDK
import os
from SlashNextPhishingIR import SlashNextPhishingIR


class SlashNextClient(object):
    def __init__(
        self, api_key, snx_ir_workspace, base_url="https://oti.slashnext.cloud/api"
    ):
        if api_key is None:
            raise ValueError("SlashNext API key is None")
        if snx_ir_workspace is not None:
            if not os.path.exists(snx_ir_workspace):
                try:
                    print("Creating directory {}".format(snx_ir_workspace))
                    os.makedirs(snx_ir_workspace)
                except Exception as error:
                    raise Exception("Error while creating workspace: " + repr(error))
        self._snx_phishing_ir = SlashNextPhishingIR(snx_ir_workspace)
        self._snx_phishing_ir.set_conf(api_key=api_key, base_url=base_url)

    @property
    def conn(self):
        return self._snx_phishing_ir

    def verify_connection(self):
        """
        Verify SlashNext cloud database connection.

        Examples
        --------
        >>> from clx.osi.slashnext import SlashNextClient
        >>> api_key = 'slashnext_cloud_apikey'
        >>> snx_ir_workspace_dir = 'snx_ir_workspace'
        >>> slashnext = SlashNextClient(api_key, snx_ir_workspace_dir)
        >>> slashnext.verify_connection()
        Successfully connected to SlashNext cloud.
        'success'
        """
        status, details = self.conn.test()
        if status == "ok":
            print("Successfully connected to SlashNext cloud.")
            return "success"
        else:
            raise Exception(
                "Connection to SlashNext cloud failed due to {}.".format(details)
            )

    def _execute(self, command):
        """
        Execute all SlashNext Phishing Incident Response SDK supported actions/commands.

        :param command: Query to execute on SlashNext cloud database.
        :type command: str
        :return Query response as list.
        :rtype: list
        """
        status, details, responses_list = self.conn.execute(command)
        if status == "ok":
            return responses_list
        else:
            raise Exception(
                "Action '{}' execution failed due to {}.".format(command, details)
            )

    def host_reputation(self, host):
        """
        Queries the SlashNext cloud database and retrieves the reputation of a host.

        :param host: The host to look up in the SlashNext Threat Intelligence database. Can be either a domain name or an IPv4 address.
        :type host: str
        :return Query response as list.
        :rtype: list

        Examples
        --------
        >>> from clx.osi.slashnext import SlashNextClient
        >>> api_key = 'slashnext_cloud_apikey'
        >>> snx_ir_workspace_dir = 'snx_ir_workspace'
        >>> slashnext = SlashNextClient(api_key, snx_ir_workspace_dir)
        >>> response_list = slashnext.host_reputation('google.com')
        >>> type(response_list[0])
        <class 'dict'>
        """
        command = "slashnext-host-reputation host={}".format(host)
        try:
            return self._execute(command)
        except Exception as error:
            raise Exception("SlashNext Host Reputation: " + repr(error))

    def host_report(self, host):
        """
        Queries the SlashNext cloud database and retrieves a detailed report.

        :param host: The host to look up in the SlashNext Threat Intelligence database. Can be either a domain name or an IPv4 address.
        :type host: str
        :return Query response as list.
        :rtype: list

        Examples
        --------
        >>> from clx.osi.slashnext import SlashNextClient
        >>> api_key = 'slashnext_cloud_apikey'
        >>> snx_ir_workspace_dir = 'snx_ir_workspace'
        >>> slashnext = SlashNextClient(api_key, snx_ir_workspace_dir)
        >>> response_list = slashnext.host_report('google.com')
        >>> type(response_list[0])
        <class 'dict'>
        """
        command = "slashnext-host-report host={}".format(host)
        try:
            return self._execute(command)
        except Exception as error:
            raise Exception("SlashNext Host Report: " + repr(error))

    def host_urls(self, host, limit=10):
        """
        Queries the SlashNext cloud database and retrieves a list of all URLs.

        :param host: The host to look up in the SlashNext Threat Intelligence database, for which to return a list of associated URLs. Can be either a domain name or an IPv4 address.
        :type host: str
        :param limit: The maximum number of URL records to fetch. Default is "10".
        :type limit: int
        :return Query response as list.
        :rtype: list

        Examples
        --------
        >>> from clx.osi.slashnext import SlashNextClient
        >>> api_key = 'slashnext_cloud_apikey'
        >>> snx_ir_workspace_dir = 'snx_ir_workspace'
        >>> slashnext = SlashNextClient(api_key, snx_ir_workspace_dir)
        >>> response_list = slashnext.host_urls('google.com', limit=1)
        >>> type(response_list[0])
        <class 'dict'>
        """
        command = "slashnext-host-urls host={} limit={}".format(host, limit)
        try:
            return self._execute(command)
        except Exception as error:
            raise Exception("SlashNext Host urls: " + repr(error))

    def url_scan(self, url, extended_info=True):
        """
        Perform a real-time URL reputation scan with SlashNext cloud-based SEER threat detection engine.

        :param url: The URL that needs to be scanned.
        :type url: str
        :param extended_info: Whether to download forensics data, such as screenshot, HTML, and rendered text.
        :type extended_info: boolean
        :return Query response as list.
        :rtype: list

        Examples
        --------
        >>> from clx.osi.slashnext import SlashNextClient
        >>> api_key = 'slashnext_cloud_apikey'
        >>> snx_ir_workspace_dir = 'snx_ir_workspace'
        >>> slashnext = SlashNextClient(api_key, snx_ir_workspace_dir)
        >>> response_list = slashnext.url_scan('http://ajeetenterprises.in/js/kbrad/drive/index.php', extended_info=False)
        >>> type(response_list[0])
        <class 'dict'>
        """
        command = "slashnext-url-scan url={} extended_info={}".format(
            url, str(extended_info).lower()
        )
        try:
            return self._execute(command)
        except Exception as error:
            raise Exception("SlashNext URL Scan: " + repr(error))

    def url_scan_sync(self, url, extended_info=True, timeout=60):
        """
        Perform a real-time URL scan with SlashNext cloud-based SEER threat detection engine in a blocking mode.

        :param url: The URL that needs to be scanned.
        :type url: str
        :param extended_info: Whether to download forensics data, such as screenshot, HTML, and rendered text.
        :type extended_info: boolean
        :param timeout: A timeout value in seconds. If no timeout value is specified, a default timeout value is 60 seconds.
        :type timeout: int
        :return Query response as list.
        :rtype: list

        Examples
        --------
        >>> from clx.osi.slashnext import SlashNextClient
        >>> api_key = 'slashnext_cloud_apikey'
        >>> snx_ir_workspace_dir = 'snx_ir_workspace'
        >>> slashnext = SlashNextClient(api_key, snx_ir_workspace_dir)
        >>> response_list = slashnext.url_scan_sync('http://ajeetenterprises.in/js/kbrad/drive/index.php', extended_info=False, timeout=10)
        >>> type(response_list[0])
        <class 'dict'>
        """
        command = "slashnext-url-scan-sync url={} extended_info={} timeout={}".format(
            url, str(extended_info).lower(), timeout
        )
        try:
            return self._execute(command)
        except Exception as error:
            raise Exception("SlashNext URL Scan Sync: " + repr(error))

    def scan_report(self, scanid, extended_info=True):
        """
        Retrieve URL scan results against a previous scan request.

        :param scanid: Scan ID of the scan for which to get the report. Can be retrieved from the "slashnext-url-scan" action or "slashnext-url-scan-sync" action.
        :type scanid: str
        :param extended_info: Whether to download forensics data, such as screenshot, HTML, and rendered text.
        :type extended_info: boolean
        :return Query response as list.
        :rtype: list

        Examples
        --------
        >>> from clx.osi.slashnext import SlashNextClient
        >>> api_key = 'slashnext_cloud_apikey'
        >>> snx_ir_workspace_dir = 'snx_ir_workspace'
        >>> slashnext = SlashNextClient(api_key, snx_ir_workspace_dir)
        >>> response_list = slashnext.scan_report('2-ba57-755a7458c8a3', extended_info=False)
        >>> type(response_list[0])
        <class 'dict'>
        """
        command = "slashnext-scan-report scanid={} extended_info={}".format(
            scanid, str(extended_info).lower()
        )
        try:
            return self._execute(command)
        except Exception as error:
            raise Exception("SlashNext Scan Report: " + repr(error))

    def download_screenshot(self, scanid, resolution="high"):
        """
        Downloads a screenshot of a web page against a previous URL scan request.

        :param scanid: Scan ID of the scan for which to get the report. Can be retrieved from the "slashnext-url-scan" action or "slashnext-url-scan-sync" action.
        :type scanid: str
        :param resolution: Resolution of the web page screenshot. Can be "high" or "medium". Default is "high".
        :type resolution: str
        :return Query response as list.
        :rtype: list

        Examples
        --------
        >>> from clx.osi.slashnext import SlashNextClient
        >>> api_key = 'slashnext_cloud_apikey'
        >>> snx_ir_workspace_dir = 'snx_ir_workspace'
        >>> slashnext = SlashNextClient(api_key, snx_ir_workspace_dir)
        >>> response_list = slashnext.download_screenshot('2-ba57-755a7458c8a3')
        >>> type(response_list[0])
        <class 'dict'>
        """
        command = "slashnext-download-screenshot scanid={} resolution={}".format(
            scanid, resolution.lower()
        )
        try:
            return self._execute(command)
        except Exception as error:
            raise Exception("SlashNext Download Screenshot: " + repr(error))

    def download_html(self, scanid):
        """
        Downloads a web page HTML against a previous URL scan request.

        :param scanid: Scan ID of the scan for which to get the report. Can be retrieved from the "slashnext-url-scan" action or "slashnext-url-scan-sync" action.
        :type scanid: str
        :return Query response as list.
        :rtype: list

        Examples
        --------
        >>> from clx.osi.slashnext import SlashNextClient
        >>> api_key = 'slashnext_cloud_apikey'
        >>> snx_ir_workspace_dir = 'snx_ir_workspace'
        >>> slashnext = SlashNextClient(api_key, snx_ir_workspace_dir)
        >>> response_list = slashnext.download_html('2-ba57-755a7458c8a3')
        >>> type(response_list[0])
        <class 'dict'>
        """
        command = "slashnext-download-html scanid={}".format(scanid)
        try:
            return self._execute(command)
        except Exception as error:
            raise Exception("SlashNext Download HTML: " + repr(error))

    def download_text(self, scanid):
        """
        Downloads the text of a web page against a previous URL scan request.

        :param scanid: Scan ID of the scan for which to get the report. Can be retrieved from the "slashnext-url-scan" action or "slashnext-url-scan-sync" action.
        :type scanid: str
        :return Query response as list.
        :rtype: list

        Examples
        --------
        >>> from clx.osi.slashnext import SlashNextClient
        >>> api_key = 'slashnext_cloud_apikey'
        >>> snx_ir_workspace_dir = 'snx_ir_workspace'
        >>> slashnext = SlashNextClient(api_key, snx_ir_workspace_dir)
        >>> response_list = slashnext.download_text('2-ba57-755a7458c8a3')
        >>> type(response_list[0])
        <class 'dict'>
        """
        command = "slashnext-download-text scanid={}".format(scanid)
        try:
            return self._execute(command)
        except Exception as error:
            raise Exception("SlashNext Download HTML: " + repr(error))

    def api_quota(self):
        """
        Find information about your API quota, like current usage, quota left etc.

        :return Query response as list.
        :rtype: list

        Examples
        --------
        >>> from clx.osi.slashnext import SlashNextClient
        >>> api_key = 'slashnext_cloud_apikey'
        >>> snx_ir_workspace_dir = 'snx_ir_workspace'
        >>> slashnext = SlashNextClient(api_key, snx_ir_workspace_dir)
        >>> response_list = slashnext.api_quota()
        >>> type(response_list[0])
        <class 'dict'>
        """
        command = "slashnext-api-quota"
        try:
            return self._execute(command)
        except Exception as error:
            raise Exception("SlashNext API Quota: " + repr(error))
