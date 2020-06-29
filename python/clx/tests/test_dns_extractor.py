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

import pytest
from cudf import DataFrame
from clx.dns import dns_extractor as dns

input_df = DataFrame(
    {
        "url": [
            "http://www.google.com",
            "gmail.com",
            "github.com",
            "https://pandas.pydata.org",
            "http://www.worldbank.org.kg/",
            "waiterrant.blogspot.com",
            "http://forums.news.cnn.com.ac/",
            "http://forums.news.cnn.ac/",
            "ftp://b.cnn.com/",
            "a.news.uk",
            "a.news.co.uk",
            "https://a.news.co.uk",
            "107-193-100-2.lightspeed.cicril.sbcglobal.net",
            "a23-44-13-2.deploy.static.akamaitechnologies.com",
        ]
    }
)


def test_dns_vars_provider():
    sv = dns.DnsVarsProvider.get_instance()
    sv2 = dns.DnsVarsProvider.get_instance()
    assert sv is sv2


def test2_dns_vars_provider():
    expected_error = Exception("This is a singleton class")

    with pytest.raises(Exception) as actual_error:
        dns.DnsVarsProvider()
        assert actual_error == expected_error


@pytest.mark.parametrize("input_df", [input_df])
def test_parse_url(input_df):
    expected_output_df = DataFrame(
        {
            "domain": [
                "google",
                "gmail",
                "github",
                "pydata",
                "worldbank",
                "waiterrant",
                "cnn",
                "cnn",
                "cnn",
                "news",
                "news",
                "news",
                "sbcglobal",
                "akamaitechnologies",
            ],
            "suffix": [
                "com",
                "com",
                "com",
                "org",
                "org.kg",
                "blogspot.com",
                "com.ac",
                "ac",
                "com",
                "uk",
                "co.uk",
                "co.uk",
                "net",
                "com",
            ],
        }
    )
    output_df = dns.parse_url(input_df["url"], req_cols={"domain", "suffix"})

    assert expected_output_df.equals(output_df)


@pytest.mark.parametrize("input_df", [input_df])
def test2_parse_url(input_df):
    expected_output_df = DataFrame(
        {
            "hostname": [
                "www.google.com",
                "gmail.com",
                "github.com",
                "pandas.pydata.org",
                "www.worldbank.org.kg",
                "waiterrant.blogspot.com",
                "forums.news.cnn.com.ac",
                "forums.news.cnn.ac",
                "b.cnn.com",
                "a.news.uk",
                "a.news.co.uk",
                "a.news.co.uk",
                "107-193-100-2.lightspeed.cicril.sbcglobal.net",
                "a23-44-13-2.deploy.static.akamaitechnologies.com",
            ],
            "subdomain": [
                "www",
                "",
                "",
                "pandas",
                "www",
                "",
                "forums.news",
                "forums.news",
                "b",
                "a",
                "a",
                "a",
                "107-193-100-2.lightspeed.cicril",
                "a23-44-13-2.deploy.static",
            ],
            "domain": [
                "google",
                "gmail",
                "github",
                "pydata",
                "worldbank",
                "waiterrant",
                "cnn",
                "cnn",
                "cnn",
                "news",
                "news",
                "news",
                "sbcglobal",
                "akamaitechnologies",
            ],
            "suffix": [
                "com",
                "com",
                "com",
                "org",
                "org.kg",
                "blogspot.com",
                "com.ac",
                "ac",
                "com",
                "uk",
                "co.uk",
                "co.uk",
                "net",
                "com",
            ],
        }
    )
    output_df = dns.parse_url(input_df["url"])

    assert expected_output_df.equals(output_df)


@pytest.mark.parametrize("input_df", [input_df])
def test_extract_hostname(input_df):
    expected_output_df = DataFrame(
        {
            "hostname": [
                "www.google.com",
                "gmail.com",
                "github.com",
                "pandas.pydata.org",
                "www.worldbank.org.kg",
                "waiterrant.blogspot.com",
                "forums.news.cnn.com.ac",
                "forums.news.cnn.ac",
                "b.cnn.com",
                "a.news.uk",
                "a.news.co.uk",
                "a.news.co.uk",
                "107-193-100-2.lightspeed.cicril.sbcglobal.net",
                "a23-44-13-2.deploy.static.akamaitechnologies.com",
            ]
        }
    )
    output = dns.extract_hostnames(input_df["url"])
    assert output.equals(expected_output_df["hostname"])


def test_generate_tld_cols():
    hostnames_df = DataFrame(
        {"hostname": ["forums.news.cnn.com.ac", "forums.news.cnn.ac", "b.cnn.com"]}
    )
    input_df = DataFrame(
        {
            4: ["ac", "", ""],
            3: ["com", "ac", ""],
            2: ["cnn", "cnn", "com"],
            1: ["news", "news", "cnn"],
            0: ["forums", "forums", "b"],
        }
    )
    expected_output_df = DataFrame(
        {
            4: ["ac", "", ""],
            3: ["com", "ac", ""],
            2: ["cnn", "cnn", "com"],
            1: ["news", "news", "cnn"],
            0: ["forums", "forums", "b"],
            "tld4": ["ac", "", ""],
            "tld3": ["com.ac", "ac", ""],
            "tld2": ["cnn.com.ac", "cnn.ac", "com"],
            "tld1": ["news.cnn.com.ac", "news.cnn.ac", "cnn.com"],
            "tld0": ["forums.news.cnn.com.ac", "forums.news.cnn.ac", "b.cnn.com"],
        }
    )
    col_len = len(input_df.columns) - 1
    actual_output_df = dns.generate_tld_cols(
        input_df, hostnames_df["hostname"], col_len
    )
    assert expected_output_df.equals(actual_output_df)


@pytest.mark.parametrize("input_df", [input_df])
def test_parse_url_invalid_req_cols(input_df):
    expected_error = ValueError(
        "Given req_cols must be subset of %s"
        % ('["hostname", "subdomain", "domain", "suffix"]')
    )
    with pytest.raises(ValueError) as actual_error:
        dns.parse_url(input_df["url"], req_cols={"test"})
        assert actual_error == expected_error
