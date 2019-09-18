import pytest
from cudf import DataFrame
from clx.dns import dns_extractor as dns

input_df = DataFrame(
    [
        (
            "url",
            [
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
            ],
        )
    ]
)


def test_dns_vars_Provider():
    sv = dns.DnsVarsProvider()
    sv2 = dns.DnsVarsProvider()
    assert sv is sv2


@pytest.mark.parametrize("input_df", [input_df])
def test_parse_url(input_df):
    expected_output_df = DataFrame(
        [
            (
                "domain",
                [
                    "sbcglobal",
                    "akamaitechnologies",
                    "cnn",
                    "cnn",
                    "google",
                    "pydata",
                    "worldbank",
                    "cnn",
                    "news",
                    "news",
                    "news",
                    "gmail",
                    "github",
                    "waiterrant",
                ],
            ),
            (
                "suffix",
                [
                    "net",
                    "com",
                    "com.ac",
                    "ac",
                    "com",
                    "org",
                    "org.kg",
                    "com",
                    "uk",
                    "co.uk",
                    "co.uk",
                    "com",
                    "com",
                    "blogspot.com",
                ],
            ),
        ]
    )
    output_df = dns.parse_url(input_df["url"], req_cols={"domain", "suffix"})
    assert output_df.equals(expected_output_df)


@pytest.mark.parametrize("input_df", [input_df])
def test2_parse_url(input_df):
    expected_output_df = DataFrame(
        [
            (
                "hostname",
                [
                    "107-193-100-2.lightspeed.cicril.sbcglobal.net",
                    "a23-44-13-2.deploy.static.akamaitechnologies.com",
                    "forums.news.cnn.com.ac",
                    "forums.news.cnn.ac",
                    "www.google.com",
                    "pandas.pydata.org",
                    "www.worldbank.org.kg",
                    "b.cnn.com",
                    "a.news.uk",
                    "a.news.co.uk",
                    "a.news.co.uk",
                    "gmail.com",
                    "github.com",
                    "waiterrant.blogspot.com",
                ],
            ),
            (
                "subdomain",
                [
                    "107-193-100-2.lightspeed.cicril",
                    "a23-44-13-2.deploy.static",
                    "forums.news",
                    "forums.news",
                    "www",
                    "pandas",
                    "www",
                    "b",
                    "a",
                    "a",
                    "a",
                    "",
                    "",
                    "",
                ],
            ),
            (
                "domain",
                [
                    "sbcglobal",
                    "akamaitechnologies",
                    "cnn",
                    "cnn",
                    "google",
                    "pydata",
                    "worldbank",
                    "cnn",
                    "news",
                    "news",
                    "news",
                    "gmail",
                    "github",
                    "waiterrant",
                ],
            ),
            (
                "suffix",
                [
                    "net",
                    "com",
                    "com.ac",
                    "ac",
                    "com",
                    "org",
                    "org.kg",
                    "com",
                    "uk",
                    "co.uk",
                    "co.uk",
                    "com",
                    "com",
                    "blogspot.com",
                ],
            ),
        ]
    )
    output_df = dns.parse_url(input_df["url"])
    assert output_df.equals(expected_output_df)


@pytest.mark.parametrize("input_df", [input_df])
def test_extract_hostname(input_df):
    expected_output_df = DataFrame(
        [
            (
                "hostname",
                [
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
            )
        ]
    )
    output = dns.extract_hostnames(input_df["url"])
    assert output.equals(expected_output_df["hostname"])


def test_get_hostname_split_df():
    input_df = DataFrame(
        [("hostname", ["forums.news.cnn.com.ac", "forums.news.cnn.ac", "b.cnn.com"])]
    )

    expected_output_df = DataFrame(
        [
            (4, ["ac", "", ""]),
            (3, ["com", "ac", ""]),
            (2, ["cnn", "cnn", "com"]),
            (1, ["news", "news", "cnn"]),
            (0, ["forums", "forums", "b"]),
        ]
    )
    actual_output_df = dns.get_hostname_split_df(input_df["hostname"])
    assert actual_output_df.equals(expected_output_df)


def test_generate_tld_cols():
    hostnames_df = DataFrame(
        [("hostname", ["forums.news.cnn.com.ac", "forums.news.cnn.ac", "b.cnn.com"])]
    )
    input_df = DataFrame(
        [
            (4, ["ac", "", ""]),
            (3, ["com", "ac", ""]),
            (2, ["cnn", "cnn", "com"]),
            (1, ["news", "news", "cnn"]),
            (0, ["forums", "forums", "b"]),
        ]
    )
    expected_output_df = DataFrame(
        [
            (4, ["ac", "", ""]),
            (3, ["com", "ac", ""]),
            (2, ["cnn", "cnn", "com"]),
            (1, ["news", "news", "cnn"]),
            (0, ["forums", "forums", "b"]),
            ("tld4", ["ac", "", ""]),
            ("tld3", ["com.ac", "ac", ""]),
            ("tld2", ["cnn.com.ac", "cnn.ac", "com"]),
            ("tld1", ["news.cnn.com.ac", "news.cnn.ac", "cnn.com"]),
            ("tld0", ["forums.news.cnn.com.ac", "forums.news.cnn.ac", "b.cnn.com"]),
        ]
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
        % (["hostname", "subdomain", "domain", "suffix"])
    )
    with pytest.raises(ValueError) as actual_error:
        output_df = dns.parse_url(input_df["url"], req_cols={"test"})
        assert actual_error == expected_error
