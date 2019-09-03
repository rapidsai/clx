import os
import cudf
import pytest
from rapidscyber.dns.dns_extractor import DNSExtractor

test_suffix_list_path = "%s/input/suffix_list.txt" % os.path.dirname(os.path.realpath(__file__))
# Read suffix list csv file
suffix_df = cudf.io.csv.read_csv(
    test_suffix_list_path, names=["suffix"], header=None, dtype=["str"]
)
# Filter commented lines from suffix.
suffix_df = suffix_df[suffix_df["suffix"].str.contains("^[^//]+$")]

input_df = cudf.DataFrame(
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
            ],
        )
    ]
)


@pytest.mark.parametrize("suffix_df", [suffix_df])
@pytest.mark.parametrize("input_df", [input_df])
def test_parse_url(suffix_df, input_df):
    expected_output_df = cudf.DataFrame(
        [
            (
                "domain",
                [
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
    dns_extractor = DNSExtractor(suffix_df)
    output_df = dns_extractor.parse_url(input_df["url"], req_cols=["domain", "suffix"])
    assert output_df.equals(expected_output_df)


@pytest.mark.parametrize("suffix_df", [suffix_df])
@pytest.mark.parametrize("input_df", [input_df])
def test2_parse_url(suffix_df, input_df):
    expected_output_df = cudf.DataFrame(
        [
            (
                "hostname",
                [
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
    dns_extractor = DNSExtractor(suffix_df)
    output_df = dns_extractor.parse_url(input_df["url"])
    assert output_df.equals(expected_output_df)


@pytest.mark.parametrize("input_df", [input_df])
def test_extract_hostname(input_df):
    expected_output_df = cudf.DataFrame(
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
                ],
            )
        ]
    )
    output = DNSExtractor.extract_hostnames(input_df["url"])
    assert output.equals(expected_output_df["hostname"])
