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
import cudf
import logging
from cudf import DataFrame

log = logging.getLogger(__name__)


class DnsVarsProvider:

    __instance = None

    @staticmethod
    def get_instance():
        if DnsVarsProvider.__instance == None:
            DnsVarsProvider()
        return DnsVarsProvider.__instance

    def __init__(self):
        if DnsVarsProvider.__instance != None:
            raise Exception("This is a singleton class")
        else:
            DnsVarsProvider.__instance = self
            DnsVarsProvider.__instance.__suffix_df = self.__load_suffix_df()
            DnsVarsProvider.__instance.__allowed_output_cols = {
                "hostname",
                "subdomain",
                "domain",
                "suffix",
            }

    @property
    def suffix_df(self):
        return self.__suffix_df

    @property
    def allowed_output_cols(self):
        return self.__allowed_output_cols

    def __load_suffix_df(self):
        suffix_list_path = "%s/resources/suffix_list.txt" % os.path.dirname(
            os.path.realpath(__file__)
        )
        log.info("Read suffix data at location %s." % (suffix_list_path))
        # Read suffix list csv file
        suffix_df = cudf.io.csv.read_csv(
            suffix_list_path, names=["suffix"], header=None, dtype=["str"]
        )
        log.info("Read suffix data is finished")
        suffix_df = suffix_df[suffix_df["suffix"].str.contains("^[^//]+$")]
        return suffix_df


def extract_hostnames(url_df_col):
    """This function extracts hostnames from the given urls.
    
    :param url_df_col: Urls that are to be handled.
    :type url_df_col: cudf.Series
    :return: Hostnames extracted from the urls.
    :rtype: cudf.Series
    
    Examples
    --------
    >>> from cudf import DataFrame
    >>> from clx.dns import dns_extractor as dns
    >>> input_df = DataFrame(
    ...     {
    ...         "url": [
    ...             "http://www.google.com",
    ...             "gmail.com",
    ...             "github.com",
    ...             "https://pandas.pydata.org",
    ...         ]
    ...     }
    ... )
    >>> dns.extract_hostnames(input_df["url"])
    0       www.google.com
    1            gmail.com
    2           github.com
    3    pandas.pydata.org
    Name: 0, dtype: object
    """

    hostnames = url_df_col.str.extract("([\\w]+[\\.].+*[^/]|[\\-\\w]+[\\.].+*[^/])")[
        0
    ].str.extract("([\\w\\.\\-]+)")[0]
    return hostnames


def get_hostname_split_df(hostnames):
    """Find all words and digits between periods.
    
    :param hostnames: Hostnames that are being split.
    :type hostnames: cudf.Series
    :return: Hostname splits.
    :rtype: cudf.DataFrame

    Examples
    --------
    >>> import cudf
    >>> from clx.dns import dns_extractor as dns
    >>> hostnames = cudf.Series(["www.google.com", "pandas.pydata.org"])
    >>> dns.get_hostname_split_df(hostnames)
         2       1       0
    0  com  google     www
    1  org  pydata  pandas
    """
    hostname_split = hostnames.str.findall("([^.]+)")
    hostname_split_df = DataFrame()
    # Assign hostname split to cudf dataframe.
    for i in range(len(hostname_split) - 1, -1, -1):
        hostname_split_df[i] = hostname_split[i]
    return hostname_split_df


def generate_tld_cols(hostname_split_df, hostnames, col_len):
    """
    This function generates tld columns.

    :param hostname_split_df: Hostname splits.
    :type hostname_split_df: cudf.DataFrame
    :param hostnames: Hostnames.
    :type hostnames: cudf.DataFrame
    :param col_len: Hostname splits dataframe columns length.
    :return: Tld columns with all combination.
    :rtype: cudf.DataFrame

    Examples
    --------
    >>> import cudf
    >>> from clx.dns import dns_extractor as dns
    >>> hostnames = cudf.Series(["www.google.com", "pandas.pydata.org"])
    >>> hostname_splits = dns.get_hostname_split_df(hostnames)
    >>> print(hostname_splits)
         2       1       0
    0  com  google     www
    1  org  pydata  pandas
    >>> col_len = len(hostname_split_df.columns) - 1
    >>> col_len = len(hostname_splits.columns) - 1
    >>> dns.generate_tld_cols(hostname_splits, hostnames, col_len)
         2       1       0 tld2        tld1               tld0
    0  com  google     www  com  google.com     www.google.com
    1  org  pydata  pandas  org  pydata.org  pandas.pydata.org
    """
    hostname_split_df = hostname_split_df.fillna("")
    hostname_split_df["tld" + str(col_len)] = hostname_split_df[col_len]
    # Add all other elements of hostname_split_df
    for j in range(col_len - 1, 0, -1):
        hostname_split_df["tld" + str(j)] = (
            hostname_split_df[j]
            .str.cat(hostname_split_df["tld" + str(j + 1)], sep=".")
            .str.rstrip(".")
        )
    # Assign hostname to tld0, to handle received input is just domain name.
    hostname_split_df["tld0"] = hostnames
    return hostname_split_df


def _extract_tld(input_df, suffix_df, col_len, col_dict, output_df):
    """
    Examples
    -------- 
        input:
               4    3                2          1           0  tld4    tld3             tld2                 tld1                        tld0    idx
            0 ac  com              cnn       news      forums    ac  com.ac       cnn.com.ac      news.cnn.com.ac      forums.news.cnn.com.ac      0
            1     ac               cnn       news      forums            ac           cnn.ac          news.cnn.ac          forums.news.cnn.ac      1
            2                      com        cnn           b                            com              cnn.com                   b.cnn.com      2
    
        output:
                              hostname      domain        suffix       subdomain   idx
            0   forums.news.cnn.com.ac         cnn        com.ac     forums.news     0
            2       forums.news.cnn.ac         cnn            ac     forums.news     1
            1                b.cnn.com         cnn           com               b     2      
    """

    tmp_suffix_df = DataFrame()
    # Iterating over each tld column starting from tld0 until it finds a match.
    for i in range(col_len + 1):
        tld_col = "tld" + str(i)
        tmp_suffix_df[tld_col] = suffix_df["suffix"]
        # Left outer join input_df with tmp_suffix_df on tld column for each iteration.
        merged_df = input_df.merge(
            tmp_suffix_df, on=tld_col, how="left", suffixes=("", "_y")
        )
        col_pos = i - 1
        tld_r_col = "tld%s_y" % (str(col_pos))
        # Check for a right side column i.e, added to merged_df when join clause satisfies.
        if tld_r_col in merged_df.columns:
            # Retrieve records which satisfies join clause.
            joined_recs_df = merged_df[merged_df[tld_r_col].isna() == False]
            if not joined_recs_df.empty:
                temp_df = DataFrame()
                temp_df["idx"] = joined_recs_df["idx"]
                if col_dict["hostname"]:
                    temp_df["hostname"] = joined_recs_df["tld0"]
                if col_dict["domain"]:
                    temp_df["domain"] = joined_recs_df[col_pos]
                if col_dict["suffix"]:
                    temp_df["suffix"] = joined_recs_df[tld_r_col]
                if col_dict["subdomain"]:
                    temp_df["subdomain"] = ""
                    if col_pos > 0:
                        for idx in range(0, col_pos):
                            temp_df["subdomain"] = temp_df["subdomain"].str.cat(
                                joined_recs_df[idx], sep="."
                            )
                        temp_df["subdomain"] = (
                            temp_df["subdomain"].str.replace(".^", "").str.lstrip(".")
                        )
                # Concat current iteration result to previous iteration result.
                output_df = cudf.concat([temp_df, output_df])
                # Assigning unprocessed records to input_df for next stage of processing.
                if i < col_len:
                    # Skip for last iteration. Since there won't be any entries to process further.
                    input_df = merged_df[merged_df[tld_r_col].isna()]
    # Release memory. Once tld_col column is no longer needed.
    tmp_suffix_df.drop(tld_col)
    input_df.drop(tld_col)
    return output_df


def _create_output_df(req_cols):
    """Create cuDF dataframe with set of predefined columns.
    """
    output_df = DataFrame([(col, "") for col in req_cols])
    # Add temp index column to preserve input index.
    output_df["idx"] = 0
    # Remove empty record i.e, added while creating dataframe.
    output_df = output_df[:0]
    return output_df


def _create_col_dict(allowed_output_cols, req_cols):
    """Creates dictionary to apply check condition while extracting tld.
    """
    col_dict = {col: True for col in allowed_output_cols}
    if req_cols != allowed_output_cols:
        for col in allowed_output_cols ^ req_cols:
            col_dict[col] = False
    return col_dict


def _verify_req_cols(req_cols, allowed_output_cols):
    """Verify user requested columns against allowed output columns.
    """
    if req_cols is not None:
        if not req_cols.issubset(allowed_output_cols):
            raise ValueError(
                "Given req_cols must be subset of %s" % (allowed_output_cols)
            )
    else:
        req_cols = allowed_output_cols
    return req_cols


def parse_url(url_df_col, req_cols=None):
    """This function extracts subdomain, domain and suffix for a given url.
    
    :param url_df_col: Urls that are to be handled.
    :type url_df_col: cudf.Series
    :param req_cols: Columns requested to extract such as (domain, subdomain, suffix and hostname).
    :type req_cols: set(strings)
    :return: Extracted information of requested columns.
    :rtype: cudf.DataFrame
    
    Examples
    --------
    >>> from cudf import DataFrame
    >>> from clx.dns import dns_extractor as dns
    >>> 
    >>> input_df = DataFrame(
    ...     {
    ...         "url": [
    ...             "http://www.google.com",
    ...             "gmail.com",
    ...             "github.com",
    ...             "https://pandas.pydata.org",
    ...         ]
    ...     }
    ... )
    >>> dns.parse_url(input_df["url"])
                hostname  domain suffix subdomain
    0     www.google.com  google    com       www
    1          gmail.com   gmail    com          
    2         github.com  github    com          
    3  pandas.pydata.org  pydata    org    pandas
    >>> dns.parse_url(input_df["url"], req_cols={'domain', 'suffix'})
       domain suffix
    0  google    com
    1   gmail    com
    2  github    com
    3  pydata    org
    """
    # Singleton object.
    sv = DnsVarsProvider.get_instance()
    req_cols = _verify_req_cols(req_cols, sv.allowed_output_cols)
    col_dict = _create_col_dict(req_cols, sv.allowed_output_cols)
    hostnames = extract_hostnames(url_df_col)
    log.info("Extracting hostnames is successfully completed.")
    hostname_split_df = get_hostname_split_df(hostnames)
    col_len = len(hostname_split_df.columns) - 1
    log.info("Generating tld columns...")
    hostname_split_df = generate_tld_cols(hostname_split_df, hostnames, col_len)
    log.info("Successfully generated tld columns.")
    output_df = _create_output_df(req_cols)
    # Assign input index to idx column.
    hostname_split_df["idx"] = url_df_col.index
    log.info("Extracting tld...")
    output_df = _extract_tld(
        hostname_split_df, sv.suffix_df, col_len, col_dict, output_df
    )
    # Sort index based on given input index order.
    output_df = output_df.sort_values("idx", ascending=True)
    # Drop temp columns.
    output_df = output_df.drop(["idx",0,1])
    # Reset the index.
    output_df = output_df.reset_index(drop=True)
    log.info("Extracting tld is successfully completed.")
    return output_df
