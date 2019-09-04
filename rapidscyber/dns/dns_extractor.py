import cudf
import logging
import nvstrings
from cudf import DataFrame

log = logging.getLogger("DNSExtractor")


class DNSExtractor(object):
    def __init__(self, suffix_df):
        self.__suffix_df = suffix_df
        self.__output_cols = ["hostname", "subdomain", "domain", "suffix"]

    @staticmethod
    def extract_hostnames(url_df_col):
        """
        Extract hostname from the url.
        Example:-
            input: 
                ["http://www.worldbank.org.kg/", "waiterrant.blogspot.com","ftp://b.cnn.com/","a.news.uk"]
            output: 
                ["www.worldbank.org.kg", "waiterrant.blogspot.com","b.cnn.com", "a.news.uk"]
        """
        hostnames = url_df_col.str.extract("([\\w]+[\\.].+*[^/]")[0].str.extract(
            "([\\w\\.]+)"
        )[0]
        return hostnames

    def __get_hostname_split_df(self, hostnames):
        # Find all words and digits between periods.
        hostname_split = hostnames.str.findall("([\\w]+)")
        hostname_split_df = DataFrame()
        # Assign hostname split to cudf dataframe.
        for i in range(len(hostname_split) - 1, -1, -1):
            hostname_split_df[i] = hostname_split[i]
        # Replace null column value with empty since merge operation may use all columns.
        hostname_split_df = hostname_split_df.fillna("")
        return hostname_split_df

    def __generate_tld_cols(self, hostname_split_df, hostnames, col_len):
        """
        This function generates tld columns.
        
        Example:- 
            input:
                    4    3                2          1           0
                0  ac  com              cnn       news      forums
                1       ac              cnn       news      forums
                2                       com        cnn           b
            output:
                  4    3                2          1           0  tld4    tld3             tld2                 tld1                        tld0
               0 ac  com              cnn       news      forums    ac  com.ac       cnn.com.ac      news.cnn.com.ac      forums.news.cnn.com.ac
               1      ac              cnn       news      forums            ac           cnn.ac          news.cnn.ac          forums.news.cnn.ac
               2                      com        cnn           b                            com              cnn.com                   b.cnn.com
        
        Adding last element for max tld column.
        Example:-
            input: 
                forums.news.cnn.com.ac
            output: 
                tld4 = ac
        """
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

    def __extract_tld(self, input_df, col_len, output_df):
        """
        Example:- 
            input:
                   4    3                2          1           0  tld4    tld3             tld2                 tld1                        tld0
                0 ac  com              cnn       news      forums    ac  com.ac       cnn.com.ac      news.cnn.com.ac      forums.news.cnn.com.ac
                1     ac               cnn       news      forums            ac           cnn.ac          news.cnn.ac          forums.news.cnn.ac
                2                      com        cnn           b                            com              cnn.com                   b.cnn.com
        
            output:
                                  hostname      domain        suffix  sd_prefix  sd_suffix
                0   forums.news.cnn.com.ac         cnn        com.ac     forums       news
                1       forums.news.cnn.ac         cnn            ac     forums       news
                2                b.cnn.com         cnn           com          b           
        """
        tmp_suffix_df = DataFrame()
        # Iterating over each tld column starting from tld0 until it finds a match.
        for i in range(col_len):
            tld_col = "tld" + str(i)
            tmp_suffix_df[tld_col] = self.__suffix_df["suffix"]
            # Left outer join input_df with tmp_suffix_df on tld column for each iteration.
            merged_df = input_df.merge(
                tmp_suffix_df, on=tld_col, how="left", suffixes=("", "_y")
            )
            col_pos = i - 1
            tld_r_col = "tld%s_y" % (str(col_pos))
            # Check for a right side column i.e, added to merged_df when join clause satisfies.
            if tld_r_col in merged_df.columns:
                temp_df = DataFrame()
                # Retrieve records which satisfies join clause.
                joined_recs_df = merged_df[merged_df[tld_r_col].isna() == False]
                temp_df["hostname"] = joined_recs_df["tld0"]
                temp_df["domain"] = joined_recs_df[col_pos]
                temp_df["suffix"] = joined_recs_df[tld_r_col]
                # Assigning values to construct subdomain at the end.
                if col_pos == 0:
                    temp_df["sd_prefix"] = ""
                else:
                    temp_df["sd_prefix"] = joined_recs_df[0]
                if col_pos > 1:
                    temp_df["sd_suffix"] = joined_recs_df[1]
                else:
                    temp_df["sd_suffix"] = ""
                # Concat current iteration result to previous iteration result.
                output_df = cudf.concat([temp_df, output_df])
                # Assigning unprocessed records to input_df for next stage of processing.
                input_df = merged_df[merged_df[tld_r_col].isna()]
        return output_df

    def __create_output_df(self):
        """
        Create cuDF dataframe with set of predefined columns.
        """
        output_df = cudf.DataFrame(
            [
                (col, "")
                for col in ["domain", "suffix", "sd_prefix", "sd_suffix", "hostname"]
            ]
        )
        return output_df

    def __verify_req_cols(self, req_cols):
        """
        Verify user requested columns against allowed output columns.
        """
        if req_cols is not None:
            if not set(req_cols).issubset(set(self.__output_cols)):
                raise ValueError(
                    "Given req_cols must be subset of %s" % (self.__output_cols)
                )
        else:
            req_cols = self.__output_cols
        return req_cols

    def parse_url(self, url_df_col, req_cols=None):
        """
        This function extracts subdomain, domain and suffix for a given url.
        returns: cuDF dataframe with requested columns. If req_cols values are passed as input parameter.
        Example:- 
            requested cols: 
                ["hostname", "domain", "suffix", "subdomain"]
            input:
                                            url
                0 http://forums.news.cnn.com.ac/
                1             forums.news.cnn.ac
                2               ftp://b.cnn.com/
        
            output:    
                                  hostname      domain        suffix    subdomain
                0   forums.news.cnn.com.ac         cnn        com.ac  forums.news
                1       forums.news.cnn.ac         cnn            ac  forums.news
                2                b.cnn.com         cnn           com            b
        """
        req_cols = self.__verify_req_cols(req_cols)
        hostnames = self.extract_hostnames(url_df_col)
        log.info("Extracting hostnames is successfully completed.")
        hostname_split_df = self.__get_hostname_split_df(hostnames)
        col_len = len(hostname_split_df.columns) - 1
        log.info("Generating tld columns...")
        hostname_split_df = self.__generate_tld_cols(
            hostname_split_df, hostnames, col_len
        )
        log.info("Successfully generated tld columns.")
        output_df = self.__create_output_df()
        log.info("Extracting tld...")
        output_df = self.__extract_tld(hostname_split_df, col_len, output_df)
        log.info("Extracting tld is successfully completed.")
        cleaned_output_df = self.__clean_output_df(output_df, req_cols)
        return cleaned_output_df

    def __clean_output_df(self, output_df, req_cols):
        """
        Example:- 
            requested cols: 
                ["hostname", "domain", "suffix", "subdomain"]
            input:
                                  hostname      domain        suffix  sd_prefix  sd_suffix
                0   forums.news.cnn.com.ac         cnn        com.ac     forums       news
                1       forums.news.cnn.ac         cnn            ac     forums       news
                2                b.cnn.com         cnn           com          b           
        
            output:    
                                  hostname      domain        suffix    subdomain
                0   forums.news.cnn.com.ac         cnn        com.ac  forums.news
                1       forums.news.cnn.ac         cnn            ac  forums.news
                2                b.cnn.com         cnn           com            b
        """
        clean_output_df = DataFrame()
        # reset index, since output_df is the result of multiple temp_df contactination.
        output_df = output_df.reset_index(drop=True)
        # Remove empty record i.e, added while creating dataframe.
        output_df = output_df[:-1]
        # If required Concat sd_prefix and sd_suffix columns to generate subdomain.
        if "subdomain" in req_cols:
            output_df["subdomain"] = (
                output_df["sd_prefix"]
                .str.cat(output_df["sd_suffix"], sep=".")
                .str.rstrip(".")
            )
        clean_output_df = output_df[req_cols]
        return clean_output_df
