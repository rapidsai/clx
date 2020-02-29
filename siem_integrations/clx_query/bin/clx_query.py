#!/usr/bin/env python
import logging
import sys, requests, json
from splunklib.searchcommands import (
    dispatch,
    GeneratingCommand,
    Configuration,
    Option,
    validators,
)
import splunklib.client as client

log = logging.getLogger(__name__)


@Configuration()
class ClxQuery(GeneratingCommand):
    query = Option(require=True)

    def generate(self):
        configs = client.Configurations(self.service)
        for config in configs:
            if config.name == "clx_query_setup":
                clx_config = config.iter().next().content

        url = self.construct_url(clx_config, self.query)
        response = requests.get(url)

        if response.status_code != 200:
            yield {"ERROR": response.content}
        else:
            results = json.loads(json.loads(response.content))
            for result in results:
                yield result

    def construct_url(self, config, query):
        base_url = "http://%s:%s/clxquery" % (
            config["clx_hostname"],
            config["clx_port"],
        )
        url = base_url + "/" + query
        return url


dispatch(ClxQuery, sys.argv, sys.stdin, sys.stdout, __name__)
