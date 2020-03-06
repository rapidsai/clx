import re
import os
import logging

from clxquery import utils
from clxquery.blazingsql_helper import BlazingSQLHelper
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt


log = logging.getLogger(__name__)

file_path = os.environ.get("BLZ_READER_CONF")

# Load tables configuration
config = utils.load_yaml(file_path)
configured_tables = set([table["table_name"] for table in config["tables"]])

REGEX_PATTERN = r"main.([\w]+)"

blz_helper = BlazingSQLHelper()


@csrf_exempt
def run_query(request, query):
    if request.method == "GET":
        # Check for the list of tables used in the query to prevent loading other tables into gpu memory
        query_tables = set(re.findall(REGEX_PATTERN, query))
        # Verify list of tables used in the query to make sure they are included in the configuration file
        if query_tables.issubset(configured_tables):
            try:
                query_config = {}
                query_config["tables"] = []
                for table in config["tables"]:
                    if table["table_name"] in query_tables:
                        query_config["tables"].append(table)
                query_config["sql"] = query
                # Run query and get the results
                df = blz_helper.run_query(query_config)
                # Drop tables to free up memory
                blz_helper.drop_table(query_tables)
                # Convert cudf to pandas dataframe
                df = df.to_pandas()
                # Convert results to json format.
                results = df.to_json(orient="records")
                response = JsonResponse(results, safe=False)
            except Exception as e:
                stacktrace = str(e)
                log.error("Error executing query: %s" % (stacktrace))
                response = JsonResponse(
                    {"status": "false", "message": stacktrace}, status=500, safe=False
                )
        else:
            message = (
                "One or more tables used in the query are not available in the server configuration. Please select from this list  %s or add new tables to your clx-blazingsql configuration."
                % (configured_tables)
            )
            response = JsonResponse(
                {"status": "false", "message": message}, status=404, safe=False
            )
        return response
