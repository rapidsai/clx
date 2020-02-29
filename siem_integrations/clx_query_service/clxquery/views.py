import re
import os
import logging

from clxquery import utils
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from clx.io.reader.blazingsql_reader import BlazingSQLReader

log = logging.getLogger(__name__)

file_path = os.environ.get("BLZ_READER_CONF")

# Load tables configuration
config = utils.load_yaml(file_path)
configured_tables = set([table["table_name"] for table in config["tables"]])

REGEX_PATTERN = r"main.([\w]+)"


@csrf_exempt
def run_query(request, query):
    if request.method == "GET":
        # Check for the list of tables used in the query to prevent loading any other tables into gpu memory
        tables = set(re.findall(REGEX_PATTERN, query))
        # Verify list of tables used in the query are included to configuration
        if tables.issubset(configured_tables):
            query_config = {}
            query_config["tables"] = []
            for table in config["tables"]:
                if table["table_name"] in tables:
                    query_config["tables"].append(table)
            query_config["sql"] = query
            blz_reader = BlazingSQLReader(query_config)

            # Run query and get the results
            df = blz_reader.fetch_data()
            df = df.to_pandas()
            # Convert results to json format.
            results = df.to_json(orient="records")
            response = JsonResponse(results, safe=False)
        else:
            message = (
                "One or more tables used in the query are not available in the server configuration. Please select from this list  %s or add new tables to your clx-blazingsql configuration."
                % (configured_tables)
            )
            response = JsonResponse(
                {"status": "false", "message": message}, status=404, safe=False
            )
        return response
