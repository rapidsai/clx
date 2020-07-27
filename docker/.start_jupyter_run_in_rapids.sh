#!/bin/bash
. /opt/conda/etc/profile.d/conda.sh
conda activate rapids
/rapids/utils/start_jupyter.sh > /dev/null
echo "Notebook server successfully started!"
echo "To access visit http://localhost:8888 on your host machine."
echo 'Ensure the following arguments to "docker run" are added to expose the server ports to your host machine:
   -p 8888:8888 -p 8787:8787 -p 8786:8786'
exec "$@" 
