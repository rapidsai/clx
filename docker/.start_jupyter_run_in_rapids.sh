#!/bin/bash
source activate clx
# echo "source activate clx" >> ~/.bashrc
/clx/docker/start_jupyter.sh > /dev/null
exec "$@"
