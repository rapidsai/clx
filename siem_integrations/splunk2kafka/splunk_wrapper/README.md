# splunk_wrapper

## Overview

Wrapper script to handle switching Python versions so start/stop/restart
commands work as expected from init.d and web UI

## Pre-reqs

1. Install Miniconda2 (https://repo.continuum.io/miniconda/) in $SPLUNKHOME as the splunk user
```
    sudo -i -u splunk bash
    Add path to ~/.bashrc
```

2. Backup the splunk python executable in `/opt/splunk/bin`
```
    mv /opt/splunk/bin/python2.7 $SPLUNKHOME/bin/python2.7.splunk
```

3. Create symlink to Miniconda Python in `/opt/splunk/bin`
```
    ln -s /opt/splunk/miniconda2/bin/python2.7 /opt/splunk/python2.7.conda
```

## Install

**NOTE:** Do not run this script twice as it will remove Splunk's Python, this
is a WIP to fix

Run `sudo bash wrapper-install.sh`