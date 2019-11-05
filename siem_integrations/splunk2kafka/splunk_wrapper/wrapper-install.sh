#!/bin/bash
sudo -i

SPLUNKHOME=/opt/splunk

cd $SPLUNKHOME/bin
mv splunk splunk.splunk
cat <<EOF > splunk.wrapper
#!/bin/bash

RETVAL=0

switch_python_splunk() {
  echo Switching Python to Splunk distro...
  rm -f $SPLUNKHOME/bin/python2.7
  cp -a $SPLUNKHOME/bin/python2.7.splunk $SPLUNKHOME/bin/python2.7
}
switch_python_conda() {
  echo Switching Python to Miniconda distro...
  rm -f $SPLUNKHOME/bin/python2.7
  cp -a $SPLUNKHOME/bin/python2.7.conda $SPLUNKHOME/bin/python2.7
}

switch_python_splunk
sleep 1
$SPLUNKHOME/bin/splunk.splunk \$@
RETVAL=\$?
sleep 5
switch_python_conda

exit \$RETVAL
EOF
chmod 755 splunk.wrapper
chown splunk:splunk splunk.wrapper 
ln -s splunk.wrapper splunk