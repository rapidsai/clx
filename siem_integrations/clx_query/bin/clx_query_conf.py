import splunk.admin as admin
import splunk.entity as en

"""
Copyright (C) 2005 - 2010 Splunk Inc. All Rights Reserved.
Description:  This skeleton python script handles the parameters in the configuration page.

      handleList method: lists configurable parameters in the configuration page
      corresponds to handleractions = list in restmap.conf

      handleEdit method: controls the parameters and saves the values 
      corresponds to handleractions = edit in restmap.conf

"""


class ConfigApp(admin.MConfigHandler):
    """
    Set up supported arguments
    """

    def setup(self):
        if self.requestedAction == admin.ACTION_EDIT:
            for arg in ["clx_hostname", "clx_port"]:
                self.supportedArgs.addOptArg(arg)

    """                                                           
    Reads configuration from the custom file clx/default/clx_query_setup.conf.                
    """

    def handleList(self, confInfo):
        confDict = self.readConf("clx_query_setup")
        if None != confDict:
            for stanza, settings in confDict.items():
                for key, val in settings.items():
                    confInfo[stanza].append(key, val)

    def handleEdit(self, confInfo):
        name = self.callerArgs.id
        args = self.callerArgs

        self.writeConf("clx_query_setup", "setupentity", self.callerArgs.data)


# initialize the handler
admin.init(ConfigApp, admin.CONTEXT_NONE)
