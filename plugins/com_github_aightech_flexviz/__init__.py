"""
KiCad Flex Viewer - Visualize folded flex PCBs in 3D

This plugin provides action buttons in KiCad PCB Editor:
1. Create Fold: Define fold lines using two-point placement
2. Open Viewer: Launch 3D visualization of the folded PCB
"""

__version__ = "1.0.0"
__author__ = "Aightech"

try:
    from .plugin import CreateFoldAction, OpenViewerAction

    # Register all action plugins
    CreateFoldAction().register()
    OpenViewerAction().register()

except Exception as e:
    # Log error and register dummy plugin to notify user
    import os
    plugin_dir = os.path.dirname(os.path.realpath(__file__))
    log_file = os.path.join(plugin_dir, 'flex_viewer_error.log')
    with open(log_file, 'w') as f:
        import traceback
        f.write(f"Error: {repr(e)}\n\n")
        f.write(traceback.format_exc())

    # Register dummy plugin to show error
    import pcbnew
    import wx

    class FlexViewerError(pcbnew.ActionPlugin):
        def defaults(self):
            self.name = "Flex Viewer - ERROR"
            self.category = "Flex PCB"
            self.description = "Error loading Flex Viewer plugin"

        def Run(self):
            message = (
                "There was an error loading the Flex Viewer plugin.\n\n"
                f"Check the log file at:\n{log_file}\n\n"
                "Please report issues on GitHub."
            )
            wx.MessageBox(message, "Flex Viewer Error", wx.OK | wx.ICON_ERROR)

    FlexViewerError().register()
