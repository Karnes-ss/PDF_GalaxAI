


Software Test Report

Project：PDF_GalaxAI
Version: v1.1.1

 
Test Overview

This test report covers additional issues and newly requested features identified during subsequent testing of the PDF_GalaxAI. The evaluation focuses on missing functionalities, UI responsiveness, keyword extraction quality, and proposed enhancements.
Test Environment
Operating Systems: Windows 11
Browsers: Edge
Screen Resolutions: 1920×1080
Testing Method: Manual testing

 
Issues Summary and Details

1. Dialog Box Typing (WASD) Inappropriately Moves the Nebula Graph
Description: When the user types the WASD keys inside the dialog/chat input box, the nebula graph view is incorrectly moved/panned, which interferes with normal text input and operation.
Steps to Reproduce:
1.Open the dialog input box.
2.Type any content using the WASD keys.
3.Observe that the nebula graph moves unexpectedly.
Expected Result: Input in the dialog box should only affect text typing; WASD should not control the nebula view.
Actual Result: Pressing WASD in the dialog moves the nebula graph incorrectly.

2. Distant Nodes Appear Too Far Away When Switching Previews
Description: When switching previews between nodes that are far apart in the nebula graph, the popup preview position is too far from the current viewport, making it inconvenient to view content and locate nodes.
Steps to Reproduce:
1.Upload a PDF and generate multiple distant nodes.
2.Switch previews between nodes that are far from each other.
3.Observe the popup position of the preview panel.
Expected Result: Preview popup should appear near the current view or target node.
Actual Result: Preview jumps to an overly distant position.

3. Close Button of Preview Box Is Misaligned; Long File Names Cannot Wrap Properly
Description: The close button of the preview panel is offset and poorly positioned. When the file name is too long (connected with symbols), it cannot wrap automatically and overflows the container.
Steps to Reproduce:
1.Open a PDF preview panel.
2.Check the position of the close button.
3.View a file with an extra-long name connected by symbols.
Expected Result: Close button should be at a standard position; long file names should wrap normally.
Actual Result: Close button is deviated; long file names cannot wrap correctly.

4. Spaces Between Words Sometimes Disappear in Preview Box
Description: In the content preview panel, spaces between some words occasionally disappear, causing words to stick together and reducing readability.
Steps to Reproduce:
1.Open the content preview of a PDF.
2.Check the text spacing in the preview area.
Expected Result: Spaces between words should be preserved as in the original PDF.
Actual Result: Some spaces between words disappear randomly.