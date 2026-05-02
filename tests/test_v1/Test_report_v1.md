


Software Test Report

Project：PDF_GalaxAI
Version: DEMO

 
Test Overview

The evaluation includes PDF upload handling, nebula visualization effects, responsive layout, operational interaction constraints, and PDF directory synchronization.
Test Environment
Operating Systems: Windows 11
Browsers: Edge
Screen Resolutions: 1920×1080
Testing Method: Manual testing
 
Issues Summary and Details
1. Inconsistent Node Connections When Re‑uploading the Same PDF
Description: When the same PDF file is uploaded multiple times, the number of connections between nodes varies between uploads, affecting graph stability. 
Steps to Reproduce:
1.Upload the same PDF twice.
2.Observe the number and distribution of node connections in the generated graph.
3.Compare the results of the two uploads.
Expected Result: The node graph should be identical for the same file.
Actual Result: The number of connections differs.
Screenshots: Screenshot(Issue_1), Screenshot(Issue_1_2)
  

2. Nebula Range Issue: Pixel Blocks Obstruct View
Description: In the simulated nebula view, some pixel blocks are too large or poorly positioned, covering key nodes or content and hindering visualization.
Steps to Reproduce:
1.Enter the nebula view.
2.Examine the distribution of pixel blocks.
3.Check if any critical areas are obscured.
Expected Result: Pixel blocks should not cover important content.
Actual Result: Important areas are partially or fully blocked.
Screenshots: Screenshot(Issue_2)
 

3. Lack of Responsive UI Adaptation
Description: The page displays correctly on large screens, but when the window is resized to smaller dimensions, some elements overlap, misalign, or overflow, indicating poor responsiveness.
Steps to Reproduce:
1.Shrink the browser window from 1920×1080 to 1280×720.
2.Observe the layout of UI components.
Expected Result: The interface should adapt smoothly to different screen sizes.
Actual Result: Layout breaks – buttons misplace, content overlaps.
Screenshot: Screenshot(Issue_3), Screenshot(Issue_3_2)
  

4. No Scroll Wheel Limit
Description: In scrollable areas, the user can scroll indefinitely using the mouse wheel, leading to excessive blank space or visual confusion beyond the actual content.
Steps to Reproduce:
1.Navigate to a scrollable region.
2.Continuously scroll with the mouse wheel.
3.Observe whether scrolling goes beyond the content boundaries.
Expected Result: Scrolling should be constrained within the content area.
Actual Result: Scrolling continues past the content, creating empty space.
Screenshots: Screenshot(Issue_4)
 

5. PDF Edit Results Not Synchronized in Directory
Description: After editing a PDF and saving, the directory view still shows the previous version instead of the updated content.
Steps to Reproduce:
1.Edit a PDF and save changes.
2.Navigate to the directory/list view.
3.Check the displayed content for the edited file.
Expected Result: The directory should reflect the latest edited version.
Actual Result: The old version is shown.
Screenshots: Screenshot(Issue_5), Screenshot(Issue_5_2)
 
 

Pending Features (To Be Tested)
The following features are not yet implemented and will be covered in future test cycles:
1.AI interface integration and logic
2.Search functionality and result display
3.Synchronization with node graphs and directory
