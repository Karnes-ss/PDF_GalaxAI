


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

1. Lack of Node Removal Functionality
Description: The application does not provide a way to remove individual nodes from the generated node graph. Once nodes are created, users cannot delete or hide them, limiting graph customization.
Steps to Reproduce:
1.Upload a PDF and generate the node graph.
2.Attempt to delete or remove a specific node (e.g., right‑click or look for a delete option).
3.Observe that no removal action is available.
Expected Result: Users should be able to remove unwanted nodes to clean up the graph.
Actual Result: No node removal functionality exists.

2.Lack of Search Functionality
Description: The application currently lacks any search feature, making it difficult to locate specific PDFs, nodes, or content within the directory or graph views.
Actual Result: No search capability exists.

3. Keyword Selection Includes Common Stop Words
Description: The keyword extraction or highlighting feature frequently selects common stop words such as "and", "the", "of" instead of meaningful content words, reducing the usefulness of keyword displays.
Expected Result: Keywords should be meaningful content words, with stop words filtered out.
Actual Result: Stop words are frequently selected, cluttering the keyword view.
Screenshot:
  

4. Punctuation Missing Occasionally in Preview
Description: In the content preview area, punctuation marks (such as periods, commas) are occasionally missing, or words are incorrectly concatenated without spaces, affecting readability.
Expected Result: Punctuation and spacing should be preserved as in the original PDF.
Actual Result: Some punctuation marks are missing, or words are merged (e.g., "GoalsA" instead of "Goals A" or "Goals.").
Screenshots: As shown above.

5. Lack of Adaptation for Long Titles
Description: When a PDF has a very long title, the UI does not properly handle it (e.g., no truncation, wrapping, or font scaling), causing the title to overflow its container, overlap with other elements, or be cut off.
Expected Result: Long titles should be truncated with ellipsis, wrap to multiple lines, or the font size should adjust to fit the container.
Actual Result: The title overflows the designated area, breaking the layout or becoming unreadable.
Screenshots: As shown above.
 
New Features Implemented and Tested
The following features have been recently added to the application and were verified during this test cycle.

File Name Tooltip on Hover
When hovering over a truncated file name, a tooltip displays the full file name.
Test Result: Pass – Tooltip appears correctly and shows the complete file name.

Nebula Visualization Optimization
Improved rendering of the nebula view to reduce occlusion and enhance clarity. Background now has dynamic effects
Test Result: Pass – Pixel blocks are better positioned; key nodes are no longer obscured.
 
Content Preview
A preview panel shows a quick view of PDF content without fully opening the file.
Test Result: Pass – Preview loads correctly and displays the first page of the PDF.

Keyword Display Area
A dedicated area now shows extracted keywords from the PDF, with stop words filtered out.
Test Result: Pass – Keywords are relevant; common stop words are excluded.
 
Pending Features (To Be Tested)
The following features are not yet implemented and will be covered in future test cycles:
1.AI interface integration and logic
2.Search functionality and result display
3.Node removal functionality
