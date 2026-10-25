# Outstanding Work
AI Agentic Instruction: 

- [x] if the mask or caption checkboxes are unchecked, the mask or captions are still created. Confirm all checkboxes stop the related process from happening and does not affect the rest of the processing.
- [x] thread manual cropping of images so multiple crops can be performed, currently if a 2nd one is created during the current crop, the 2nd crop it is not performed.
- [x] Crop tab, pressing D key moves to the next image, pressing A key moves to previous image.
- [] upgrade imagesharp library to latest version
- [] Restyle the application to use fully MaterialDesign with MaterialDesignInXamlToolkit library
- [x] Settings tab, add option to append captions to current captions files and to overwrite existing masks or output images
- [x] When detecting concepts, take the strongest detection from the cropped images of the same image and do not emit the rest, if all equal values take the middle image, e.g. the 2nd of 3 crops of the same image.
- [x] Config tab, add a setting for the minimum resolution that a subject can be cropped to.
