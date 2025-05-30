# Project Structure:

| Path                                                          | Description                                                                |
| :------------------------------------------------------------ | :------------------------------------------------------------------------- |
| &ensp;&ensp;&boxvr;&nbsp; batch_gen.py                        | Main file. Starts Batch image generation or classification                 |
| &ensp;&ensp;&boxur;&nbsp; ft_utils                            | My code                                                                    |
| &ensp;&ensp;&ensp;&ensp;&boxvr;&nbsp; BatchImageClassifier.py | Class for generating and loading generated image with / from StyleGAN2-ada |
| &ensp;&ensp;&ensp;&ensp;&boxvr;&nbsp; BatchImageGenerator.py  | Class for classifying images in batches using OpenCLIP                     |
| &ensp;&ensp;&ensp;&ensp;&boxur;&nbsp; utils.py                | Utility functions, e.g. handling filenames                                 |
