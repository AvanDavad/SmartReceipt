## phase 0

`CNNModulePhase0Points` is the model that identifies the 4 corner points on the image.

To train the model:

```bash
python -m src.models.phase0_points.run_training --from_scratch
```

To infere the model (from a video stream):

```bash
python -m src.models.phase0_points.run_inference
```

To infere the model (from input image):

```bash
python -m src.models.phase0_points.run_inference_img -i image.jpg
```

To infere the model (from input video):

```bash
python -m src.models.phase0_points.run_inference_video -i vid.mp4
```

