#!/bin/sh
# Launch FiftyOne App. If FIFTYONE_DATASET_NAME is set, open with that dataset.
# --address 0.0.0.0 and -p 5151:5151 per https://docs.voxel51.com/installation/environments.html#docker
set -e
if [ -n "$FIFTYONE_DATASET_NAME" ]; then
  exec uv run fiftyone app launch "$FIFTYONE_DATASET_NAME" --remote --port 5151 --address 0.0.0.0
else
  exec uv run fiftyone app launch --remote --port 5151 --address 0.0.0.0
fi
