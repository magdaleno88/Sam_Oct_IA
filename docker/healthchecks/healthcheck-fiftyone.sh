#!/bin/sh
# FiftyOne app responds on port 5151. Check that the root returns HTTP 200.
curl -sf http://127.0.0.1:5151/ || exit 1
