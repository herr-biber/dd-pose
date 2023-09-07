#!/usr/bin/env python
import tempfile
import plotly.express as px

fig = px.scatter(px.data.iris(), x="sepal_length", y="sepal_width", color="species")
with tempfile.TemporaryFile(suffix=".png") as fp:
    fig.write_image(fp)
