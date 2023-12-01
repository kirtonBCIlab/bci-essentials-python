from .sources import MarkerSource, EegSource


class LslEegSource(EegSource):
    def __init__(self):
        print("hello")


class LslMarkerSource(MarkerSource):
    def __init__(self):
        print("hello")
