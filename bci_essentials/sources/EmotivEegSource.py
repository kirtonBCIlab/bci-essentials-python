from pylsl import StreamInfo, FOREVER

from bci_essentials.io.lsl_sources import LslEegSource


class EmotivEegSource(LslEegSource):
    def __init__(self, stream: StreamInfo = None, timeout: float = FOREVER):
        super().__init__(stream, timeout)

    @property
    def channel_labels(self) -> list[str]:
        return self.get_channel_properties("label")
