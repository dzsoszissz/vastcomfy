class GetAudioDurationSeconds:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"audio": ("AUDIO",)}}

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("duration_seconds",)
    FUNCTION = "get_duration"
    CATEGORY = "audio"

    def get_duration(self, audio):
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        duration = float(waveform.shape[-1]) / float(sample_rate)
        return (duration,)


NODE_CLASS_MAPPINGS = {
    "GetAudioDurationSeconds": GetAudioDurationSeconds,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "GetAudioDurationSeconds": "Get Audio Duration (seconds)",
}
