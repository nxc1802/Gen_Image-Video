from factory.core.model_adapter import Engine


class SpeechEngine(Engine):
    engine="speech"

    def run(self,request):
        return self.execute(request,request.capability)

    def transcribe(self,request):
        return self.execute(request,"speech.asr")

    def synthesize(self,request):
        return self.execute(request,"speech.tts")
