from factory.core.model_adapter import Engine


class MusicEngine(Engine):
    engine="music"

    def run(self,request): return self.execute(request,request.capability)
