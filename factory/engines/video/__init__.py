from factory.core.model_adapter import Engine


class VideoEngine(Engine):
    engine="video"

    def run(self,request): return self.execute(request,request.capability)
