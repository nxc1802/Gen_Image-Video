from factory.core.model_adapter import Engine


class ImageEngine(Engine):
    engine="image"

    def run(self,request): return self.execute(request,request.capability)
