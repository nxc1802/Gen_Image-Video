from factory.core.model_adapter import Engine


class EnhancementEngine(Engine):
    engine="enhancement"

    def run(self,request): return self.execute(request,request.capability)
