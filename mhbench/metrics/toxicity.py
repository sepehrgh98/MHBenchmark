from detoxify import Detoxify

class ToxicityScorer:
    def __init__(self, model="original"):
        self.model = Detoxify(model)

    def score(self, texts):
        out = self.model.predict(texts)
        # return mean toxicity; you can expose all keys if you want
        tox = out.get("toxicity", None)
        return float(sum(tox)/len(tox)) if tox is not None else None
