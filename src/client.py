import flagon


class Client(flagon.Client):
    def __init__(self, data, create_model_fn, seed=None):
        self.data = data
        self.model = create_model_fn(seed)

    def fit(self, parameters, config):
        self.model.set_parameters(parameters)
        metrics = self.model.step(
            self.data['train']['X'],
            self.data['train']['Y'],
            epochs=config['num_epochs'],
            steps_per_epoch=config.get("num_steps"),
            verbose=0
        )
        return self.model.get_parameters(), len(self.data['train']), metrics

    def evaluate(self, parameters, config):
        self.model.set_parameters(parameters)
        metrics = self.model.evaluate(self.data['test']['X'], self.data['test']['Y'], verbose=0)
        return len(self.data['test']), metrics